"""
/api/auth  — handles all auth operations via ?action= query param.

  POST /api/auth?action=register   { username, email, password }
  POST /api/auth?action=login      { email, password }
  GET  /api/auth?action=logout
  GET  /api/auth?action=me

Session is stored in a signed cookie via itsdangerous.  The SECRET_KEY env var
must be set in Vercel project settings (Settings → Environment Variables).
MongoDB URI must be set as MONGO_URI.
"""
import json
import os
import sys

# Make lib importable from the repo root when Vercel runs api/auth.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from http.cookies import SimpleCookie
import itsdangerous
from pymongo import MongoClient
from flask_bcrypt import generate_password_hash, check_password_hash

# ── Config ────────────────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-me")
MONGO_URI   = os.environ.get("MONGO_URI", "")
COOKIE_NAME = "judo_session"

signer = itsdangerous.URLSafeSerializer(SECRET_KEY)

# ── DB (lazy init — reused across warm invocations) ───────────────────────────
_db = None

def get_db():
    global _db
    if _db is None and MONGO_URI:
        client = MongoClient(MONGO_URI)
        _db = client["judo_analyzer"]
    return _db


# ── Cookie helpers ────────────────────────────────────────────────────────────
def make_session_cookie(payload: dict) -> str:
    token = signer.dumps(payload)
    return f"{COOKIE_NAME}={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age=86400"


def read_session(event) -> dict:
    raw = (event.get("headers") or {}).get("cookie", "")
    c = SimpleCookie()
    c.load(raw)
    if COOKIE_NAME not in c:
        return {}
    try:
        return signer.loads(c[COOKIE_NAME].value)
    except Exception:
        return {}


# ── Response helpers ──────────────────────────────────────────────────────────
def ok(body: dict, extra_headers: dict = None):
    headers = {"Content-Type": "application/json"}
    if extra_headers:
        headers.update(extra_headers)
    return {"statusCode": 200, "headers": headers, "body": json.dumps(body)}


def err(body: dict, status: int = 400):
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(body),
    }


# ── Handler ───────────────────────────────────────────────────────────────────
def handler(event, context):
    action = (event.get("queryStringParameters") or {}).get("action", "")
    method = event.get("httpMethod", "GET")

    # ── /me ───────────────────────────────────────────────────────────────────
    if action == "me":
        sess = read_session(event)
        if not sess:
            return err({"error": "not logged in"}, 401)
        return ok({"email": sess.get("email", ""), "username": sess.get("username", "")})

    # ── /logout ───────────────────────────────────────────────────────────────
    if action == "logout":
        clear_cookie = f"{COOKIE_NAME}=; Path=/; Max-Age=0"
        return {
            "statusCode": 302,
            "headers": {"Location": "/", "Set-Cookie": clear_cookie},
            "body": "",
        }

    # ── POST only below ───────────────────────────────────────────────────────
    if method != "POST":
        return err({"error": "Method not allowed"}, 405)

    try:
        data = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return err({"error": "Invalid JSON"})

    db = get_db()
    if db is None:
        return err({"error": "Database not configured — set MONGO_URI env var"}, 500)

    users_col = db["users"]

    # ── /register ─────────────────────────────────────────────────────────────
    if action == "register":
        for field in ("username", "email", "password"):
            if not data.get(field):
                return err({"error": f"Missing field: {field}"})
        if len(data["password"]) < 6:
            return err({"error": "Password must be at least 6 characters"})
        if users_col.find_one({"email": data["email"]}):
            return err({"error": "Email already registered"}, 409)
        pw_hash = generate_password_hash(data["password"]).decode("utf-8")
        users_col.insert_one(
            {"username": data["username"], "email": data["email"], "password": pw_hash}
        )
        cookie = make_session_cookie({"email": data["email"], "username": data["username"]})
        return ok({"message": "Account created", "username": data["username"]},
                  {"Set-Cookie": cookie})

    # ── /login ────────────────────────────────────────────────────────────────
    if action == "login":
        user = users_col.find_one({"email": data.get("email", "")})
        if user and check_password_hash(user["password"], data.get("password", "")):
            cookie = make_session_cookie({"email": user["email"], "username": user["username"]})
            return ok({"message": "OK", "username": user["username"]}, {"Set-Cookie": cookie})
        return err({"error": "Invalid email or password"}, 401)

    return err({"error": f"Unknown action: {action}"})
