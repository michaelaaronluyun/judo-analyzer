import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flask import Flask, request, jsonify, make_response
import itsdangerous
from pymongo import MongoClient
from flask_bcrypt import Bcrypt

SECRET_KEY  = os.environ.get("SECRET_KEY", "dev-secret-change-me")
MONGO_URI   = os.environ.get("MONGO_URI", "")
COOKIE_NAME = "judo_session"

signer = itsdangerous.URLSafeSerializer(SECRET_KEY)
app    = Flask(__name__)
bcrypt = Bcrypt(app)

_db = None
def get_db():
    global _db
    if _db is None and MONGO_URI:
        client = MongoClient(MONGO_URI)
        _db = client["judo_analyzer"]
    return _db

def make_token(payload):
    return signer.dumps(payload)

def read_token():
    raw = request.cookies.get(COOKIE_NAME, "")
    if not raw:
        return {}
    try:
        return signer.loads(raw)
    except Exception:
        return {}

@app.route("/", methods=["GET", "POST"])
@app.route("/api/auth", methods=["GET", "POST"])
def auth():
    action = request.args.get("action", "")

    if action == "me":
        sess = read_token()
        if not sess:
            return jsonify({"error": "not logged in"}), 401
        return jsonify({"email": sess.get("email", ""), "username": sess.get("username", "")})

    if action == "logout":
        resp = make_response(("", 302))
        resp.headers["Location"] = "/"
        resp.delete_cookie(COOKIE_NAME)
        return resp

    if request.method != "POST":
        return jsonify({"error": "Method not allowed"}), 405

    data = request.get_json(silent=True) or {}
    db   = get_db()
    if db is None:
        return jsonify({"error": "Database not configured"}), 500

    users_col = db["users"]
    users_col.create_index("email", unique=True)

    if action == "register":
        for field in ("username", "email", "password"):
            if not data.get(field):
                return jsonify({"error": f"Missing field: {field}"}), 400
        if len(data["password"]) < 6:
            return jsonify({"error": "Password must be at least 6 characters"}), 400
        if users_col.find_one({"email": data["email"]}):
            return jsonify({"error": "Email already registered"}), 409
        pw_hash = bcrypt.generate_password_hash(data["password"]).decode("utf-8")
        users_col.insert_one({"username": data["username"], "email": data["email"], "password": pw_hash})
        token = make_token({"email": data["email"], "username": data["username"]})
        resp  = make_response(jsonify({"message": "Account created", "username": data["username"]}), 201)
        resp.set_cookie(COOKIE_NAME, token, max_age=86400, httponly=True, samesite="Lax")
        return resp

    if action == "login":
        user = users_col.find_one({"email": data.get("email", "")})
        if user and bcrypt.check_password_hash(user["password"], data.get("password", "")):
            token = make_token({"email": user["email"], "username": user["username"]})
            resp  = make_response(jsonify({"message": "OK", "username": user["username"]}))
            resp.set_cookie(COOKIE_NAME, token, max_age=86400, httponly=True, samesite="Lax")
            return resp
        return jsonify({"error": "Invalid email or password"}), 401

    return jsonify({"error": f"Unknown action: {action}"}), 400
