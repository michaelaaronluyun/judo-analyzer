"""
Vercel entrypoint — routes all requests to the correct handler.
Vercel looks for `app` in api/index.py as the Flask WSGI entrypoint.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, make_response, Response
import itsdangerous
from pymongo import MongoClient
from flask_bcrypt import Bcrypt
import numpy as np

from lib.state import classifier, heatmap_tracker

# ── Config ────────────────────────────────────────────────────────────────────
SECRET_KEY  = os.environ.get("SECRET_KEY", "dev-secret-change-me")
MONGO_URI   = os.environ.get("MONGO_URI", "")
COOKIE_NAME = "judo_session"

signer = itsdangerous.URLSafeSerializer(SECRET_KEY)
app    = Flask(__name__, static_folder=None)
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

# ── Auth ──────────────────────────────────────────────────────────────────────
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
        return jsonify({"error": "Database not configured — set MONGO_URI in Vercel env vars"}), 500

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

# ── Analyze ───────────────────────────────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
def analyze():
    data       = request.get_json(silent=True) or {}
    poses      = data.get("poses", {})
    video_time = data.get("videoTime", 0)

    clean = {int(aid): {int(k): v for k, v in kps.items()} for aid, kps in poses.items()}
    results  = {"athletes": {}, "interaction": False, "interaction_confidence": 0.0}
    all_grip = []
    aids     = sorted(clean.keys())[:2]

    for i, aid in enumerate(aids):
        kps     = clean[aid]
        opp_kps = clean[aids[1 - i]] if len(aids) == 2 else None
        result  = classifier.classify(aid, kps, opp_kps, video_time=video_time)
        results["athletes"][aid] = result
        if result["grip"]["gripping"]:
            all_grip.append(result["grip"]["grip_strength"])

    if len(aids) == 2:
        c1 = classifier._com(clean[aids[0]])
        c2 = classifier._com(clean[aids[1]])
        if c1 is not None and c2 is not None:
            dist = float(np.linalg.norm(c1 - c2))
            results["interaction"]            = dist < 0.22
            results["interaction_confidence"] = round(max(0.0, 1.0 - dist / 0.25), 3)
        heatmap_tracker.update(clean.get(aids[0], {}), clean.get(aids[1], {}))

    results["grip_active"]   = len(all_grip) > 0
    results["recent_events"] = classifier.event_log[-30:]
    results["total_events"]  = len(classifier.event_log)
    return jsonify(results)

# ── Heatmap ───────────────────────────────────────────────────────────────────
@app.route("/api/heatmap", methods=["GET"])
def get_heatmap():
    return jsonify(heatmap_tracker.get_heatmap_data())

@app.route("/api/heatmap/reset", methods=["POST"])
def reset_heatmap():
    heatmap_tracker.reset()
    return jsonify({"ok": True})

# ── Export CSV ────────────────────────────────────────────────────────────────
@app.route("/api/export-csv", methods=["GET"])
def export_csv():
    return Response(
        classifier.export_csv(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=judo_events.csv"},
    )

# ── Static pages ──────────────────────────────────────────────────────────────
@app.route("/")
def home():
    with open(os.path.join(os.path.dirname(__file__), "..", "public", "index.html")) as f:
        return f.read(), 200, {"Content-Type": "text/html"}

@app.route("/app")
def analyzer_app():
    sess = read_token()
    if not sess:
        resp = make_response(("", 302))
        resp.headers["Location"] = "/"
        return resp
    with open(os.path.join(os.path.dirname(__file__), "..", "public", "app.html")) as f:
        return f.read(), 200, {"Content-Type": "text/html"}
