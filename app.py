"""
Local development server.
Run with:  python app.py
Then open: http://localhost:5000

This is NOT deployed to Vercel — it's for local testing only.
Vercel uses the api/*.py serverless functions instead.
"""
import os
import json
from flask import Flask, request, jsonify, session, redirect, Response, send_file

from lib.classifier import JudoTechniqueClassifier
from lib.heatmap_tracker import ContactHeatmapTracker

import numpy as np
from pymongo import MongoClient
from flask_bcrypt import Bcrypt
import secrets

# ── Load env from .env.local if present ──────────────────────────────────────
try:
    with open(".env.local") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
except FileNotFoundError:
    pass

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="public", static_url_path="")
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(24))
bcrypt = Bcrypt(app)

MONGO_URI = os.environ.get("MONGO_URI", "")
db = None
if MONGO_URI:
    client = MongoClient(MONGO_URI)
    db = client["judo_analyzer"]

classifier      = JudoTechniqueClassifier()
heatmap_tracker = ContactHeatmapTracker()

# ── Static pages ──────────────────────────────────────────────────────────────
@app.route("/")
def home():
    if session.get("user"):
        return redirect("/app")
    return send_file("public/index.html")

@app.route("/app")
def analyzer_app():
    if not session.get("user"):
        return redirect("/")
    return send_file("public/app.html")

# ── Auth routes (mirrors api/auth.py) ─────────────────────────────────────────
@app.route("/api/auth")
@app.route("/api/auth", methods=["POST"])
def auth():
    action = request.args.get("action", "")

    if action == "me":
        if not session.get("user"):
            return jsonify({"error": "not logged in"}), 401
        return jsonify({"email": session["user"], "username": session.get("username", "")})

    if action == "logout":
        session.clear()
        return redirect("/")

    data = request.get_json() or {}
    if db is None:
        return jsonify({"error": "Database not configured — set MONGO_URI in .env.local"}), 500
    users_col = db["users"]

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
        session["user"] = data["email"]
        session["username"] = data["username"]
        return jsonify({"message": "Account created", "username": data["username"]}), 201

    if action == "login":
        user = users_col.find_one({"email": data.get("email", "")})
        if user and bcrypt.check_password_hash(user["password"], data.get("password", "")):
            session["user"] = user["email"]
            session["username"] = user["username"]
            return jsonify({"message": "OK", "username": user["username"]})
        return jsonify({"error": "Invalid email or password"}), 401

    return jsonify({"error": f"Unknown action: {action}"}), 400

# ── Analysis routes ────────────────────────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
def analyze():
    data       = request.get_json()
    poses      = data.get("poses", {})
    video_time = data.get("videoTime", 0)
    clean = {int(aid): {int(k): v for k, v in kps.items()} for aid, kps in poses.items()}
    results = {"athletes": {}, "interaction": False, "interaction_confidence": 0.0}
    all_grip = []
    aids = sorted(clean.keys())[:2]
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

@app.route("/api/export-csv")
def export_csv():
    return Response(
        classifier.export_csv(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=judo_events.csv"},
    )

@app.route("/api/heatmap")
def get_heatmap():
    return jsonify(heatmap_tracker.get_heatmap_data())

@app.route("/api/heatmap/reset", methods=["POST"])
def reset_heatmap():
    heatmap_tracker.reset()
    return jsonify({"ok": True})

# ── Run ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Local dev server starting at http://localhost:5000")
    app.run(port=5000, debug=True)
