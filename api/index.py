"""
DLSU Judo Analyzer — single-file Vercel entrypoint.
All logic inlined: no cross-directory imports needed.
"""
import os, sys, time
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from flask import Flask, request, jsonify, make_response, Response
import itsdangerous
from pymongo import MongoClient
from flask_bcrypt import Bcrypt

# ── App setup ─────────────────────────────────────────────────────────────────
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
        _db = MongoClient(MONGO_URI)["judo_analyzer"]
    return _db

def make_token(payload):
    return signer.dumps(payload)

def read_token():
    raw = request.cookies.get(COOKIE_NAME, "")
    try:
        return signer.loads(raw) if raw else {}
    except Exception:
        return {}

# ── Kalman tracker (pure numpy) ───────────────────────────────────────────────
class KalmanTrack:
    def __init__(self, pos):
        self.x = np.array([pos[0], pos[1], 0.0, 0.0], dtype=float)
        dt = 1.0
        self.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]], dtype=float)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
        self.P = np.eye(4) * 0.1
        self.Q = np.eye(4) * 0.001
        self.R = np.eye(2) * 0.01
        self.missed = 0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.missed += 1
        return self.x[:2]

    def update(self, z):
        z = np.array(z, dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.missed = 0
        return self.x[:2]

    @property
    def position(self):
        return self.x[:2]

# ── Technique classifier ──────────────────────────────────────────────────────
class JudoTechniqueClassifier:
    HISTORY_LEN = 60

    def __init__(self):
        self.histories = defaultdict(lambda: deque(maxlen=self.HISTORY_LEN))
        self.event_log = []
        self._last = {}

    @staticmethod
    def _angle(a, b, c):
        try:
            va = np.array([a["x"]-b["x"], a["y"]-b["y"]])
            vc = np.array([c["x"]-b["x"], c["y"]-b["y"]])
            n1, n2 = np.linalg.norm(va), np.linalg.norm(vc)
            if n1 == 0 or n2 == 0:
                return 0.0
            return float(np.degrees(np.arccos(np.clip(np.dot(va,vc)/(n1*n2), -1, 1))))
        except Exception:
            return 0.0

    @staticmethod
    def _com(kps):
        pts = [[kps[i]["x"], kps[i]["y"]] for i in [5,6,11,12] if i in kps]
        return np.mean(pts, axis=0) if pts else None

    def _angles(self, kps):
        a = {}
        if all(i in kps for i in [5,7,9]):    a["left_arm"]   = self._angle(kps[5],kps[7],kps[9])
        if all(i in kps for i in [6,8,10]):   a["right_arm"]  = self._angle(kps[6],kps[8],kps[10])
        if all(i in kps for i in [11,13,15]): a["left_leg"]   = self._angle(kps[11],kps[13],kps[15])
        if all(i in kps for i in [12,14,16]): a["right_leg"]  = self._angle(kps[12],kps[14],kps[16])
        if 5 in kps and 11 in kps:
            dx = kps[5]["x"]-kps[11]["x"]; dy = kps[5]["y"]-kps[11]["y"]
            a["torso_lean"] = float(np.degrees(np.arctan2(abs(dx), abs(dy))))
        if 11 in kps and 12 in kps:
            a["hip_y"] = (kps[11]["y"]+kps[12]["y"]) / 2
        if 5 in kps and 6 in kps:
            a["shoulder_y"] = (kps[5]["y"]+kps[6]["y"]) / 2
        return a

    @staticmethod
    def detect_grip(kps1, kps2):
        THRESH = 0.10; score = 0; pts = []
        for w in [9, 10]:
            if w not in kps1: continue
            for t in [5,6,7,8,11,12]:
                if t not in kps2: continue
                d = ((kps1[w]["x"]-kps2[t]["x"])**2 + (kps1[w]["y"]-kps2[t]["y"])**2)**0.5
                if d < THRESH:
                    score += (1 - d/THRESH)
                    pts.append({"x":(kps1[w]["x"]+kps2[t]["x"])/2, "y":(kps1[w]["y"]+kps2[t]["y"])/2})
        for w in [9, 10]:
            if w not in kps2: continue
            for t in [5,6,7,8,11,12]:
                if t not in kps1: continue
                d = ((kps2[w]["x"]-kps1[t]["x"])**2 + (kps2[w]["y"]-kps1[t]["y"])**2)**0.5
                if d < THRESH:
                    score += (1 - d/THRESH)
        gs = min(score/4.0, 1.0)
        return {"gripping": gs > 0.15, "grip_strength": round(gs,3), "grip_points": pts}

    def balance_disruption(self, aid, kps):
        hist = self.histories[aid]
        if len(hist) < 10: return 0.0
        coms = [self._com(f["kps"]) for f in list(hist)[-10:]]
        coms = [c for c in coms if c is not None]
        if len(coms) < 5: return 0.0
        coms = np.array(coms)
        accels = np.diff(np.diff(coms, axis=0), axis=0)
        return min(float(np.mean(np.linalg.norm(accels, axis=1))) * 20, 1.0)

    def classify(self, aid, kps, opp_kps=None, video_time=0):
        ts = time.time()
        angles = self._angles(kps)
        self.histories[aid].append({"kps": kps, "angles": angles, "ts": ts})
        hist = self.histories[aid]
        balance = self.balance_disruption(aid, kps)
        grip = self.detect_grip(kps, opp_kps) if opp_kps else {"gripping":False,"grip_strength":0.0,"grip_points":[]}

        tech = "Standing / Moving"; conf = 0.40; desc = "Normal movement"
        hip_y = angles.get("hip_y", 0.5)
        ll    = angles.get("left_leg", 180);   rl = angles.get("right_leg", 180)
        la    = angles.get("left_arm", 180);   ra = angles.get("right_arm", 180)
        tl    = angles.get("torso_lean", 0);   sy = angles.get("shoulder_y", 0.3)
        al    = (ll+rl)/2;                     aa = (la+ra)/2

        if al < 130 and tl > 20 and aa < 130:
            tech, conf = ("Ippon Seoi Nage", 0.70) if la < ra else ("Seoi Nage", 0.72)
            desc = "Low entry — shoulder throw"
        elif hip_y < 0.55 and sy < 0.40 and aa < 150 and tl > 15:
            tech, conf = "O Goshi", 0.65; desc = "Hip throw"
        elif (rl > 155 or ll > 155) and tl > 10 and len(hist) > 10:
            hys = [f["angles"].get("hip_y", 0.5) for f in list(hist)[-10:]]
            if np.std(hys) > 0.015:
                tech, conf = "Harai Goshi", 0.63; desc = "Hip sweep"
        if aa < 120 and (ll > 160 or rl > 160) and tl > 12:
            tech, conf = "Tai Otoshi", 0.62; desc = "Body drop"
        if len(hist) > 10:
            r_ank = [f["kps"][16]["y"] for f in list(hist)[-12:] if 16 in f["kps"]]
            if r_ank and (max(r_ank)-min(r_ank)) > 0.13:
                tech, conf = "Osoto Gari", 0.68; desc = "Major outer reap"
        if len(hist) > 8:
            l_ank = [f["kps"][15]["y"] for f in list(hist)[-8:] if 15 in f["kps"]]
            if l_ank and 0.04 < (max(l_ank)-min(l_ank)) < 0.09 and hip_y > 0.55:
                tech, conf = "Ko Uchi Gari", 0.60; desc = "Minor inner reap"
        if hip_y > 0.72:
            tech, conf = "Ne-Waza (Ground)", 0.75; desc = "Ground work"
        if balance > 0.5 and tech == "Standing / Moving":
            tech, conf = "Transition", 0.55; desc = "High CoM acceleration"

        prev = self._last.get(aid, "")
        if tech != prev and tech != "Standing / Moving":
            self.event_log.append({"ts":ts,"athlete":aid,"technique":tech,"confidence":conf,"videoTime":video_time})
        self._last[aid] = tech

        return {
            "technique": tech, "confidence": round(conf,3), "description": desc,
            "angles": {k:round(v,1) for k,v in angles.items()},
            "balance_disruption": round(balance,3), "grip": grip,
            "com": self._com(kps).tolist() if self._com(kps) is not None else None,
        }

    def export_csv(self):
        return pd.DataFrame(self.event_log).to_csv(index=False)

# ── Heatmap tracker ───────────────────────────────────────────────────────────
ZONE_MAP = {
    0:{"name":"Head","kp":0,"cx":0.500,"cy":0.060},
    1:{"name":"L Shoulder","kp":5,"cx":0.330,"cy":0.195},
    2:{"name":"R Shoulder","kp":6,"cx":0.670,"cy":0.195},
    3:{"name":"L Elbow","kp":7,"cx":0.215,"cy":0.360},
    4:{"name":"R Elbow","kp":8,"cx":0.785,"cy":0.360},
    5:{"name":"L Wrist","kp":9,"cx":0.145,"cy":0.510},
    6:{"name":"R Wrist","kp":10,"cx":0.855,"cy":0.510},
    7:{"name":"Chest/Core","kp":-1,"cx":0.500,"cy":0.270},
    8:{"name":"L Hip","kp":11,"cx":0.395,"cy":0.490},
    9:{"name":"R Hip","kp":12,"cx":0.605,"cy":0.490},
    10:{"name":"L Knee","kp":13,"cx":0.390,"cy":0.680},
    11:{"name":"R Knee","kp":14,"cx":0.610,"cy":0.680},
    12:{"name":"L Ankle","kp":15,"cx":0.390,"cy":0.875},
    13:{"name":"R Ankle","kp":16,"cx":0.610,"cy":0.875},
}

class ContactHeatmapTracker:
    CONTACT_THRESH = 0.12; DECAY = 0.995

    def __init__(self):
        self.heat = {0:{z:0.0 for z in ZONE_MAP}, 1:{z:0.0 for z in ZONE_MAP}}
        self.total_contacts = {0:0, 1:0}

    def _get_kp(self, kps, idx):
        if idx < 0:
            l = kps.get(5); r = kps.get(6)
            return {"x":(l["x"]+r["x"])/2,"y":(l["y"]+r["y"])/2} if l and r else None
        return kps.get(idx)

    def update(self, kps0, kps1):
        for vid, vkps, akps in [(0,kps0,kps1),(1,kps1,kps0)]:
            if not vkps or not akps: continue
            apts = [akps.get(i) for i in [9,10,7,8] if akps.get(i)]
            for zid, zone in ZONE_MAP.items():
                vpt = self._get_kp(vkps, zone["kp"])
                if not vpt: continue
                for ap in apts:
                    d = ((vpt["x"]-ap["x"])**2+(vpt["y"]-ap["y"])**2)**0.5
                    if d < self.CONTACT_THRESH:
                        s = (1.0-d/self.CONTACT_THRESH)**2
                        self.heat[vid][zid] = min(self.heat[vid][zid]+s*0.15, 10.0)
                        self.total_contacts[vid] += 1
                        break
            for z in ZONE_MAP:
                self.heat[vid][z] *= self.DECAY

    def get_heatmap_data(self):
        result = {}
        for aid in [0,1]:
            mx = max(self.heat[aid].values()) or 1.0
            zones = [{"id":zid,"name":z["name"],"cx":z["cx"],"cy":z["cy"],
                      "heat":round(self.heat[aid][zid],3),"norm":round(self.heat[aid][zid]/mx,3)}
                     for zid,z in ZONE_MAP.items()]
            result[aid] = {"zones":zones,"total_contacts":self.total_contacts[aid],"max_heat":round(mx,3)}
        return result

    def reset(self):
        self.__init__()

# ── Shared state (warm-container persistence) ─────────────────────────────────
classifier      = JudoTechniqueClassifier()
heatmap_tracker = ContactHeatmapTracker()

# ── Routes: pages served statically by Vercel — no Flask routes needed ──────

# ── Routes: auth ──────────────────────────────────────────────────────────────
@app.route("/api/auth", methods=["GET", "POST"])
def auth():
    action = request.args.get("action", "")

    if action == "me":
        sess = read_token()
        if not sess: return jsonify({"error":"not logged in"}), 401
        return jsonify({"email":sess.get("email",""), "username":sess.get("username",""), "role":sess.get("role","athlete")})

    if action == "logout":
        r = make_response("", 302); r.headers["Location"] = "/"; r.delete_cookie(COOKIE_NAME); return r

    if request.method != "POST":
        return jsonify({"error":"Method not allowed"}), 405

    data = request.get_json(silent=True) or {}
    db   = get_db()
    if db is None:
        return jsonify({"error":"Database not configured — set MONGO_URI in Vercel env vars"}), 500

    col = db["users"]

    if action == "register":
        for f in ("username","email","password"):
            if not data.get(f): return jsonify({"error":f"Missing: {f}"}), 400
        if len(data["password"]) < 6: return jsonify({"error":"Password min 6 chars"}), 400
        if col.find_one({"email":data["email"]}): return jsonify({"error":"Email already registered"}), 409
        col.create_index("email", unique=True)
        col.insert_one({"username":data["username"],"email":data["email"],
                         "password":bcrypt.generate_password_hash(data["password"]).decode(),
                         "role": data.get("role","athlete")})
        token = make_token({"email":data["email"],"username":data["username"],"role":data.get("role","athlete")})
        r = make_response(jsonify({"message":"Account created","username":data["username"],"role":data.get("role","athlete")}), 201)
        r.set_cookie(COOKIE_NAME, token, max_age=86400, httponly=True, samesite="Lax"); return r

    if action == "login":
        user = col.find_one({"email":data.get("email","")})
        if user and bcrypt.check_password_hash(user["password"], data.get("password","")):
            role = user.get("role", "athlete")
            token = make_token({"email":user["email"],"username":user["username"],"role":role})
            r = make_response(jsonify({"message":"OK","username":user["username"],"role":role}))
            r.set_cookie(COOKIE_NAME, token, max_age=86400, httponly=True, samesite="Lax"); return r
        return jsonify({"error":"Invalid email or password"}), 401

    return jsonify({"error":f"Unknown action: {action}"}), 400

# ── Routes: analyze ───────────────────────────────────────────────────────────
@app.route("/api/analyze", methods=["POST"])
def analyze():
    data       = request.get_json(silent=True) or {}
    poses      = data.get("poses", {})
    video_time = data.get("videoTime", 0)
    clean      = {int(aid):{int(k):v for k,v in kps.items()} for aid,kps in poses.items()}
    results    = {"athletes":{}, "interaction":False, "interaction_confidence":0.0}
    all_grip   = []
    aids       = sorted(clean.keys())[:2]

    for i, aid in enumerate(aids):
        kps     = clean[aid]
        opp_kps = clean[aids[1-i]] if len(aids)==2 else None
        result  = classifier.classify(aid, kps, opp_kps, video_time=video_time)
        results["athletes"][aid] = result
        if result["grip"]["gripping"]:
            all_grip.append(result["grip"]["grip_strength"])

    if len(aids) == 2:
        c1 = classifier._com(clean[aids[0]])
        c2 = classifier._com(clean[aids[1]])
        if c1 is not None and c2 is not None:
            dist = float(np.linalg.norm(c1-c2))
            results["interaction"]            = dist < 0.22
            results["interaction_confidence"] = round(max(0.0, 1.0-dist/0.25), 3)
        heatmap_tracker.update(clean.get(aids[0],{}), clean.get(aids[1],{}))

    results["grip_active"]   = len(all_grip) > 0
    results["recent_events"] = classifier.event_log[-30:]
    results["total_events"]  = len(classifier.event_log)
    return jsonify(results)

# ── Routes: heatmap ───────────────────────────────────────────────────────────
@app.route("/api/heatmap", methods=["GET"])
def get_heatmap():
    return jsonify(heatmap_tracker.get_heatmap_data())

@app.route("/api/heatmap/reset", methods=["POST"])
def reset_heatmap():
    heatmap_tracker.reset()
    return jsonify({"ok":True})

# ── Routes: export ────────────────────────────────────────────────────────────
@app.route("/api/export-csv", methods=["GET"])
def export_csv():
    return Response(classifier.export_csv(), mimetype="text/csv",
                    headers={"Content-Disposition":"attachment; filename=judo_events.csv"})

# ── Helper: admin/coach guard ─────────────────────────────────────────────────
def require_admin():
    """Returns (sess, None) if admin/coach, or (None, error_response)."""
    sess = read_token()
    if not sess:
        return None, (jsonify({"error": "not logged in"}), 401)
    if sess.get("role") not in ("admin", "coach"):
        return None, (jsonify({"error": "Forbidden — admin or coach role required"}), 403)
    return sess, None

# ── Routes: user management (admin/coach only) ────────────────────────────────
@app.route("/api/users", methods=["GET"])
def list_users():
    sess, err = require_admin()
    if err: return err
    db = get_db()
    if db is None: return jsonify({"error": "DB not configured"}), 500
    users = list(db["users"].find({}, {"password": 0}))
    for u in users:
        u["_id"] = str(u["_id"])
        u.setdefault("role", "athlete")
    return jsonify(users)

@app.route("/api/users/<user_id>", methods=["PUT"])
def update_user(user_id):
    sess, err = require_admin()
    if err: return err
    data = request.get_json(silent=True) or {}
    db = get_db()
    if db is None: return jsonify({"error": "DB not configured"}), 500
    from bson import ObjectId
    allowed = {}
    if "username" in data: allowed["username"] = data["username"].strip()
    if "email"    in data: allowed["email"]    = data["email"].strip()
    if "role"     in data:
        if data["role"] not in ("admin", "coach", "athlete"):
            return jsonify({"error": "Invalid role"}), 400
        allowed["role"] = data["role"]
    if "password" in data and data["password"]:
        if len(data["password"]) < 6:
            return jsonify({"error": "Password min 6 chars"}), 400
        allowed["password"] = bcrypt.generate_password_hash(data["password"]).decode()
    if not allowed:
        return jsonify({"error": "No valid fields to update"}), 400
    result = db["users"].update_one({"_id": ObjectId(user_id)}, {"$set": allowed})
    if result.matched_count == 0:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"message": "User updated"})

@app.route("/api/users/<user_id>", methods=["DELETE"])
def delete_user(user_id):
    sess, err = require_admin()
    if err: return err
    # Prevent self-deletion
    if sess.get("email"):
        db = get_db()
        if db is None: return jsonify({"error": "DB not configured"}), 500
        from bson import ObjectId
        target = db["users"].find_one({"_id": ObjectId(user_id)}, {"email": 1})
        if target and target.get("email") == sess["email"]:
            return jsonify({"error": "Cannot delete your own account"}), 400
        result = db["users"].delete_one({"_id": ObjectId(user_id)})
        if result.deleted_count == 0:
            return jsonify({"error": "User not found"}), 404
        return jsonify({"message": "User deleted"})
    return jsonify({"error": "Session error"}), 400
