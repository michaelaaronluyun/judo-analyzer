"""
Judo technique classifier with grip detection and balance disruption analysis.
Extracted from DLSU Judo Analyzer notebook (cell 17).
"""
import time
import numpy as np
import pandas as pd
from collections import deque, defaultdict


class JudoTechniqueClassifier:
    HISTORY_LEN = 60

    def __init__(self):
        self.histories = defaultdict(lambda: deque(maxlen=self.HISTORY_LEN))
        self.event_log = []
        self._last_technique = {}

    @staticmethod
    def _angle(a, b, c):
        try:
            va = np.array([a["x"] - b["x"], a["y"] - b["y"]])
            vc = np.array([c["x"] - b["x"], c["y"] - b["y"]])
            n1, n2 = np.linalg.norm(va), np.linalg.norm(vc)
            if n1 == 0 or n2 == 0:
                return 0.0
            return float(np.degrees(np.arccos(np.clip(np.dot(va, vc) / (n1 * n2), -1, 1))))
        except Exception:
            return 0.0

    @staticmethod
    def _com(kps):
        pts = [[kps[i]["x"], kps[i]["y"]] for i in [5, 6, 11, 12] if i in kps]
        return np.mean(pts, axis=0) if pts else None

    def _angles(self, kps):
        a = {}
        if all(i in kps for i in [5, 7, 9]):
            a["left_arm"] = self._angle(kps[5], kps[7], kps[9])
        if all(i in kps for i in [6, 8, 10]):
            a["right_arm"] = self._angle(kps[6], kps[8], kps[10])
        if all(i in kps for i in [11, 13, 15]):
            a["left_leg"] = self._angle(kps[11], kps[13], kps[15])
        if all(i in kps for i in [12, 14, 16]):
            a["right_leg"] = self._angle(kps[12], kps[14], kps[16])
        if 5 in kps and 11 in kps:
            dx = kps[5]["x"] - kps[11]["x"]
            dy = kps[5]["y"] - kps[11]["y"]
            a["torso_lean"] = float(np.degrees(np.arctan2(abs(dx), abs(dy))))
        if 11 in kps and 12 in kps:
            a["hip_y"] = (kps[11]["y"] + kps[12]["y"]) / 2
        if 5 in kps and 6 in kps:
            a["shoulder_y"] = (kps[5]["y"] + kps[6]["y"]) / 2
        return a

    @staticmethod
    def detect_grip(kps1, kps2):
        THRESH = 0.10
        score, pts = 0.0, []
        for w_idx in [9, 10]:
            if w_idx not in kps1:
                continue
            w = kps1[w_idx]
            for t_idx in [5, 6, 7, 8, 11, 12]:
                if t_idx not in kps2:
                    continue
                t = kps2[t_idx]
                d = ((w["x"] - t["x"]) ** 2 + (w["y"] - t["y"]) ** 2) ** 0.5
                if d < THRESH:
                    score += 1 - d / THRESH
                    pts.append({"x": (w["x"] + t["x"]) / 2, "y": (w["y"] + t["y"]) / 2})
        for w_idx in [9, 10]:
            if w_idx not in kps2:
                continue
            w = kps2[w_idx]
            for t_idx in [5, 6, 7, 8, 11, 12]:
                if t_idx not in kps1:
                    continue
                t = kps1[t_idx]
                d = ((w["x"] - t["x"]) ** 2 + (w["y"] - t["y"]) ** 2) ** 0.5
                if d < THRESH:
                    score += 1 - d / THRESH
        grip_strength = min(score / 4.0, 1.0)
        return {
            "gripping": grip_strength > 0.15,
            "grip_strength": round(grip_strength, 3),
            "grip_points": pts,
        }

    def balance_disruption(self, athlete_id, kps):
        hist = self.histories[athlete_id]
        if len(hist) < 10:
            return 0.0
        coms = []
        for frame in list(hist)[-10:]:
            c = self._com(frame["kps"])
            if c is not None:
                coms.append(c)
        if len(coms) < 5:
            return 0.0
        coms = np.array(coms)
        accels = np.diff(np.diff(coms, axis=0), axis=0)
        return min(float(np.mean(np.linalg.norm(accels, axis=1))) * 20, 1.0)

    def classify(self, athlete_id, kps, opponent_kps=None, video_time=0):
        ts = time.time()
        angles = self._angles(kps)
        self.histories[athlete_id].append({"kps": kps, "angles": angles, "ts": ts})
        hist = self.histories[athlete_id]
        balance = self.balance_disruption(athlete_id, kps)

        grip_info = {"gripping": False, "grip_strength": 0.0, "grip_points": []}
        if opponent_kps:
            grip_info = self.detect_grip(kps, opponent_kps)

        technique = "Standing / Moving"
        confidence = 0.40
        description = "Normal movement"

        hip_y = angles.get("hip_y", 0.5)
        left_leg = angles.get("left_leg", 180)
        right_leg = angles.get("right_leg", 180)
        left_arm = angles.get("left_arm", 180)
        right_arm = angles.get("right_arm", 180)
        torso_lean = angles.get("torso_lean", 0)
        shoulder_y = angles.get("shoulder_y", 0.3)
        avg_leg = (left_leg + right_leg) / 2
        avg_arm = (left_arm + right_arm) / 2

        if avg_leg < 130 and torso_lean > 20 and avg_arm < 130:
            technique, confidence = (
                ("Ippon Seoi Nage", 0.70) if left_arm < right_arm else ("Seoi Nage", 0.72)
            )
            description = "Low entry — shoulder throw setup"
        elif hip_y < 0.55 and shoulder_y < 0.40 and avg_arm < 150 and torso_lean > 15:
            technique, confidence = "O Goshi", 0.65
            description = "Hip contact — hip throw"
        elif (right_leg > 155 or left_leg > 155) and torso_lean > 10 and len(hist) > 10:
            hip_ys = [f["angles"].get("hip_y", 0.5) for f in list(hist)[-10:]]
            if np.std(hip_ys) > 0.015:
                technique, confidence = "Harai Goshi", 0.63
                description = "Hip sweep throw"

        if avg_arm < 120 and (left_leg > 160 or right_leg > 160) and torso_lean > 12:
            technique, confidence = "Tai Otoshi", 0.62
            description = "Body drop — leg block throw"

        if len(hist) > 10:
            r_ankles = [f["kps"][16]["y"] for f in list(hist)[-12:] if 16 in f["kps"]]
            if r_ankles and (max(r_ankles) - min(r_ankles)) > 0.13:
                technique, confidence = "Osoto Gari", 0.68
                description = "Major outer reap"
        if len(hist) > 8:
            l_ankles = [f["kps"][15]["y"] for f in list(hist)[-8:] if 15 in f["kps"]]
            if l_ankles and 0.04 < (max(l_ankles) - min(l_ankles)) < 0.09 and hip_y > 0.55:
                technique, confidence = "Ko Uchi Gari", 0.60
                description = "Minor inner reap"

        if hip_y > 0.72:
            technique, confidence = "Ne-Waza (Ground)", 0.75
            description = "Ground work / grappling"
        if balance > 0.5 and technique == "Standing / Moving":
            technique, confidence = "Transition", 0.55
            description = "High CoM acceleration — kuzushi phase"

        prev = self._last_technique.get(athlete_id, "")
        if technique != prev and technique != "Standing / Moving":
            self.event_log.append(
                {
                    "ts": ts,
                    "athlete": athlete_id,
                    "technique": technique,
                    "confidence": confidence,
                    "videoTime": video_time,
                }
            )
        self._last_technique[athlete_id] = technique

        return {
            "technique": technique,
            "confidence": round(confidence, 3),
            "description": description,
            "angles": {k: round(v, 1) for k, v in angles.items()},
            "balance_disruption": round(balance, 3),
            "grip": grip_info,
            "com": self._com(kps).tolist() if self._com(kps) is not None else None,
        }

    def export_csv(self):
        return pd.DataFrame(self.event_log).to_csv(index=False)
