"""
Contact heatmap tracker.
Extracted from DLSU Judo Analyzer notebook (cell 17).
"""

ZONE_MAP = {
    0:  {"name": "Head",       "kp": 0,  "cx": 0.500, "cy": 0.060},
    1:  {"name": "L Shoulder", "kp": 5,  "cx": 0.330, "cy": 0.195},
    2:  {"name": "R Shoulder", "kp": 6,  "cx": 0.670, "cy": 0.195},
    3:  {"name": "L Elbow",    "kp": 7,  "cx": 0.215, "cy": 0.360},
    4:  {"name": "R Elbow",    "kp": 8,  "cx": 0.785, "cy": 0.360},
    5:  {"name": "L Wrist",    "kp": 9,  "cx": 0.145, "cy": 0.510},
    6:  {"name": "R Wrist",    "kp": 10, "cx": 0.855, "cy": 0.510},
    7:  {"name": "Chest/Core", "kp": -1, "cx": 0.500, "cy": 0.270},
    8:  {"name": "L Hip",      "kp": 11, "cx": 0.395, "cy": 0.490},
    9:  {"name": "R Hip",      "kp": 12, "cx": 0.605, "cy": 0.490},
    10: {"name": "L Knee",     "kp": 13, "cx": 0.390, "cy": 0.680},
    11: {"name": "R Knee",     "kp": 14, "cx": 0.610, "cy": 0.680},
    12: {"name": "L Ankle",    "kp": 15, "cx": 0.390, "cy": 0.875},
    13: {"name": "R Ankle",    "kp": 16, "cx": 0.610, "cy": 0.875},
}


class ContactHeatmapTracker:
    CONTACT_THRESH = 0.12
    DECAY = 0.995

    def __init__(self):
        self.heat = {0: {z: 0.0 for z in ZONE_MAP}, 1: {z: 0.0 for z in ZONE_MAP}}
        self.total_contacts = {0: 0, 1: 0}

    def _get_kp(self, kps, idx):
        if idx < 0:
            l = kps.get(5)
            r = kps.get(6)
            if l and r:
                return {"x": (l["x"] + r["x"]) / 2, "y": (l["y"] + r["y"]) / 2}
            return None
        return kps.get(idx)

    def update(self, kps0, kps1):
        for victim_id, victim_kps, attacker_kps in [(0, kps0, kps1), (1, kps1, kps0)]:
            if not victim_kps or not attacker_kps:
                continue
            attacker_pts = [
                attacker_kps.get(9),
                attacker_kps.get(10),
                attacker_kps.get(7),
                attacker_kps.get(8),
            ]
            attacker_pts = [w for w in attacker_pts if w]
            for zone_id, zone in ZONE_MAP.items():
                victim_pt = self._get_kp(victim_kps, zone["kp"])
                if not victim_pt:
                    continue
                for ap in attacker_pts:
                    d = ((victim_pt["x"] - ap["x"]) ** 2 + (victim_pt["y"] - ap["y"]) ** 2) ** 0.5
                    if d < self.CONTACT_THRESH:
                        strength = (1.0 - d / self.CONTACT_THRESH) ** 2
                        self.heat[victim_id][zone_id] = min(
                            self.heat[victim_id][zone_id] + strength * 0.15, 10.0
                        )
                        self.total_contacts[victim_id] += 1
                        break
            for z in ZONE_MAP:
                self.heat[victim_id][z] *= self.DECAY

    def get_heatmap_data(self):
        result = {}
        for aid in [0, 1]:
            max_heat = max(self.heat[aid].values()) or 1.0
            zones = []
            for zone_id, zone in ZONE_MAP.items():
                zones.append(
                    {
                        "id": zone_id,
                        "name": zone["name"],
                        "cx": zone["cx"],
                        "cy": zone["cy"],
                        "heat": round(self.heat[aid][zone_id], 3),
                        "norm": round(self.heat[aid][zone_id] / max_heat, 3),
                    }
                )
            result[aid] = {
                "zones": zones,
                "total_contacts": self.total_contacts[aid],
                "max_heat": round(max_heat, 3),
            }
        return result

    def reset(self):
        self.__init__()
