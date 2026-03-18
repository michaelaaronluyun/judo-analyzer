"""
POST /api/analyze
Body: { poses: { "0": { kpIndex: {x,y,s}, ... }, "1": {...} }, videoTime: float }
Returns athlete analysis, interaction state, recent events.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from lib.state import classifier, heatmap_tracker


def handler(event, context):
    if event.get("httpMethod") != "POST":
        return {"statusCode": 405, "body": json.dumps({"error": "Method not allowed"})}

    try:
        data = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return {"statusCode": 400, "body": json.dumps({"error": "Invalid JSON"})}

    poses      = data.get("poses", {})
    video_time = data.get("videoTime", 0)

    # Keypoint indices arrive as strings from JSON — cast to int
    clean = {
        int(aid): {int(k): v for k, v in kps.items()}
        for aid, kps in poses.items()
    }

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

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(results),
    }
