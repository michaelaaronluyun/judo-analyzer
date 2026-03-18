"""
GET  /api/heatmap        — returns current heatmap data for both athletes
POST /api/heatmap/reset  — clears heatmap state
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.state import heatmap_tracker


def handler(event, context):
    method = event.get("httpMethod", "GET")
    path   = event.get("path", "")

    # POST /api/heatmap/reset
    if method == "POST" and path.endswith("/reset"):
        heatmap_tracker.reset()
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"ok": True}),
        }

    # GET /api/heatmap
    if method == "GET":
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(heatmap_tracker.get_heatmap_data()),
        }

    return {"statusCode": 405, "body": json.dumps({"error": "Method not allowed"})}
