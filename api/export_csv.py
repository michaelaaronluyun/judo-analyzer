"""
GET /api/export-csv — returns the event log as a downloadable CSV file.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.state import classifier


def handler(event, context):
    if event.get("httpMethod") != "GET":
        return {"statusCode": 405, "body": json.dumps({"error": "Method not allowed"})}

    csv_data = classifier.export_csv()
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "text/csv; charset=utf-8",
            "Content-Disposition": "attachment; filename=judo_events.csv",
        },
        "body": csv_data,
    }
