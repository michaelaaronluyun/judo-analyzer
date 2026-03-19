import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flask import Flask, Response
from lib.state import classifier

app = Flask(__name__)

@app.route("/", methods=["GET"])
@app.route("/api/export-csv", methods=["GET"])
def export_csv():
    return Response(
        classifier.export_csv(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=judo_events.csv"},
    )
