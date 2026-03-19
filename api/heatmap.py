import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from flask import Flask, jsonify
from lib.state import heatmap_tracker

app = Flask(__name__)

@app.route("/", methods=["GET"])
@app.route("/api/heatmap", methods=["GET"])
def get_heatmap():
    return jsonify(heatmap_tracker.get_heatmap_data())

@app.route("/reset", methods=["POST"])
@app.route("/api/heatmap/reset", methods=["POST"])
def reset_heatmap():
    heatmap_tracker.reset()
    return jsonify({"ok": True})
