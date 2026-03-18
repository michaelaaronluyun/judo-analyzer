"""
Shared in-memory state for the Judo Analyzer API.

IMPORTANT — Vercel serverless functions:
  Each function invocation may run in a fresh container OR reuse a warm one.
  State here is best-effort in-memory. For production persistence across cold
  starts you would move classifier.event_log and heatmap_tracker state into
  a database (e.g. MongoDB Atlas — you already have one connected).

For the current academic prototype this warm-container behaviour is fine:
  within a session the container stays warm and state accumulates normally.
"""
from lib.classifier import JudoTechniqueClassifier
from lib.heatmap_tracker import ContactHeatmapTracker

classifier = JudoTechniqueClassifier()
heatmap_tracker = ContactHeatmapTracker()
