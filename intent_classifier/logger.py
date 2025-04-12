# logger.py
import csv
import os
from datetime import datetime

LOG_FILE = "logs/query_log.csv"
os.makedirs("logs", exist_ok=True)

def log_query(query, predicted_intent, confidence, resolved, user_feedback=None):
    timestamp = datetime.now().isoformat()
    row = [timestamp, query, predicted_intent, confidence, resolved, user_feedback]
    headers = ["timestamp", "query", "predicted_intent", "confidence", "resolved", "user_feedback"]

    file_exists = os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(headers)
        writer.writerow(row)
