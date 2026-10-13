# from flask import Blueprint, redirect, url_for
# import os

# upload_bp = Blueprint("upload", __name__)

# UPLOAD_FOLDER = "test/"

# @upload_bp.route("/upload", methods=["GET"])
# def upload():
#     # Hard-coded test values
#     filename = "Gardasil9_test.csv"  
#     medicine = "Gardasil9"

#     filepath = os.path.join(UPLOAD_FOLDER, filename)

#     if not os.path.exists(filepath):
#         return f"Error: File {filename} not found in test folder."

#     # Redirect to predict route with chosen file + medicine
#     return redirect(url_for("predict.predict", file=filepath, medicine=medicine))

from flask import Blueprint, redirect, url_for
import os
import csv
import requests   # TinyFish API call

upload_bp = Blueprint("upload", __name__)

UPLOAD_FOLDER = "test/"

# TinyFish Agent API details
TINYFISH_API_URL = "https://agent.tinyfish.ai/v1/automation/run"
TINYFISH_API_KEY = os.getenv("TINYFISH_API_KEY", "sk-tinyfish-nxYITikL4Bh2EV_7P9TXEFb1uGstqXRi")
WAREHOUSE_EMAIL = "shrikantdixit36@gmail.com"

def check_csv_for_nulls(filepath, threshold=10):
    """Check if there are threshold consecutive empty rows in Temperature or MedicineName."""
    null_count = 0
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row['Temperature'] or not row['MedicineName']:
                null_count += 1
            else:
                null_count = 0
            if null_count >= threshold:
                return True
    return False

def send_alert():
    """Send Gmail alert via TinyFish Agent API and log result."""
    headers = {"X-API-Key": TINYFISH_API_KEY}
    payload = {
        "goal": f"Send email via Gmail to {WAREHOUSE_EMAIL} "
                f"with subject: ALERT and body: Hardware stopped logging data. Multiple NULL values detected in CSV.",
        "url": "https://agent.tinyfish.ai/"   # context URL
    }
    try:
        resp = requests.post(TINYFISH_API_URL, json=payload, headers=headers)
        print("TinyFish API raw response:", resp.status_code, resp.text)

        resp_json = resp.json()
        result_text = resp_json.get("result", {}).get("result", "")

        print("=== TinyFish Agent Execution ===")
        print("Result of running TinyFish:", result_text)

        if "cannot log in to Gmail" in result_text or "No credentials" in result_text:
            print("❌ Gmail alert failed: TinyFish has no Gmail credentials linked.")
        elif "sent" in result_text.lower():
            print("✅ Gmail alert sent successfully via TinyFish.")
        else:
            print("⚠️ TinyFish completed but did not confirm Gmail delivery.")

    except Exception as e:
        print("Error calling TinyFish API:", e)

@upload_bp.route("/upload", methods=["GET"])
def upload():
    filename = "dataTesting.csv"  
    medicine = "dataTesting"

    filepath = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(filepath):
        return f"Error: File {filename} not found in test folder."

    if check_csv_for_nulls(filepath, threshold=10):
        send_alert()
        return "Error: Hardware failure detected. Gmail alert attempted via TinyFish."

    return redirect(url_for("predict.predict", file=filepath, medicine=medicine))
