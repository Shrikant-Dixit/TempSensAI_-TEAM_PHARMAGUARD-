from flask import Blueprint, redirect, url_for
import os

upload_bp = Blueprint("upload", __name__)

UPLOAD_FOLDER = "test/"

@upload_bp.route("/upload", methods=["GET"])
def upload():
    # Hard-coded test values
    filename = "Gardasil9_test.csv"  
    medicine = "Gardasil9"

    filepath = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(filepath):
        return f"Error: File {filename} not found in test folder."

    # Redirect to predict route with chosen file + medicine
    return redirect(url_for("predict.predict", file=filepath, medicine=medicine))
