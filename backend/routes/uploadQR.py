from flask import Blueprint, request, redirect, url_for
import os

uploadQR_bp = Blueprint("uploadQR", __name__)

UPLOAD_FOLDER = "test/"

@uploadQR_bp.route("/uploadQR", methods=["GET"])
def upload_qr():
    # Get file and medicine from query parameters
    file_param = request.args.get("file")
    medicine = request.args.get("medicine")

    if not file_param or not medicine:
        return "Error: file and medicine must be provided."

    # If it's a Blob URL, use it directly
    if file_param.startswith("http"):
        filepath = file_param
    else:
        # Otherwise assume it's a local file in test/
        filepath = os.path.join(UPLOAD_FOLDER, file_param)
        if not os.path.exists(filepath):
            return f"Error: File {file_param} not found in test folder."

    # Redirect to predictQR route with chosen file + medicine
    return redirect(url_for("predictQR.predict_qr", file=filepath, medicine=medicine))
