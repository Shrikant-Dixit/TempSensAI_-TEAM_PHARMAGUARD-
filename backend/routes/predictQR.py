from flask import Blueprint, request, render_template
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
import io, base64, os

predictQR_bp = Blueprint("predictQR", __name__)

MODEL_PATH = "models/gradient_boosting.pkl"

# Medicine mapping
medicine_map = {
    "Comirnaty": 0,
    "Lantus(insulin glargine)": 1,
    "Humira": 2,
    "Gardasil9": 3,
    "HepatitisB": 4
}

def preprocess_test_csv(file_path):
    # Read CSV from Blob URL or local path
    if file_path.startswith("http"):
        df = pd.read_csv(file_path)
        filename = file_path.split("/")[-1]
    else:
        df = pd.read_csv(file_path)
        filename = os.path.basename(file_path)

    # Deviation from optimal range
    df["Deviation"] = df.apply(
        lambda row: 0 if row["OptimalMin"] <= row["Temperature"] <= row["OptimalMax"]
        else min(abs(row["Temperature"] - row["OptimalMin"]),
                 abs(row["Temperature"] - row["OptimalMax"])),
        axis=1
    )

    # Outside range flag
    df["OutsideRange"] = df.apply(
        lambda row: 1 if not (row["OptimalMin"] <= row["Temperature"] <= row["OptimalMax"]) else 0,
        axis=1
    )

    # Feature calculations
    avg_temp_deviation = df["Deviation"].mean()
    max_temp_deviation = df["Deviation"].max()
    bad_temp_duration = df["OutsideRange"].sum() * 30 / 3600.0  # hours
    time_above_25 = (df["Temperature"] > 25).sum() * 30 / 3600.0
    time_below_0 = (df["Temperature"] < 0).sum() * 30 / 3600.0
    temp_variance = df["Temperature"].var()

    medicine_type = medicine_map.get(df["MedicineName"].iloc[0], -1)

    features = {
        "File": filename,
        "MedicineName": df["MedicineName"].iloc[0],
        "MedicineCode": medicine_type,
        "AvgTempDeviation": avg_temp_deviation,
        "MaxTempDeviation": max_temp_deviation,
        "BadTempDurationHours": bad_temp_duration,
        "TimeAbove25C": time_above_25,
        "TimeBelow0C": time_below_0,
        "TemperatureVariance": temp_variance
    }

    return features, df

@predictQR_bp.route("/predictQR", methods=["GET"])
def predict_qr():
    file_path = request.args.get("file")
    medicine = request.args.get("medicine")

    if not file_path or not medicine:
        return "Error: file path and medicine name must be provided."

    # Preprocess CSV
    features, df = preprocess_test_csv(file_path)

    # Load model
    model = joblib.load(MODEL_PATH)

    # Build DataFrame with correct feature names
    X = pd.DataFrame([[
        features["MedicineCode"],
        features["AvgTempDeviation"],
        features["MaxTempDeviation"],
        features["BadTempDurationHours"],
        features["TimeAbove25C"],
        features["TimeBelow0C"],
        features["TemperatureVariance"]
    ]], columns=["MedicineCode", "AvgTempDeviation", "MaxTempDeviation",
                 "BadTempDurationHours", "TimeAbove25C", "TimeBelow0C", "TemperatureVariance"])

    # Predict potency loss
    predicted_loss = model.predict(X)[0]

    # Downsample: take every 50th row
    df_downsampled = df.iloc[::50, :]

    # Combine Date + Timestamp into full datetime
    df_downsampled["DateTime"] = pd.to_datetime(
           df_downsampled["Date"] + " " + df_downsampled["Timestamp"],
           format="%m/%d/%Y %H:%M:%S"
       )

    # Calculate elapsed hours
    elapsed_seconds = (df_downsampled["DateTime"] - df_downsampled["DateTime"].iloc[0]).dt.total_seconds()
    elapsed_hours = elapsed_seconds / 3600.0

    # Apply rolling average smoothing
    smoothed_temp = df_downsampled["Temperature"].rolling(window=5, center=True).mean()

    # Plot
    plt.figure(figsize=(8,4))
    plt.plot(elapsed_hours, smoothed_temp, color="red", linewidth=2, label="Smoothed trend")

    for i in range(len(df_downsampled)):
        temp = df_downsampled["Temperature"].iloc[i]
        x = elapsed_hours.iloc[i]
        if 2 <= temp <= 8:
            plt.scatter(x, temp, color="green", s=20)
        elif (-5 <= temp < 2) or (8 < temp <= 15):
            plt.scatter(x, temp, color="orange", s=20)
        else:
            plt.scatter(x, temp, color="red", s=20)

    plt.axhline(y=2, color="green", linestyle="--", linewidth=1, label="Optimal Min (2°C)")
    plt.axhline(y=8, color="orange", linestyle="--", linewidth=1, label="Optimal Max (8°C)")
    plt.title(f"Temperature vs Time for {medicine}")
    plt.xlabel("Elapsed Time (hours)")
    plt.ylabel("Temperature (°C)")
    plt.grid(True)
    plt.legend()

    img = io.BytesIO()
    plt.savefig(img, format="png", dpi=250, bbox_inches="tight")
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()

    # Badge class
    if predicted_loss <= 15:
        badge_class = "prediction-safe"
    elif 15 < predicted_loss <= 25:
        badge_class = "prediction-warning"
    else:
        badge_class = "prediction-danger"

    return render_template("result.html",
                           medicine=medicine,
                           prediction=predicted_loss,
                           badge_class=badge_class,
                           graph_data=graph_url)
