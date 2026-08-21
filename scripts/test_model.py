# import pandas as pd
# import joblib
# import os
# from preprocess import preprocess_csv

# def test_model(test_dir="test", model_path="models/linear_regression.pkl"):
#     # Load trained model
#     model = joblib.load(model_path)

#     results = []

#     for file in os.listdir(test_dir):
#         if file.endswith(".csv"):
#             file_path = os.path.join(test_dir, file)
#             features = preprocess_csv(file_path)

#             # Convert features into DataFrame for prediction
#             X = pd.DataFrame([[
#                 features["MedicineCode"],
#                 features["AvgTempDeviation"],
#                 features["BadTempDurationHours"]
#             ]], columns=["MedicineCode", "AvgTempDeviation", "BadTempDurationHours"])

#             # Predict potency loss
#             predicted_loss = model.predict(X)[0]

#             results.append({
#                 "File": file,
#                 "MedicineName": features["MedicineName"],
#                 "PredictedPotencyLoss": predicted_loss
#             })

#     return pd.DataFrame(results)

# if __name__ == "__main__":
#     df_results = test_model()
#     print("Test Results:\n", df_results)
#     os.makedirs("results", exist_ok=True)
#     df_results.to_csv("results/test_predictions.csv", index=False)
#     print("Predictions saved to results/test_predictions.csv")


import pandas as pd
import joblib
import os

# Medicine mapping (same as in preprocess_GB.py)
medicine_map = {
    "Comirnaty": 0,
    "Lantus(insulin glargine)": 1,
    "Humira": 2,
    "Gardasil9": 3,
    "HepatitisB": 4
}

# HERE WE ARE NOT IMPORTING PREPROCESS.PY INSTEAD WE ARE WRITING FUNCTION BECAUSE OUR BACKEND NEED TO BE INDEPENDENT OF OUR AI MODEL FOLDER 
#BECAUSE WE WANT TO MAKE IT WORK INDEPENDENTLY FOR OUR SITE WITH THE QR DATA FETCHING

def preprocess_test_csv(file_path):
    df = pd.read_csv(file_path)

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
        "File": os.path.basename(file_path),
        "MedicineName": df["MedicineName"].iloc[0],
        "MedicineCode": medicine_type,
        "AvgTempDeviation": avg_temp_deviation,
        "MaxTempDeviation": max_temp_deviation,
        "BadTempDurationHours": bad_temp_duration,
        "TimeAbove25C": time_above_25,
        "TimeBelow0C": time_below_0,
        "TemperatureVariance": temp_variance
    }

    return features

def test_model(test_dir="test", model_path="models/gradient_boosting.pkl"):
    # Loading trained Gradient Boosting model
    model = joblib.load(model_path)

    results = []

    for file in os.listdir(test_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(test_dir, file)
            features = preprocess_test_csv(file_path)

            # Converting features into DataFrame for prediction
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

            results.append({
                "File": features["File"],
                "MedicineName": features["MedicineName"],
                "PredictedPotencyLoss": predicted_loss
            })

    return pd.DataFrame(results)

if __name__ == "__main__":
    df_results = test_model()
    print("Test Results:\n", df_results)
    os.makedirs("results", exist_ok=True)
    df_results.to_csv("results/test_predictions_GB.csv", index=False)
    print("Predictions saved to results/test_predictions_GB.csv")
