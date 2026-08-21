import pandas as pd
import numpy as np
import os

# THIS IS AI_MODEL_1 PREPROCESS CODE USING LINEAR REGRESSION, WE HAVE SHIFTED TO GRADIENT BOOSTING FOR MORE 
# PRECISE RESULTS AND TO GET MORE REAL WORLD SCENARIOS GETTING INCLUDED.

# # Map medicine names to categorical codes
# medicine_map = {
#     "Comirnaty": 0,
#     "Lantus(insulin glargine)": 1,
#     "Humira": 2,
#     "Gardasil9": 3,
#     "HepatitisB": 4
# }

# def preprocess_csv(file_path):
#     df = pd.read_csv(file_path)

#     # Calculate deviation from optimal range
#     df["Deviation"] = df.apply(
#         lambda row: 0 if row["OptimalMin"] <= row["Temperature"] <= row["OptimalMax"]
#         else min(abs(row["Temperature"] - row["OptimalMin"]),
#                  abs(row["Temperature"] - row["OptimalMax"])),
#         axis=1
#     )

#     # Calculate time outside optimal range (each row = 30s)
#     df["OutsideRange"] = df.apply(
#         lambda row: 1 if not (row["OptimalMin"] <= row["Temperature"] <= row["OptimalMax"]) else 0,
#         axis=1
#     )

#     # Summarize features for this CSV
#     avg_temp_deviation = df["Deviation"].mean()
#     bad_temp_duration = df["OutsideRange"].sum() * 30 / 3600.0  # hours
#     medicine_type = medicine_map.get(df["MedicineName"].iloc[0], -1)

#     features = {
#         "File": os.path.basename(file_path),
#         "MedicineName": df["MedicineName"].iloc[0],
#         "MedicineCode": medicine_type,
#         "AvgTempDeviation": avg_temp_deviation,
#         "BadTempDurationHours": bad_temp_duration
#     }

#     return features


# def build_training_dataset(train_dir="train"):  
#     all_features = []

#     for file in os.listdir(train_dir):
#         if file.endswith(".csv"):
#             file_path = os.path.join(train_dir, file)
#             features = preprocess_csv(file_path)
#             all_features.append(features)

#     dataset = pd.DataFrame(all_features)
#     return dataset


# if __name__ == "__main__":
#     train_dir = "train"  
#     dataset = build_training_dataset(train_dir)
#     print("Training dataset built with shape:", dataset.shape)
#     print(dataset.head())

#     # Save dataset for training step
#     os.makedirs("results", exist_ok=True)
#     dataset.to_csv("results/preprocessed_training.csv", index=False)
#     print("Preprocessed dataset saved to results/preprocessed_training.csv")


#GRADIENT BOOSTING PREPROCESSING CODE WITH MORE FEATURES AND TRAINED OVER A VERY LARGE DATASET

medicine_map = {
    "Comirnaty": 0,
    "Lantus(insulin glargine)": 1,
    "Humira": 2,
    "Gardasil9": 3,
    "HepatitisB": 4
}

def preprocess_csv(file_path):
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
    avg_loss = df["PotencyLoss"].mean()  # target

    features = {
        "File": os.path.basename(file_path),
        "MedicineName": df["MedicineName"].iloc[0],
        "MedicineCode": medicine_type,
        "AvgTempDeviation": avg_temp_deviation,
        "MaxTempDeviation": max_temp_deviation,
        "BadTempDurationHours": bad_temp_duration,
        "TimeAbove25C": time_above_25,
        "TimeBelow0C": time_below_0,
        "TemperatureVariance": temp_variance,
        "PotencyLoss": avg_loss
    }

    return features

def build_training_dataset(train_dir="train"):
    all_features = []
    for file in os.listdir(train_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(train_dir, file)
            features = preprocess_csv(file_path)
            all_features.append(features)

    dataset = pd.DataFrame(all_features)
    return dataset

if __name__ == "__main__":
    dataset = build_training_dataset("train")
    print("Training dataset built with shape:", dataset.shape)
    print(dataset.head())

    os.makedirs("results", exist_ok=True)
    dataset.to_csv("results/preprocessed_training_GB.csv", index=False)
    print("Preprocessed dataset saved to results/preprocessed_training_GB.csv")
