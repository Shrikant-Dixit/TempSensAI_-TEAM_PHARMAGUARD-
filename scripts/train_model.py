import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import joblib

def load_preprocessed_data(preprocessed_file="results/preprocessed_training.csv", train_dir="train"):
    # Load features
    features_df = pd.read_csv(preprocessed_file)

    # Collect potency loss from synthetic CSVs
    potency_values = []
    for file in features_df["File"]:
        csv_path = os.path.join(train_dir, file)
        df = pd.read_csv(csv_path)
        # Average potency loss across the file
        avg_loss = df["PotencyLoss"].mean()
        potency_values.append(avg_loss)

    # Add target column
    features_df["PotencyLoss"] = potency_values
    return features_df

def train_model(dataset):
    # Features (X) and target (y)
    X = dataset[["MedicineCode", "AvgTempDeviation", "BadTempDurationHours"]]
    y = dataset["PotencyLoss"]

    # Train Linear Regression
    model = LinearRegression()
    model.fit(X, y)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/linear_regression.pkl")
    print("Model trained and saved to models/linear_regression.pkl")

if __name__ == "__main__":
    dataset = load_preprocessed_data()
    print("Training dataset:\n", dataset.head())
    train_model(dataset)
