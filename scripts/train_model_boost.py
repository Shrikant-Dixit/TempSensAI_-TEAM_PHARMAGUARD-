import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_gradient_boosting(data_path="results/preprocessed_training_GB.csv", model_path="models/gradient_boosting.pkl"):
    # Load preprocessed dataset
    df = pd.read_csv(data_path)

    # Features and target
    X = df[["MedicineCode", "AvgTempDeviation", "MaxTempDeviation",
            "BadTempDurationHours", "TimeAbove25C", "TimeBelow0C", "TemperatureVariance"]]
    y = df["PotencyLoss"]

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define Gradient Boosting model
    model = GradientBoostingRegressor(
        n_estimators=300,      # number of boosting stages
        learning_rate=0.05,    # step size shrinkage
        max_depth=4,           # depth of individual trees
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)

    print("Validation MSE:", mse)
    print("Validation R²:", r2)

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Gradient Boosting model saved to {model_path}")

if __name__ == "__main__":
    train_gradient_boosting()
