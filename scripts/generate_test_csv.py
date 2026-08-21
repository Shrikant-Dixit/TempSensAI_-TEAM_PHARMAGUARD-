import pandas as pd
import random
import datetime
import os

# Optimal ranges per medicine
optimal_ranges = {
    "Comirnaty": [2, 8],
    "Lantus(insulin glargine)": [2, 8],
    "Humira": [2, 8],
    "Gardasil9": [2, 8],
    "HepatitisB": [2, 8]
}

def generate_test_csv(medicine, rows=1000, output_dir="test"):
    os.makedirs(output_dir, exist_ok=True)

    start_time = datetime.datetime.now()
    data = []

    for i in range(rows):
        timestamp = (start_time + datetime.timedelta(seconds=30*i)).strftime("%H:%M:%S")
        date = start_time.strftime("%Y-%m-%d")

        # Random temperature simulation
        temp = round(random.uniform(2, 40), 2)
        optimal_min, optimal_max = optimal_ranges[medicine]

        data.append([date, medicine, temp, timestamp, optimal_min, optimal_max])

    df = pd.DataFrame(data, columns=["Date", "MedicineName", "Temperature", "Timestamp", "OptimalMin", "OptimalMax"])
    filename = f"{output_dir}/{medicine.replace(' ', '_')}_test.csv"
    df.to_csv(filename, index=False)
    print(f"Test CSV generated: {filename}")

if __name__ == "__main__":
    medicines = list(optimal_ranges.keys())
    for med in medicines:
        generate_test_csv(med, rows=1000)
