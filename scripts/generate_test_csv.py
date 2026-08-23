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

# Probability configurations for each 1000-row block
prob_configs = [
    (0.40, 0.60),  # Block 1: 40% inside, 60% outside
    (0.50, 0.50),  # Block 2: 50% inside, 50% outside
    (0.70, 0.30),  # Block 3: 70% inside, 30% outside
    (0.75, 0.25),  # Block 4: 75% inside, 25% outside
    (0.78, 0.22)   # Block 5: 78% inside, 22% outside
]

def generate_test_csv(medicine, output_dir="test"):
    os.makedirs(output_dir, exist_ok=True)
    start_time = datetime.datetime.now()
    data = []

    optimal_min, optimal_max = optimal_ranges[medicine]

    # Generate 5 blocks of 1000 rows each
    for block_idx, (p_in, p_out) in enumerate(prob_configs, start=1):
        for i in range(1000):
            timestamp = (start_time + datetime.timedelta(seconds=30*(len(data)))).strftime("%H:%M:%S")
            date = start_time.strftime("%Y-%m-%d")

            # Decide whether to generate inside or outside range
            if random.random() < p_in:
                temp = round(random.uniform(optimal_min, optimal_max), 2)
            else:
                # Outside range: either below -10 to 2 OR above 8 to 24
                if random.random() < 0.5:
                    temp = round(random.uniform(-10, 2), 2)
                else:
                    temp = round(random.uniform(8, 24), 2)

            data.append([date, medicine, temp, timestamp, optimal_min, optimal_max])

    # Save one file per medicine with 5000 rows
    df = pd.DataFrame(data, columns=["Date", "MedicineName", "Temperature", "Timestamp", "OptimalMin", "OptimalMax"])
    filename = f"{output_dir}/{medicine.replace(' ', '_')}_test.csv"
    df.to_csv(filename, index=False)
    print(f"Test CSV generated: {filename} with {len(df)} rows")

if __name__ == "__main__":
    medicines = list(optimal_ranges.keys())
    for med in medicines:
        generate_test_csv(med)
