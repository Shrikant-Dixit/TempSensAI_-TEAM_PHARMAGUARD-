# import pandas as pd
# import random
# import datetime
# import os

# #  optimal ranges per medicine
# optimal_ranges = {
#     "Comirnaty": [2, 8],
#     "Lantus(insulin glargine)": [2, 8],
#     "Humira": [2, 8],
#     "Gardasil9": [2, 8],
#     "HepatitisB": [2, 8]
# }

# #  potency reduction rules per medicine
# def calculate_potency_loss(medicine, temp, hours_outside):
#     if medicine == "Comirnaty":
#         if 8 <= temp <= 25:
#             return 1.3 * hours_outside
#         elif 25 < temp <= 40:
#             return 4 * hours_outside

#     elif medicine == "Lantus(insulin glargine)":
#         if temp < 0:
#             return 100  # discarded
#         elif 8 <= temp <= 25:
#             return 1.3 * hours_outside
#         elif 25 < temp <= 40:
#             return 4 * hours_outside

#     elif medicine == "Humira":
#         if temp < 0:
#             return 100
#         elif 8 <= temp <= 25:
#             return 1.3 * hours_outside
#         elif 25 < temp <= 40:
#             return 4 * hours_outside

#     elif medicine == "Gardasil9":
#         if temp < 0:
#             return 100
#         elif 0 <= temp < 2:
#             return 5 * hours_outside
#         elif 8 <= temp <= 25:
#             return 1.3 * hours_outside
#         elif 25 < temp <= 40:
#             return 3 * hours_outside

#     elif medicine == "HepatitisB":
#         if temp < 0:
#             return 100
#         elif 8 <= temp <= 25:
#             return 1.3 * hours_outside
#         elif 25 < temp <= 38:
#             return 1.5 * hours_outside

#     return 0


# def generate_csv(medicine, rows=2000, filename="data.csv"):
#     data = []
#     start_date = datetime.date.today()
#     start_time = datetime.datetime.now()

#     for i in range(rows):
#         date = start_date
#         timestamp = (start_time + datetime.timedelta(seconds=i*30)).strftime("%H:%M:%S")

#         # Simulate temperature (random inside/outside range)
#         if random.random() < 0.7:  # 70% chance inside range
#             temp = random.uniform(optimal_ranges[medicine][0], optimal_ranges[medicine][1])
#         else:  # 30% chance outside range
#             temp = random.uniform(optimal_ranges[medicine][1]+1, optimal_ranges[medicine][1]+10)

#         # Calculation of potency loss per hour (approximate)
#         hours_outside = (i*30) / 3600.0
#         potency_loss = calculate_potency_loss(medicine, temp, hours_outside)

#         data.append([date, medicine, round(temp, 2), timestamp,
#                      optimal_ranges[medicine][0], optimal_ranges[medicine][1],
#                      round(potency_loss, 2)])

#     df = pd.DataFrame(data, columns=["Date", "MedicineName", "Temperature", "Timestamp", "OptimalMin", "OptimalMax", "PotencyLoss"])
#     df.to_csv(filename, index=False)
#     print(f"CSV file '{filename}' generated with {rows} rows.")


# # Generate 2 CSVs per medicine
# output_dir = "train"
# os.makedirs(output_dir, exist_ok=True)

# for medicine in optimal_ranges.keys():
#     for i in range(1, 3):
#         filename = f"{output_dir}/{medicine.replace(' ', '_')}_{i}.csv"
#         generate_csv(medicine, rows=2000, filename=filename)

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

# Potency reduction rules per medicine
def calculate_potency_loss(medicine, temp, hours_outside):
    if medicine == "Comirnaty":
        if 8 <= temp <= 25:
            return 1.3 * hours_outside
        elif 25 < temp <= 40:
            return 4 * hours_outside

    elif medicine == "Lantus(insulin glargine)":
        if temp < 0:
            return 100
        elif 8 <= temp <= 25:
            return 1.3 * hours_outside
        elif 25 < temp <= 40:
            return 4 * hours_outside

    elif medicine == "Humira":
        if temp < 0:
            return 100
        elif 8 <= temp <= 25:
            return 1.3 * hours_outside
        elif 25 < temp <= 40:
            return 4 * hours_outside

    elif medicine == "Gardasil9":
        if temp < 0:
            return 100
        elif 0 <= temp < 2:
            return 5 * hours_outside
        elif 8 <= temp <= 25:
            return 1.3 * hours_outside
        elif 25 < temp <= 40:
            return 3 * hours_outside

    elif medicine == "HepatitisB":
        if temp < 0:
            return 100
        elif 8 <= temp <= 25:
            return 1.3 * hours_outside
        elif 25 < temp <= 38:
            return 1.5 * hours_outside

    return 0


def generate_csv(medicine, rows, inside_prob, filename):
    data = []
    start_date = datetime.date.today()
    start_time = datetime.datetime.now()

    for i in range(rows):
        date = start_date
        timestamp = (start_time + datetime.timedelta(seconds=i*30)).strftime("%H:%M:%S")

        # Probability-based temperature simulation
        if random.random() < inside_prob:
            temp = random.uniform(optimal_ranges[medicine][0], optimal_ranges[medicine][1])
        else:
            temp = random.uniform(optimal_ranges[medicine][1]+1, optimal_ranges[medicine][1]+10)

        hours_outside = (i*30) / 3600.0
        potency_loss = calculate_potency_loss(medicine, temp, hours_outside)

        data.append([date, medicine, round(temp, 2), timestamp,
                     optimal_ranges[medicine][0], optimal_ranges[medicine][1],
                     round(potency_loss, 2)])

    df = pd.DataFrame(data, columns=["Date", "MedicineName", "Temperature", "Timestamp", "OptimalMin", "OptimalMax", "PotencyLoss"])
    df.to_csv(filename, index=False)
    print(f"CSV file '{filename}' generated with {rows} rows.")


if __name__ == "__main__":
    output_dir = "train"
    os.makedirs(output_dir, exist_ok=True)

    # Define scenarios: (inside probability, outside probability)
    scenarios = [
        (0.40, 0.60, "40in60out"),
        (0.50, 0.50, "50in50out"),
        (0.35, 0.65, "35in65out"),
        (0.84, 0.16, "84in16out"),
        (0.20, 0.80, "20in80out")
    ]

    for inside_prob, outside_prob, tag in scenarios:
        for medicine in optimal_ranges.keys():
            filename = f"{output_dir}/{medicine.replace(' ', '_')}_{tag}.csv"
            generate_csv(medicine, rows=20000, inside_prob=inside_prob, filename=filename)
