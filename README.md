# TempSensAI – TEAM PHARMAGUARD

TempSensAI is an AI-powered system for **medicine temperature monitoring and potency prediction**, fully compatible with ESP32 IoT data posts.  

The model analyzes temperature logs uploaded via QR-based fetching and predicts the **percentage potency loss** of the medicine based on exposure outside its optimal storage range.  

## ✨ Features
- Gradient Boosting AI model trained on synthetic + rule-based datasets.  
- Predicts **potency loss (%)** for multiple medicines (Comirnaty, Lantus, Humira, Gardasil9, HepatitisB).  
- Preprocessing pipeline extracts rich features (temperature deviation, variance, time above 25°C, time below 0°C, etc.).  
- Backend-ready: integrates with ESP32 data posts and QR code fetching.  
- Outputs precise potency loss predictions for safe medicine usage decisions.  

## 🚀 Workflow
1. ESP32 posts temperature data to server.  
2. Data is fetched via QR code.  
3. Backend preprocesses the CSV.  
4. Gradient Boosting model predicts potency loss.  
5. Result is returned to the dashboard/API.  

## 📂 Project Structure
- `train/` → Training CSVs (synthetic datasets).  
- `results/` → Preprocessed datasets + predictions.  
- `models/` → Saved AI models (`gradient_boosting.pkl`).  
- `scripts/` → Python scripts (`generate_csv.py`, `preprocess_GB.py`, `train_model_boost.py`, `test_model.py`).  

---


