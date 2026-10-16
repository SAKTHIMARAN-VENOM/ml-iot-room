# Smart Room Occupancy Detection (ML & IoT)

A beginner-friendly, production-structured IoT Machine Learning project that determines whether an indoor room is **Occupied** or **Unoccupied** using environmental sensors and a separate PIR motion tracking logic layer.

---

## 📌 Architecture Overview

This project implements **Option B**:
1. **Environmental ML Model (Random Forest)**:
   - Trained on genuine historical data ([Occupancy.csv](Occupancy.csv)).
   - Takes only the 3 available environmental sensor features:
     - `Temperature` (°C) from DHT11
     - `Humidity` (%) from DHT11
     - `Light` (Lux) from bare LDR voltage divider
   - Evaluates sustained environmental presence without requiring an unavailable CO2 sensor.
2. **PIR Motion Tracking & Heuristic Rule Layer ([pir_tracker.py](pir_tracker.py))**:
   - Manages genuine physical PIR motion pulses (HIGH = 1, LOW = 0).
   - Tracks time elapsed since the last motion event (`seconds_since_last_motion`).
   - Provides a configurable occupancy hold timeout (default: 300 seconds).
   - Treats PIR LOW as "no movement right now", **never** as proof that the room is empty.
3. **Flask Backend & Web Dashboard ([app.py](app.py))**:
   - Exposes RESTful endpoints for prediction, motion pulses, timeout configuration, and metrics.
   - Interactive dashboard with real-time Chart.js trends, preset test scenarios, and a verified confusion matrix modal.
   - Clear simulation banners stating that inputs are manually entered test values until the ESP32 is physically connected.

---

## 📊 Dataset Schema & Model Training

### Why 3 Environmental Features?
- `Occupancy.csv` contains: `date, Temperature, Humidity, Light, CO2, HumidityRatio, Occupancy`.
- **CO2 is omitted**: You do not have a CO2 sensor (e.g. MQ-135 or NDIR). Chronological benchmarking confirmed that omitting CO2 resulted in virtually identical test accuracy (99.54% vs 99.56%).
- **PIR is NOT in `Occupancy.csv`**: Rather than fabricating artificial synthetic PIR rows, PIR is handled cleanly as a separate physical rule layer.

### Why Time-Based Chronological Split?
Random k-fold splitting on continuous time-series data causes **temporal data leakage** (the model sees future minutes to predict past minutes). To guarantee genuine generalization:
- **Chronological Split**: First 80% (16,448 samples, Feb 2 – Feb 15) for training; final 20% (4,112 samples, Feb 15 – Feb 18) for testing.

### Verified Test Evaluation Metrics

| Metric | Dummy Baseline | Logistic Regression | **Random Forest (Final Model)** |
| :--- | :--- | :--- | :--- |
| **Accuracy** | 73.32% | 99.56% | **99.54%** |
| **Precision (Class 1)** | 0.00% | 98.56% | **98.56%** |
| **Recall (Class 1)** | 0.00% | 99.82% | **99.73%** |
| **F1-Score (Class 1)** | 0.00% | 99.18% | **99.14%** |
| **ROC-AUC** | 0.5000 | 0.9983 | **0.9984** |

#### Test Confusion Matrix (4,112 unseen samples):
- **True Negatives (Unoccupied correctly predicted)**: `2,999`
- **False Positives (Unoccupied predicted as occupied)**: `16`
- **False Negatives (Occupied missed as unoccupied)**: `3`
- **True Positives (Occupied correctly predicted)**: `1,094`

#### Feature Importances:
- `Light`: **66.69%** (office lighting strongly signals human presence)
- `Temperature`: **28.20%** (body heat and room heating dynamics)
- `Humidity`: **5.11%** (ambient respiration changes)

---

## 🧠 Combined Decision Logic & Truth Table

A physical PIR sensor only triggers on **differential infrared movement**. Analysis of real room occupancy motion data reveals:
> **When a room IS occupied, PIR is `0` (no motion) ~38% of the time** because people sit still at desks, read, or sleep.

The combined logic layer integrates the ML prediction with PIR state using an explicit decision matrix:

| Case | PIR Pin | Within Timeout | ML Predict | Combined Status | Confidence | Reasoning |
| :---: | :---: | :---: | :---: | :--- | :---: | :--- |
| **1** | HIGH | Yes | Occupied (1) | **Confirmed Occupied** | Very High | Dual-verification: Physical motion actively detected + environmental sensors confirm occupancy. |
| **2** | LOW | Yes | Occupied (1) | **Confirmed Occupied** | High | Motion registered recently within timeout window + environmental conditions confirm presence. |
| **3** | LOW | No | Occupied (1) | **Occupied (Still/Seated Presence)** | Moderate-High | PIR detects no movement (occupant seated still/reading), but environmental ML strongly indicates presence. PIR LOW does NOT equal empty room. |
| **4** | HIGH | Yes | Unoccupied (0) | **Transitory Motion Detected** | Moderate | Motion triggered, but environmental conditions remain at baseline. Suggests quick doorway walk-through. |
| **5** | LOW | Yes | Unoccupied (0) | **Awaiting Verification** | Low-Moderate | Motion was registered recently, but environmental sensors show room is empty. Holding until timeout expires. |
| **6** | LOW | No | Unoccupied (0) | **Confirmed Unoccupied** | High | No recent motion past timeout window AND environmental model confirms room is vacant. |

### ⚠️ Heuristic Layer Disclaimer
> **Notice**: This rule layer is an empirical sensor-fusion heuristic. It must be calibrated against real-world physical sensor deployments before being treated as scientifically validated.

---

## 🚀 Setup and Running Instructions

### 1. Prerequisites & Virtual Environment
Ensure you have Python 3.10+ installed.

```bash
# Clone or navigate to the project directory
cd c:\mL-IOT

# Install dependencies
python -m pip install -r requirements.txt
```

### 2. Train the Environmental Model
Run the model training pipeline to inspect the dataset, train the Random Forest, evaluate test metrics, and export the model bundle:

```bash
python train_model.py
```

Generated artifacts:
- `models/occupancy_rf_model.joblib`: Trained model and ordered feature dictionary.
- `models/metrics.json`: Genuine test metrics and split metadata.
- `static/img/confusion_matrix.png`: Matplotlib visualization of the test confusion matrix.

### 3. Start the Flask Server
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 📡 REST API Reference

### 1. `POST /api/predict`
Evaluates room occupancy based on sensor readings.

**Request Body:**
```json
{
  "temperature": 23.5,
  "humidity": 28.0,
  "light": 450.0,
  "pir": 1,
  "source": "manual_test"
}
```

**Response (Example):**
```json
{
  "status": "success",
  "timestamp": "2026-10-16 09:35:00",
  "input_mode": "simulated_test_input",
  "input_notice": "Awaiting ESP32 hardware connection. These values are simulated test inputs.",
  "inputs": {
    "temperature": 23.5,
    "humidity": 28.0,
    "light": 450.0,
    "pir_pin": 1
  },
  "ml_prediction": {
    "target": 1,
    "label": "Occupied",
    "occupied_probability": 0.985,
    "unoccupied_probability": 0.015,
    "confidence_percent": 98.5,
    "features_used": ["Temperature", "Humidity", "Light"]
  },
  "pir_status": {
    "pin_state": 1,
    "pin_label": "HIGH (Motion)",
    "seconds_since_last_motion": 0.0,
    "timeout_seconds": 300,
    "is_within_timeout": true
  },
  "combined_decision": {
    "combined_status": "Confirmed Occupied",
    "status_code": "OCCUPIED_ACTIVE_MOTION",
    "badge_variant": "success",
    "confidence_level": "Very High",
    "decision_reasoning": "Dual-verification: Physical infrared motion is actively detected and environmental sensors align with an occupied room.",
    "is_occupied_overall": true
  }
}
```

### 2. `POST /api/pir/motion`
Registers a PIR pulse from hardware or UI.
```json
{ "pin_state": 1, "source": "esp32" }
```

### 3. `POST /api/pir/config`
Updates the occupancy hold timeout.
```json
{ "timeout_seconds": 180 }
```

### 4. `GET /api/metrics`
Returns verified model test performance metrics and feature importances.

### 5. `GET /api/health`
Checks server uptime, model load status, and configured timeout.

---

## 🔌 Future ESP32 Hardware Integration Guide

When you are ready to connect your physical hardware:

### Wiring Pinout (ESP32)
- **DHT11**:
  - `VCC` -> 3.3V
  - `GND` -> GND
  - `DATA` -> GPIO 4 (with 10k pull-up resistor)
- **LDR Light Sensor**:
  - 10k resistor in voltage divider with LDR
  - Divider node -> GPIO 34 (ADC1 analog input)
- **PIR Motion Sensor (HC-SR501 / AM312)**:
  - `VCC` -> 5V (or 3.3V for AM312)
  - `GND` -> GND
  - `OUT` -> GPIO 13 (Digital input)

### ESP32 C++ Snippet (HTTP POST to Flask):
```cpp
#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>

#define DHTPIN 4
#define DHTTYPE DHT11
#define LDRPIN 34
#define PIRPIN 13

DHT dht(DHTPIN, DHTTYPE);
const char* serverUrl = "http://<YOUR_COMPUTER_IP>:5000/api/predict";

void sendSensorData() {
  float temp = dht.readTemperature();
  float hum = dht.readHumidity();
  int ldrRaw = analogRead(LDRPIN);
  float lux = map(ldrRaw, 0, 4095, 0, 1000); // Calibrated Lux scale
  int pirState = digitalRead(PIRPIN);

  if (isnan(temp) || isnan(hum)) return;

  HTTPClient http;
  http.begin(serverUrl);
  http.addHeader("Content-Type", "application/json");

  String jsonPayload = String("{\"temperature\":") + temp + 
                       ",\"humidity\":" + hum + 
                       ",\"light\":" + lux + 
                       ",\"pir\":" + pirState + 
                       ",\"source\":\"esp32\"}";

  int httpCode = http.POST(jsonPayload);
  http.end();
}
```
