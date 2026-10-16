"""
app.py
======
Flask Backend for Smart Room Occupancy Detection System.

Features:
- Loads fitted Random Forest 3-feature model ['Temperature', 'Humidity', 'Light'].
- Integrates modular PIRMotionTracker for motion events and timeout management.
- Evaluates combined occupancy state via explicit heuristic sensor-fusion rules.
- Robust input validation and error responses.
- Explicit labeling of test/simulated inputs until ESP32 hardware is connected.
- In-memory recent readings buffer for dashboard trend visualization.
"""

from datetime import datetime
import json
import os
import time
from flask import Flask, jsonify, render_template, request, send_from_directory
import joblib
import numpy as np
import pandas as pd

from pir_tracker import PIRMotionTracker, combine_occupancy_decision

app = Flask(__name__)

# Constants and Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'occupancy_rf_model.joblib')
METRICS_PATH = os.path.join(os.path.dirname(__file__), 'models', 'metrics.json')

# Global State
START_TIME = time.time()
pir_tracker = PIRMotionTracker(default_timeout_seconds=300)
prediction_history = []  # Ring buffer for recent predictions (max 50)
latest_prediction = None  # Holds the most recent prediction payload for dashboard polling
MAX_HISTORY_LEN = 50

# Load Model Bundle
model_bundle = None
rf_model = None
feature_names = []

if os.path.exists(MODEL_PATH):
    try:
        model_bundle = joblib.load(MODEL_PATH)
        rf_model = model_bundle['model']
        feature_names = model_bundle.get('feature_names', ['Temperature', 'Humidity', 'Light'])
        print(f"[INFO] Successfully loaded model bundle from {MODEL_PATH}")
        print(f"[INFO] Features expected: {feature_names}")
    except Exception as e:
        print(f"[ERROR] Failed to load model bundle: {e}")
else:
    print(f"[WARNING] Model file {MODEL_PATH} not found. Please run train_model.py first.")


def validate_prediction_input(data):
    """
    Validate environmental and motion sensor inputs.
    Returns (cleaned_dict, error_message).
    """
    if not isinstance(data, dict):
        return None, "Request body must be a JSON object."

    required = ['temperature', 'humidity', 'light']
    for field in required:
        if field not in data:
            return None, f"Missing required field: '{field}'."
        try:
            val = float(data[field])
            if np.isnan(val) or np.isinf(val):
                return None, f"Field '{field}' must be a finite number."
        except (ValueError, TypeError):
            return None, f"Field '{field}' must be a valid numeric value."

    temp = float(data['temperature'])
    hum = float(data['humidity'])
    light = float(data['light'])

    # Sanity checks with helpful guidance
    if temp < -20.0 or temp > 70.0:
        return None, f"Temperature {temp}°C is outside realistic indoor limits (-20°C to 70°C)."

    if hum < 0.0 or hum > 100.0:
        return None, f"Relative humidity {hum}% must be between 0% and 100%."

    if light < 0.0:
        return None, f"Light level {light} Lux cannot be negative."

    pir_val = 0
    if 'pir' in data and data['pir'] is not None:
        try:
            pir_val = int(data['pir'])
            if pir_val not in (0, 1):
                return None, "Field 'pir' must be either 0 (LOW) or 1 (HIGH)."
        except (ValueError, TypeError):
            return None, "Field 'pir' must be integer 0 or 1."

    source = str(data.get('source', 'manual_test')).strip()
    return {
        'temperature': temp,
        'humidity': hum,
        'light': light,
        'pir': pir_val,
        'source': source
    }, None


# --------------------------------------------------------------------------
# Web Routes
# --------------------------------------------------------------------------
@app.route('/')
def index():
    """Render the primary dashboard."""
    return render_template('index.html')


# --------------------------------------------------------------------------
# API Routes
# --------------------------------------------------------------------------
@app.route('/api/health', methods=['GET'])
def health_check():
    """System health check and status."""
    uptime_seconds = round(time.time() - START_TIME, 1)
    return jsonify({
        'status': 'healthy',
        'model_loaded': (rf_model is not None),
        'uptime_seconds': uptime_seconds,
        'expected_features': feature_names,
        'pir_timeout_seconds': pir_tracker.timeout_seconds,
        'timestamp': datetime.now().isoformat()
    }), 200


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Retrieve model training & evaluation metrics."""
    if not os.path.exists(METRICS_PATH):
        return jsonify({'error': 'Metrics not found. Run train_model.py first.'}), 404

    with open(METRICS_PATH, 'r') as f:
        metrics_data = json.load(f)

    return jsonify(metrics_data), 200


@app.route('/api/predict', methods=['POST'])
def predict_occupancy():
    """
    Main prediction endpoint.
    Takes environmental readings [temperature, humidity, light] and optional PIR signal.
    Returns ML prediction, PIR tracking state, and combined occupancy decision.
    """
    if rf_model is None:
        return jsonify({
            'error': 'ML model is not loaded. Run train_model.py to generate models/occupancy_rf_model.joblib.'
        }), 503

    raw_data = request.get_json(silent=True)
    if not raw_data:
        return jsonify({'error': 'Invalid JSON body in request.'}), 400

    clean_data, err = validate_prediction_input(raw_data)
    if err:
        return jsonify({'error': err}), 400

    temp = clean_data['temperature']
    hum = clean_data['humidity']
    light = clean_data['light']
    pir_val = clean_data['pir']
    source = clean_data['source']

    # 1. Update PIR motion tracker
    pir_status = pir_tracker.register_signal(pir_val, source=source)

    # 2. Environmental ML Inference
    # DataFrame ensures exact ordered column names
    input_df = pd.DataFrame([{
        'Temperature': temp,
        'Humidity': hum,
        'Light': light
    }], columns=feature_names)

    ml_pred = int(rf_model.predict(input_df)[0])
    ml_probs = rf_model.predict_proba(input_df)[0]
    p_unocc = float(ml_probs[0])
    p_occ = float(ml_probs[1])

    # 3. Combined Heuristic Decision Layer
    combined_result = combine_occupancy_decision(
        ml_prediction=ml_pred,
        ml_probability=p_occ,
        pir_status=pir_status
    )

    # Format response
    response_payload = {
        'status': 'success',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'input_mode': 'live_hardware' if source == 'esp32' else 'simulated_test_input',
        'input_notice': (
            'Awaiting ESP32 hardware connection. These values are simulated test inputs.'
            if source != 'esp32' else 'Live hardware readings received from ESP32.'
        ),
        'inputs': {
            'temperature': temp,
            'humidity': hum,
            'light': light,
            'pir_pin': pir_val
        },
        'ml_prediction': {
            'target': ml_pred,
            'label': 'Occupied' if ml_pred == 1 else 'Unoccupied',
            'occupied_probability': round(p_occ, 4),
            'unoccupied_probability': round(p_unocc, 4),
            'confidence_percent': round(max(p_occ, p_unocc) * 100, 2),
            'features_used': feature_names
        },
        'pir_status': pir_status,
        'combined_decision': combined_result
    }

    global latest_prediction
    latest_prediction = response_payload

    # Record into history buffer (circular, max 50 items)
    prediction_history.append({
        'timestamp': response_payload['timestamp'],
        'temperature': temp,
        'humidity': hum,
        'light': light,
        'pir_pin': pir_val,
        'ml_label': response_payload['ml_prediction']['label'],
        'occ_prob': response_payload['ml_prediction']['occupied_probability'],
        'combined_status': combined_result['combined_status'],
        'badge_variant': combined_result['badge_variant'],
        'source': source
    })
    if len(prediction_history) > MAX_HISTORY_LEN:
        prediction_history.pop(0)

    return jsonify(response_payload), 200


@app.route('/api/latest', methods=['GET'])
def get_latest_prediction():
    """Retrieve the most recent prediction payload for frontend auto-polling."""
    if latest_prediction is None:
        return jsonify({'status': 'no_data'}), 404
    return jsonify(latest_prediction), 200


@app.route('/api/pir/motion', methods=['POST'])
def trigger_pir_motion():
    """
    Dedicated endpoint for hardware (ESP32) or UI to send a PIR motion pulse.
    Body: {"pin_state": 1, "source": "esp32" | "manual_test"}
    """
    data = request.get_json(silent=True) or {}
    pin_state = 1 if data.get('pin_state', 1) else 0
    source = str(data.get('source', 'manual_test'))

    updated_status = pir_tracker.register_signal(pin_state, source=source)
    return jsonify({
        'status': 'success',
        'message': f"PIR motion signal registered as {updated_status['pin_label']}.",
        'pir_status': updated_status
    }), 200


@app.route('/api/pir/status', methods=['GET'])
def get_pir_status():
    """Check current PIR state, seconds since motion, and timeout status."""
    return jsonify(pir_tracker.get_status()), 200


@app.route('/api/pir/config', methods=['POST'])
def update_pir_config():
    """Configure the occupancy timeout in seconds."""
    data = request.get_json(silent=True) or {}
    timeout = data.get('timeout_seconds')

    if timeout is None:
        return jsonify({'error': "Missing 'timeout_seconds' in request body."}), 400

    try:
        timeout = int(timeout)
        new_val = pir_tracker.set_timeout(timeout)
        return jsonify({
            'status': 'success',
            'message': f"PIR timeout updated to {new_val} seconds.",
            'timeout_seconds': new_val
        }), 200
    except ValueError as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/history', methods=['GET'])
def get_prediction_history():
    """Retrieve history of recent predictions for charting."""
    return jsonify({
        'count': len(prediction_history),
        'history': prediction_history
    }), 200


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Resource not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Local development server
    port = int(os.environ.get('PORT', 5000))
    print(f"\n[SERVER] Starting Smart Room Occupancy Detection Flask App on http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
