from flask import Flask, render_template, jsonify, request, redirect, url_for
import joblib
import pandas as pd
import numpy as np
import random
import os
import threading
import time

app = Flask(__name__)

# --- Global State ---
current_system_state = {
    "prediction": 0,
    "risk_probability": 0.0,
    "metrics": {
       "it": {"bandwidth": 0, "backbone_latency": 0, "server_resp_time": 0},
       "hr": {"active_sessions": 0, "auth_failures": 0, "system_load": 0},
       "iot": {"sensor_temp": 0, "ingestion_delay": 0, "pdr": 100},
       "finance": {"trans_vol": 0, "db_query_duration": 0, "api_handshake": 0},
       "hub": {"cpu": 0, "ram": 0, "disk": 0, "net_in": 0}
    },
    "Active Threat": False
}

# --- Model Loading ---
MODEL_PATH = 'nexus_model.pkl'
COLUMNS_PATH = 'model_columns.pkl'
DATA_PATH = 'enterprise_data.csv'

try:
    model = joblib.load(MODEL_PATH)
    model_columns = joblib.load(COLUMNS_PATH)
    enterprise_data = pd.read_csv(DATA_PATH)
    print("SUCCESS: Model, Columns, and Data successfully loaded.")
except Exception as e:
    print(f"ERROR: Error loading initial files: {e}")
    # In a real app we might exit or handle gracefully, keeping it simple here

# --- Routing Setup ---

@app.route('/')
def login():
    """Initial entry point - Login Page"""
    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    """Main Executive Dashboard"""
    return render_template('dashboard.html')

@app.route('/network')
def network():
    """IT Infrastructure Metrics"""
    return render_template('network.html')

@app.route('/hr')
def hr():
    """HR/IK System Logs"""
    return render_template('hr.html')

@app.route('/iot')
def iot():
    """Sensor & Network Telemetry"""
    return render_template('iot.html')

@app.route('/finance')
def finance():
    """Transactional & DB Queries"""
    return render_template('finance.html')

# --- API Endpoints ---

def master_simulation_loop():
    global current_system_state
    while True:
        try:
            # 1. Simulate Live Data Feed (Randomly sample 1 row from CSV)
            sampled_row = enterprise_data.sample(n=1).iloc[0].to_dict()
            
            # Synthetic Attack Vector Injection
            # ~5% chance to inject an anomaly to test UI oscilloscope
            if random.random() < 0.05:
                attack_type = random.choice(["iot_drop", "it_spike", "finance_latency"])
                if attack_type == "iot_drop":
                    sampled_row["iot_pdr"] = random.uniform(50.0, 85.0)
                elif attack_type == "it_spike":
                    sampled_row["it_server_resp_time"] = random.uniform(500.0, 2000.0)
                elif attack_type == "finance_latency":
                    sampled_row["fin_db_query_duration"] = random.uniform(15.0, 45.0)

            # 2. Prepare data for Model Prediction
            input_df = pd.DataFrame([sampled_row])
            for col in model_columns:
                if col not in input_df.columns:
                    input_df[col] = 0.0
            input_df = input_df[model_columns]
            
            # 3. Model Prediction
            prediction_array = model.predict(input_df)
            prediction = int(prediction_array[0])
            prob_array = model.predict_proba(input_df)
            risk_probability = float(prob_array[0][1])
            
            # 4. Update Global State
            new_state = {
                "prediction": prediction,
                "risk_probability": risk_probability,
                "metrics": {
                    "it": {
                        "bandwidth": sampled_row.get("it_bandwidth", 0),
                        "backbone_latency": sampled_row.get("it_backbone_latency", 0),
                        "server_resp_time": sampled_row.get("it_server_resp_time", 0)
                    },
                    "hr": {
                        "active_sessions": random.randint(100, 500),
                        "auth_failures": random.randint(0, 10),
                        "system_load": random.uniform(10.0, 60.0)
                    },
                    "iot": {
                        "sensor_temp": sampled_row.get("iot_sensor_temp", 0),
                        "ingestion_delay": sampled_row.get("iot_ingestion_delay", 0),
                        "pdr": sampled_row.get("iot_pdr", 0)
                    },
                    "finance": {
                        "trans_vol": sampled_row.get("fin_trans_vol", 0),
                        "db_query_duration": sampled_row.get("fin_db_query_duration", 0),
                        "api_handshake": sampled_row.get("fin_api_handshake", 0)
                    },
                    "hub": {
                        "cpu": sampled_row.get("hub_cpu", 0),
                        "ram": sampled_row.get("hub_ram", 0),
                        "disk": sampled_row.get("hub_disk", 0),
                        "net_in": sampled_row.get("hub_net_in", 0)
                    }
                },
                "Active Threat": True if prediction == 1 else False
            }
            
            current_system_state.update(new_state)
        except Exception as e:
            print(f"Simulation Error: {e}")
            
        time.sleep(2) # Update every 2000ms

@app.route('/api/v1/status', methods=['GET'])
def api_status():
    """Unified API Endpoint serving the global state"""
    return jsonify(current_system_state)

if __name__ == '__main__':
    # Start the background simulation thread
    sim_thread = threading.Thread(target=master_simulation_loop, daemon=True)
    sim_thread.start()
    
    app.run(debug=True, port=5000)