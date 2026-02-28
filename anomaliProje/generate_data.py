import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_enterprise_data(num_rows=1000, output_path='enterprise_data.csv'):
    # Columns required by the model
    columns = [
        "hub_cpu", "hub_ram", "hub_disk", "hub_net_in",
        "it_bandwidth", "it_backbone_latency", "it_server_resp_time",
        "iot_sensor_temp", "iot_ingestion_delay", "iot_pdr",
        "fin_trans_vol", "fin_db_query_duration", "fin_api_handshake",
        "hour", "minute"
    ]
    
    data = []
    base_time = datetime.now() - timedelta(days=1)
    
    for i in range(num_rows):
        current_time = base_time + timedelta(minutes=i)
        
        # Determine if this row should be anomalous (approx 10% chance)
        is_anomaly = random.random() < 0.10
        anomaly_type = random.choice(['hub', 'it', 'iot', 'fin']) if is_anomaly else None
        
        # Base normal values
        row = {
            "hour": current_time.hour,
            "minute": current_time.minute,
            
            # HUB (Central Server) - Normal: CPU 20-60%, RAM 30-70%, Disk 40-60%, Net 100-500
            "hub_cpu": random.uniform(20.0, 60.0),
            "hub_ram": random.uniform(30.0, 70.0),
            "hub_disk": random.uniform(40.0, 60.0),
            "hub_net_in": random.uniform(100.0, 500.0),
            
            # IT Infrastructure - Normal: BW 60-90%, Latency 5-20ms, Resp 20-50ms
            "it_bandwidth": random.uniform(60.0, 90.0),
            "it_backbone_latency": random.uniform(5.0, 20.0),
            "it_server_resp_time": random.uniform(20.0, 50.0),
            
            # IoT Telemetry - Normal: Temp 30-50C, Delay 50-200ms, PDR 98-100%
            "iot_sensor_temp": random.uniform(30.0, 50.0),
            "iot_ingestion_delay": random.uniform(50.0, 200.0),
            "iot_pdr": random.uniform(98.0, 100.0),
            
            # Finance - Normal: Trans 20-100/s, DB query 2-10ms, API 30-80ms
            "fin_trans_vol": random.uniform(20.0, 100.0),
            "fin_db_query_duration": random.uniform(2.0, 10.0),
            "fin_api_handshake": random.uniform(30.0, 80.0)
        }
        
        # Inject anomalies
        if anomaly_type == 'hub':
            row["hub_cpu"] = random.uniform(85.0, 100.0)
            row["hub_ram"] = random.uniform(90.0, 100.0)
        elif anomaly_type == 'it':
            row["it_backbone_latency"] = random.uniform(100.0, 500.0)
            row["it_server_resp_time"] = random.uniform(200.0, 1000.0)
        elif anomaly_type == 'iot':
            row["iot_pdr"] = random.uniform(50.0, 80.0)
            row["iot_sensor_temp"] = random.uniform(70.0, 95.0)
        elif anomaly_type == 'fin':
            row["fin_db_query_duration"] = random.uniform(50.0, 200.0)
            row["fin_trans_vol"] = random.uniform(500.0, 1500.0) # sudden spike
            
        data.append(row)
        
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(output_path, index=False)
    print(f"Generated {num_rows} rows of simulated enterprise data to {output_path}")

if __name__ == "__main__":
    generate_enterprise_data()
