import time
import requests

# Endpoints
PROMETHEUS_URL = "http://localhost:9090"
GRAFANA_URL = "http://localhost:3000"
API_PREDICT_URL = "http://localhost:8000/predict"

# Sample clinical payload for heart disease risk prediction
sample_payload = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0.0,
    "thal": 1.0
}

print("=== MLOps Monitoring & Telemetry Verification ===")

# 1. Check Prometheus Health
try:
    prom_res = requests.get(PROMETHEUS_URL, timeout=3)
    if prom_res.status_code == 200:
        print(f"[✓] Prometheus is active and reachable at: {PROMETHEUS_URL}")
    else:
        print(f"[!] Prometheus responded with status code: {prom_res.status_code}")
except requests.exceptions.ConnectionError:
    print(f"[x] Prometheus is NOT running at {PROMETHEUS_URL}. Start your docker compose stack first.")

# 2. Check Grafana Health
try:
    graf_res = requests.get(GRAFANA_URL, timeout=3)
    if graf_res.status_code == 200:
        print(f"[✓] Grafana is active and reachable at: {GRAFANA_URL}")
    else:
        print(f"[!] Grafana responded with status code: {graf_res.status_code}")
except requests.exceptions.ConnectionError:
    print(f"[x] Grafana is NOT running at {GRAFANA_URL}. Start your docker compose stack first.")

# 3. Generate Live Traffic to Populate Charts
print("\nGenerating simulated inference traffic for Grafana graphs...")
success_count = 0

for i in range(1, 26):
    try:
        response = requests.post(API_PREDICT_URL, json=sample_payload, timeout=2)
        if response.status_code == 200:
            success_count += 1
            print(f"  -> Request {i}/25 sent successfully (Status: 200)")
        else:
            print(f"  -> Request {i}/25 failed with status: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print(f"  -> Error: FastAPI app not reachable at {API_PREDICT_URL}. Is the container running?")
        break
    time.sleep(0.1)  # Small pacing interval

print(f"\nTraffic generation complete: {success_count}/25 requests successfully processed.")
print("\n--- Next Steps to View Your Charts ---")
print(f"1. Open Grafana in your browser: {GRAFANA_URL}")
print("   (Default Login: username 'admin', password 'admin')")
print("2. Ensure Prometheus (http://prometheus:9090) is connected as your data source.")
print("3. Create or open your dashboard to view active Request Rate (RPS) and Latency graphs!")