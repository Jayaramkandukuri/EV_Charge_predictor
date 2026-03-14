"""
STEP 3 — Web Server
====================
Serves the HTML frontend + ML prediction API

Run  : python server.py
Open : http://localhost:8000
"""

import os, json
import numpy as np
import pandas as pd
import joblib
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# ── Load Models ───────────────────────────────────────────────
try:
    TIME_MODEL   = joblib.load("models/time_model.pkl")
    ENERGY_MODEL = joblib.load("models/energy_model.pkl")
    FEATURES     = joblib.load("models/features.pkl")
    MODEL_NAME   = joblib.load("models/model_name.pkl")
    MODELS_OK    = True
    print(f"✅ ML Models loaded — {MODEL_NAME}")
except Exception as e:
    MODELS_OK  = False
    MODEL_NAME = "Not loaded"
    print(f"❌ Models not found. Run train_model.py first.\n   {e}")


def make_features(d):
    soc_delta = d["target_soc"] - d["initial_soc"]
    row = {
        "battery_capacity_kwh": d["battery_capacity"],
        "charger_power_kw":     d["charger_power"],
        "initial_soc_pct":      d["initial_soc"],
        "target_soc_pct":       d["target_soc"],
        "soc_delta":            soc_delta,
        "energy_delta":         d["battery_capacity"] * soc_delta / 100,
        "power_cap_ratio":      d["charger_power"] / d["battery_capacity"],
        "temperature_c":        d["temperature"],
        "temp_deviation":       abs(d["temperature"] - 20),
        "battery_age_cycles":   d["battery_age"],
        "vehicle_weight_kg":    d["vehicle_weight"],
        "hour_of_day":          d["hour_of_day"],
        "is_fast_charger":      int(d["charger_power"] >= 50),
        "is_night":             int(d["hour_of_day"] >= 22 or d["hour_of_day"] <= 6),
        "high_soc_charge":      int(d["target_soc"] >= 90),
    }
    return pd.DataFrame([row])[FEATURES]


def do_predict(d):
    X      = make_features(d)
    t_hrs  = float(TIME_MODEL.predict(X)[0])
    e_kwh  = float(ENERGY_MODEL.predict(X)[0])
    cost   = round(e_kwh * 8, 2)
    h      = int(t_hrs)
    m      = int((t_hrs - h) * 60)
    soc_d  = d["target_soc"] - d["initial_soc"]

    health  = round(max(0, 100 - (d["battery_age"] / 1500) * 20), 1)
    temp_ef = round(max(70, 100 - abs(d["temperature"] - 20) * 0.5), 1)

    comparison = []
    for pw, lbl in [(3.3,"AC L1 3.3kW"), (7.2,"AC L2 7.2kW"), (22,"AC 22kW"), (50,"DC 50kW"), (150,"DC 150kW")]:
        tmp = dict(d); tmp["charger_power"] = pw
        t2  = float(TIME_MODEL.predict(make_features(tmp))[0])
        comparison.append({"label": lbl, "power": pw, "hours": round(t2, 2)})

    return {
        "time_hours":    round(t_hrs, 2),
        "time_display":  f"{h}h {m}m",
        "energy_kwh":    round(e_kwh, 2),
        "cost_inr":      cost,
        "soc_delta":     round(soc_d, 1),
        "health_score":  health,
        "temp_eff":      temp_ef,
        "comparison":    comparison,
        "model_used":    MODEL_NAME,
        "is_night":      bool(d["hour_of_day"] >= 22 or d["hour_of_day"] <= 6),
        "is_cold":       bool(d["temperature"] < 5),
        "is_hot":        bool(d["temperature"] > 37),
        "old_battery":   bool(d["battery_age"] > 1000),
    }


def get_history():
    try:
        df = pd.read_csv("data/ev_dataset.csv")
        sample = df.sample(min(len(df), len(df)), random_state=42)
        cols = ["battery_capacity_kwh","charger_power_kw","initial_soc_pct",
                "target_soc_pct","temperature_c","battery_age_cycles",
                "charging_time_hours","energy_required_kwh","estimated_cost_inr"]
        # Only use cols that actually exist
        cols = [c for c in cols if c in sample.columns]
        return sample[cols].round(2).to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}


def get_stats():
    try:
        df = pd.read_csv("data/ev_dataset.csv")
        stats = {
            "total":          len(df),
            "avg_time":       round(df["charging_time_hours"].mean(), 2),
            "avg_energy":     round(df["energy_required_kwh"].mean(), 2),
            "avg_cost":       round(df["estimated_cost_inr"].mean(), 2),
            "max_time":       round(df["charging_time_hours"].max(), 2),
            "min_time":       round(df["charging_time_hours"].min(), 2),
            "fast_pct":       round((df["charger_power_kw"] >= 50).mean() * 100, 1),
            "model_used":     MODEL_NAME,
        }
        if "charger_power_kw" in df.columns:
            dist = df["charger_power_kw"].value_counts().to_dict()
            stats["charger_dist"] = {str(k)+"kW": int(v) for k, v in dist.items()}
        if "vehicle_model" in df.columns:
            stats["vehicle_dist"] = df["vehicle_model"].value_counts().head(6).to_dict()
        return stats
    except Exception as e:
        return {"error": str(e)}


# ── HTTP Handler ──────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        print(f"  {self.address_string()} → {fmt % args}")

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, ctype):
        try:
            with open(path, "rb") as f:
                body = f.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        except FileNotFoundError:
            self.send_response(404); self.end_headers()
            self.wfile.write(b"Not found")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        p = urlparse(self.path).path
        if p in ("/", "/index.html"):
            self.send_file("templates/index.html", "text/html; charset=utf-8")
        elif p == "/static/css/style.css":
            self.send_file("static/css/style.css", "text/css")
        elif p == "/static/js/main.js":
            self.send_file("static/js/main.js", "application/javascript")
        elif p == "/api/history":
            self.send_json(get_history())
        elif p == "/api/stats":
            self.send_json(get_stats())
        elif p == "/api/status":
            self.send_json({"ok": MODELS_OK, "model": MODEL_NAME,
                            "dataset": os.path.exists("data/ev_dataset.csv")})
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        if self.path != "/api/predict":
            self.send_response(404); self.end_headers(); return
        if not MODELS_OK:
            self.send_json({"error": "Models not trained. Run train_model.py."}, 503)
            return
        try:
            n    = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n))
            self.send_json(do_predict(body))
        except Exception as e:
            self.send_json({"error": str(e)}, 400)


if __name__ == "__main__":
    PORT = 8000
    srv  = HTTPServer(("0.0.0.0", PORT), Handler)
    print("\n" + "═"*52)
    print("  🔋 EV CHARGE PREDICTOR")
    print("═"*52)
    print(f"  🌐  http://localhost:{PORT}")
    print(f"  🤖  Model : {MODEL_NAME}")
    print(f"  📊  Dataset: {'data/ev_dataset.csv' if os.path.exists('data/ev_dataset.csv') else 'NOT FOUND'}")
    print(f"  Ctrl+C to stop")
    print("═"*52 + "\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Stopped.")
