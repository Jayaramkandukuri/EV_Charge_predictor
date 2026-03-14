"""
STEP 2 — Train ML Models on Real EV Dataset
=============================================
Reads the cleaned real Kaggle EV dataset and trains:
  - Random Forest
  - Gradient Boosting  ← Best model
Saves trained models to models/
"""

import pandas as pd
import numpy as np
import joblib, os, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

os.makedirs("models", exist_ok=True)

# ── Load Real Dataset ─────────────────────────────────────────
CSV = "data/ev_dataset.csv"
if not os.path.exists(CSV):
    print("❌ Cleaned dataset not found!")
    print("   Run:  python download_dataset.py  first")
    exit(1)

df = pd.read_csv(CSV)
print("="*60)
print("  Training on REAL EV Charging Dataset")
print("="*60)
print(f"  Records : {len(df)}")
print(f"  Columns : {list(df.columns)}")

# ── Feature Engineering ───────────────────────────────────────
df["soc_delta"]         = df["target_soc_pct"] - df["initial_soc_pct"]
df["energy_delta"]      = df["battery_capacity_kwh"] * df["soc_delta"] / 100
df["power_cap_ratio"]   = df["charger_power_kw"] / df["battery_capacity_kwh"]
df["temp_deviation"]    = abs(df["temperature_c"] - 20)
df["is_fast_charger"]   = (df["charger_power_kw"] >= 50).astype(int)
df["is_night"]          = ((df["hour_of_day"] >= 22) | (df["hour_of_day"] <= 6)).astype(int)
df["high_soc_charge"]   = (df["target_soc_pct"] >= 90).astype(int)  # taper effect

FEATURES = [
    "battery_capacity_kwh", "charger_power_kw",
    "initial_soc_pct",      "target_soc_pct",
    "soc_delta",            "energy_delta",
    "power_cap_ratio",      "temperature_c",
    "temp_deviation",       "battery_age_cycles",
    "vehicle_weight_kg",    "hour_of_day",
    "is_fast_charger",      "is_night",
    "high_soc_charge",
]

X       = df[FEATURES]
y_time  = df["charging_time_hours"]
y_enrg  = df["energy_required_kwh"]

X_tr, X_te, yt_tr, yt_te, ye_tr, ye_te = train_test_split(
    X, y_time, y_enrg, test_size=0.2, random_state=42
)

print(f"\n  Train: {len(X_tr)} records  |  Test: {len(X_te)} records")

# ── Train Models ──────────────────────────────────────────────
print("\n" + "="*60)
print("  CHARGING TIME PREDICTION")
print("="*60)
print(f"  {'Model':<24} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("-"*60)

candidates = {
    "Linear Regression":  LinearRegression(),
    "Random Forest":      RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Gradient Boosting":  GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
}

best_r2, best_model, best_name = -999, None, ""

for name, m in candidates.items():
    m.fit(X_tr, yt_tr)
    pred = m.predict(X_te)
    mae  = mean_absolute_error(yt_te, pred)
    rmse = np.sqrt(mean_squared_error(yt_te, pred))
    r2   = r2_score(yt_te, pred)
    print(f"  {name:<24} {mae:>8.4f} {rmse:>8.4f} {r2:>8.4f}")
    if r2 > best_r2:
        best_r2, best_model, best_name = r2, m, name

print(f"\n  🏆 Best: {best_name}  (R² = {best_r2:.4f})")

# Energy model
print("\n" + "="*60)
print("  ENERGY CONSUMPTION PREDICTION")
print("="*60)
e_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
e_model.fit(X_tr, ye_tr)
ep = e_model.predict(X_te)
print(f"  Random Forest  R² = {r2_score(ye_te, ep):.4f}  MAE = {mean_absolute_error(ye_te, ep):.4f}")

# ── Feature Importance ─────────────────────────────────────────
if hasattr(best_model, "feature_importances_"):
    fi = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print(f"\n  Top Features ({best_name}):")
    for feat, score in fi.head(7).items():
        bar = "█" * int(score * 60)
        print(f"    {feat:<22} {bar} {score:.4f}")

# ── Save ──────────────────────────────────────────────────────
joblib.dump(best_model, "models/time_model.pkl")
joblib.dump(e_model,    "models/energy_model.pkl")
joblib.dump(FEATURES,   "models/features.pkl")
joblib.dump(best_name,  "models/model_name.pkl")

print("\n✅ Models saved → models/")
print(f"   time_model.pkl   ({best_name}, R²={best_r2:.4f})")
print(f"   energy_model.pkl (Random Forest, R²={r2_score(ye_te, ep):.4f})")
