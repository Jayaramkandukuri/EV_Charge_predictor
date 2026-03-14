"""
STEP 1 — Download Real EV Dataset
===================================
This script tries to download REAL EV charging data from Kaggle.

Dataset used:
  Name   : Electric Vehicle Charging Patterns
  Source : https://www.kaggle.com/datasets/valakhorasani/electric-vehicle-charging-patterns
  Size   : ~1000 real EV charging sessions
  Columns: User_ID, Vehicle_Model, Battery_Capacity_kWh, Charging_Station_ID,
           Charging_Station_Location, Start_Time, End_Time, Energy_Consumed_kWh,
           Charging_Duration_Hours, Charging_Rate_kW, Charging_Cost_USD,
           Time_of_Day, Day_of_Week, State_of_Charge_Start_%,
           State_of_Charge_End_%, Distance_Driven_km, Temperature_C,
           Vehicle_Age_years, Charger_Type, User_Type

HOW TO DOWNLOAD:
  Option A (Automatic - Kaggle API):
    1. Go to https://www.kaggle.com/settings → API → Create New Token
    2. Save kaggle.json to  C:/Users/<you>/.kaggle/kaggle.json  (Windows)
                         or ~/.kaggle/kaggle.json               (Linux/Mac)
    3. Run:  python download_dataset.py

  Option B (Manual - No account needed):
    1. Visit: https://www.kaggle.com/datasets/valakhorasani/electric-vehicle-charging-patterns
    2. Click Download
    3. Unzip and place  ev_charging_patterns.csv  inside the  data/  folder
    4. Run:  python download_dataset.py  (it will detect the file automatically)
"""

import os
import sys
import pandas as pd

RAW_CSV     = "data/ev_charging_patterns.csv"
CLEAN_CSV   = "data/ev_dataset.csv"

os.makedirs("data", exist_ok=True)


# ── Try Kaggle API download ───────────────────────────────────
def try_kaggle_download():
    try:
        import kaggle
        print("📥 Kaggle API found — downloading dataset...")
        os.makedirs("data/raw", exist_ok=True)
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "valakhorasani/electric-vehicle-charging-patterns",
            path="data/raw",
            unzip=True,
        )
        # Find the CSV
        for f in os.listdir("data/raw"):
            if f.endswith(".csv"):
                os.rename(f"data/raw/{f}", RAW_CSV)
                print(f"✅ Downloaded: {f}")
                return True
    except Exception as e:
        print(f"⚠️  Kaggle API not available: {e}")
    return False


# ── Check if file already exists ─────────────────────────────
def find_existing():
    candidates = [
        RAW_CSV,
        "data/electric_vehicle_charging_patterns.csv",
        "data/ev_charging_patterns.csv",
        "ev_charging_patterns.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"✅ Found existing dataset: {path}")
            return path
    return None


# ── Clean & standardize columns ──────────────────────────────
def clean_and_save(source_path):
    print(f"\n🔧 Cleaning dataset: {source_path}")
    df = pd.read_csv(source_path)

    print(f"   Raw shape: {df.shape}")
    print(f"   Raw columns: {list(df.columns)}")

    # ── Normalize column names (handles different versions of the dataset)
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("(", "").str.replace(")", "")

    col_map = {
        # Kaggle: valakhorasani/electric-vehicle-charging-patterns
        "Battery_Capacity_kWh":       "battery_capacity_kwh",
        "Charging_Rate_kW":           "charger_power_kw",
        "State_of_Charge_Start_%":    "initial_soc_pct",
        "State_of_Charge_End_%":      "target_soc_pct",
        "Charging_Duration_Hours":    "charging_time_hours",
        "Energy_Consumed_kWh":        "energy_required_kwh",
        "Temperature_C":              "temperature_c",
        "Vehicle_Age_years":          "vehicle_age_years",
        "Charging_Cost_USD":          "charging_cost_usd",
        "Distance_Driven_km":         "distance_driven_km",
        "Charger_Type":               "charger_type",
        "User_Type":                  "user_type",
        "Vehicle_Model":              "vehicle_model",
        "Time_of_Day":                "time_of_day",
        "Day_of_Week":                "day_of_week",
        "User_ID":                    "user_id",
    }

    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # ── Derive required columns if missing ──────────────────────
    required = ["battery_capacity_kwh", "charger_power_kw",
                "initial_soc_pct", "target_soc_pct",
                "charging_time_hours", "energy_required_kwh", "temperature_c"]

    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"❌ Missing required columns after mapping: {missing}")
        print("   Available columns:", list(df.columns))
        sys.exit(1)

    # ── Hour of day from time_of_day string ──────────────────────
    if "hour_of_day" not in df.columns:
        if "time_of_day" in df.columns:
            time_map = {"Morning": 8, "Afternoon": 14, "Evening": 19, "Night": 23}
            df["hour_of_day"] = df["time_of_day"].map(time_map).fillna(12).astype(int)
        else:
            df["hour_of_day"] = 12

    # ── Vehicle weight proxy from model if missing ───────────────
    if "vehicle_weight_kg" not in df.columns:
        weight_map = {
            "Tesla Model 3": 1844, "Tesla Model S": 2241,
            "Tesla Model X": 2509, "Tesla Model Y": 1979,
            "Nissan Leaf":   1598, "Chevy Bolt":    1625,
            "BMW i3":        1195, "Audi e-tron":   2585,
            "Hyundai Ioniq": 1565, "Kia EV6":       1960,
        }
        if "vehicle_model" in df.columns:
            df["vehicle_weight_kg"] = df["vehicle_model"].map(weight_map).fillna(1800)
        else:
            df["vehicle_weight_kg"] = 1800

    # ── Battery age in cycles proxy from vehicle age ─────────────
    if "battery_age_cycles" not in df.columns:
        if "vehicle_age_years" in df.columns:
            df["battery_age_cycles"] = (df["vehicle_age_years"] * 200).clip(0, 1500).astype(int)
        else:
            df["battery_age_cycles"] = 300

    # ── Estimated cost in INR ────────────────────────────────────
    df["estimated_cost_inr"] = (df["energy_required_kwh"] * 8).round(2)

    # ── Drop rows with bad values ────────────────────────────────
    df = df.dropna(subset=required)
    df = df[df["charging_time_hours"] > 0]
    df = df[df["battery_capacity_kwh"] > 0]
    df = df[df["initial_soc_pct"] < df["target_soc_pct"]]
    df = df.reset_index(drop=True)

    # ── Save cleaned ─────────────────────────────────────────────
    final_cols = [
        "battery_capacity_kwh", "charger_power_kw",
        "initial_soc_pct", "target_soc_pct",
        "charging_time_hours", "energy_required_kwh",
        "temperature_c", "battery_age_cycles",
        "vehicle_weight_kg", "hour_of_day",
        "estimated_cost_inr",
    ]
    # Add optional label cols if present
    for c in ["vehicle_model", "charger_type", "user_type", "day_of_week"]:
        if c in df.columns:
            final_cols.append(c)

    df = df[[c for c in final_cols if c in df.columns]]
    df.to_csv(CLEAN_CSV, index=False)

    print(f"\n✅ Cleaned dataset saved → {CLEAN_CSV}")
    print(f"   Rows   : {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print(f"\n📊 Key Statistics:")
    print(df[["battery_capacity_kwh","charger_power_kw",
              "initial_soc_pct","target_soc_pct",
              "charging_time_hours","energy_required_kwh",
              "temperature_c"]].describe().round(2).to_string())


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("  EV Dataset Downloader & Cleaner")
    print("="*60)

    if os.path.exists(CLEAN_CSV):
        print(f"✅ Cleaned dataset already exists: {CLEAN_CSV}")
        df = pd.read_csv(CLEAN_CSV)
        print(f"   Shape: {df.shape}")
        sys.exit(0)

    source = find_existing()

    if source is None:
        print("\n📥 No local file found. Trying Kaggle API...")
        downloaded = try_kaggle_download()
        if downloaded:
            source = RAW_CSV
        else:
            print("\n" + "="*60)
            print("  ❌ DATASET NOT FOUND")
            print("="*60)
            print("""
Please download the dataset manually:

  1. Go to:
     https://www.kaggle.com/datasets/valakhorasani/electric-vehicle-charging-patterns

  2. Click the [Download] button (free Kaggle account required)

  3. Unzip and copy  ev_charging_patterns.csv  into the  data/  folder

  4. Run this script again:
     python download_dataset.py

OR install kaggle API and set kaggle.json:
  pip install kaggle
  # Place kaggle.json in ~/.kaggle/
  python download_dataset.py
""")
            sys.exit(1)

    clean_and_save(source)
