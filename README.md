# ⚡ EV Charge Predictor

---

⚡ EV Charge Predictor

📌 What is this project?
This is a Machine Learning web application that predicts:

⏱️ How long it takes to charge an Electric Vehicle
⚡ How much electricity (energy) it will use
₹ How much it will cost in Indian Rupees

You select your car, charger type, battery level — and the app gives you
the prediction using a real trained ML model.

---

## 🗂️ Project Structure

```
EV_Charge_Predictor/
│
├── 📁 data/
│   ├── ev_charging_patterns.csv   ← Raw Kaggle dataset (1000 rows, 19 columns)
│   └── ev_dataset.csv             ← Cleaned dataset (run download_dataset.py to generate this)
│
├── 📁 models/                     ← Trained ML models (auto-generated)
│   ├── time_model.pkl             ← Gradient Boosting — predicts charge time
│   ├── energy_model.pkl           ← Random Forest — predicts energy used
│   ├── features.pkl               ← Feature list used by models
│   └── model_name.pkl             ← Best model name
│
├── 📁 templates/
│   └── index.html                 ← Complete website (HTML + CSS + JS)
│
├── 📁 static/
│   ├── css/style.css              ← Website design and styling
│   └── js/main.js                 ← Website interactivity and API calls
│
├── download_dataset.py            ← Cleans raw CSV into ML-ready format
├── create_real_dataset.py         ← Fallback if Kaggle CSV not available
├── train_model.py                 ← Trains and saves ML models
├── server.py                      ← Python HTTP server + prediction API
├── requirements.txt               ← Python dependencies
├── START.bat                      ← Windows one-click run
└── README.md                      ← This file
```

---

## 🗂️ Dataset

**Source:** [Electric Vehicle Charging Patterns — Kaggle](https://www.kaggle.com/datasets/valakhorasani/electric-vehicle-charging-patterns)

**1,000 real EV charging sessions** with these details:

|           Column       |          Description               |
|------------------------|------------------------------------|
| Battery Capacity (kWh) | Size of the EV battery |
| Charging Rate (kW) | Speed of the charger used |
| State of Charge Start % | Battery level when plugged in |
| State of Charge End % | Battery level when unplugged |
| Charging Duration (hours) | How long it took — **Target Y₁** |
| Energy Consumed (kWh) | Electricity used — **Target Y₂** |
| Temperature (°C) | Ambient temperature |
| Vehicle Age (years) | Age of the vehicle |
| Vehicle Model | Tesla, Nissan, Audi, Hyundai etc. |
| Charger Type | Level 1 / Level 2 / DC Fast |
| Time of Day | Morning / Afternoon / Evening / Night |

### What is Data Cleaning?

The raw Kaggle CSV has messy column names (spaces, brackets, special characters)
and some columns useless for ML (User ID, Station ID etc.).

`download_dataset.py` fixes all of this:

|        Raw Column        |    After Cleaning      |    Action       |
|--------------------------|------------------------|-------------  --|
| `Battery Capacity (kWh)` | `battery_capacity_kwh` | Renamed |
| `State of Charge (Start %)` | `initial_soc_pct` | Renamed |
| `Charging Duration (hours)` | `charging_time_hours` | **Target Y₁** |
| `Energy Consumed (kWh)` | `energy_required_kwh` | **Target Y₂** |
| `Temperature (Â°C)` | `temperature_c` | Fixed broken encoding |
| `Vehicle Age (years)` | `battery_age_cycles` | Converted (×200 cycles/yr) |
| `Time of Day` | `hour_of_day` | Morning→8, Evening→19, Night→23 |
| `User ID` | — | ❌ Dropped (useless for ML) |
| `Charging Station ID` | — | ❌ Dropped (useless for ML) |
| `Charging Start/End Time` | — | ❌ Dropped (replaced by hour_of_day) |
| `Charging Cost (USD)` | — | ❌ Dropped (recalculated in ₹) |

---

## 🧠 Machine Learning

### What is ML in Simple Words?

Instead of writing rules manually like:
```
IF charger=50kW AND battery=75kWh AND soc=20% THEN time=1.2hrs
```
We give **1000 real examples** and let the model find the pattern itself.

### Feature Engineering — 15 Input Features

From the cleaned columns we create extra calculated features:

| Feature     |     How Calculated |         Why            |
|-------------|--------------------|---------------------|
| `soc_delta` | target % − start % | Exact charge needed |
| `energy_delta` | battery × soc_delta / 100 | kWh to be added |
| `power_cap_ratio` | charger_kw / battery_kwh | Relative charge speed |
| `temp_deviation` | abs(temp − 20) | 20°C is optimal |
| `is_fast_charger` | charger ≥ 50kW → 1 else 0 | DC vs AC flag |
| `is_night` | hour ≥ 22 or ≤ 6 → 1 else 0 | Off-peak flag |
| `high_soc_charge` | target ≥ 90% → 1 else 0 | Taper effect near 100% |

### Model Comparison

| Model | Accuracy (R²) |
|---|---|
| Linear Regression | 70.7% |
| Random Forest | 99.1% |
| **Gradient Boosting ✅ Winner** | **99.4%** |

### Top Feature Importances

```
power_cap_ratio      ████████████████████████████████████  76%
energy_delta         ████████                              12%
soc_delta            █████                                  8%
charger_power_kw     ██                                     3%
```

---

## 🖥️ Website

### Tab 1 — Predictor (Main Page)

**Left Panel — Your Inputs:**
- EV Model (Tesla Model 3/Y/S, Nissan Leaf, Chevy Bolt,
  Audi e-tron, Hyundai Ioniq 5, Kia EV6)
- Charger Type (3.3kW → 150kW)
- Current Battery % slider
- Target Battery % slider
- Temperature slider
- Battery Age (cycles) slider
- Hour of Day slider
- ⚡ Run Prediction button

**Right Panel — Results (after clicking Run Prediction):**
- Charging Time, Energy Required, Estimated Cost cards
- Battery bar (shows current % and what will be added)
- Charger comparison chart (all 5 charger types compared)
- Smart insights (temperature warning, battery health, night tip)
- Model info (algorithm, R² score, dataset size)

### Tab 2 — Dataset

Shows all real EV charging records from the Kaggle dataset in a table.
Click **"Load Records"** to fetch them from the Python backend.

### Tab 3 — Analytics

Shows statistics computed from the entire dataset:
- Total sessions, avg charge time, avg energy, avg cost
- Charger type distribution (bar chart)
- EV model distribution

---

## 🚗 EV Models and Specs

| Shown on Screen | Full Name | Battery | Weight |
|---|---|---|---|
| Tesla Model 3 | Tesla Model 3 | 75 kWh | 1844 kg |
| Tesla Model Y | Tesla Model Y | 75 kWh | 1979 kg |
| Tesla Model S | Tesla Model S | 100 kWh | 2241 kg |
| Nissan Leaf | Nissan Leaf | 40 kWh | 1598 kg |
| Chevy Bolt | Chevrolet Bolt EV | 65 kWh | 1625 kg |
| Audi e-tron | Audi e-tron | 95 kWh | 2585 kg |
| **Ioniq 5** | **Hyundai Ioniq 5** | 72.6 kWh | 1985 kg |
| Kia EV6 | Kia EV6 | 77.4 kWh | 1960 kg |


---

## 🔁 How It All Connects

```
Raw Kaggle CSV (19 messy columns)
         ↓  download_dataset.py  (cleaning)
Clean CSV (15 neat columns)
         ↓  train_model.py  (learning)
models/*.pkl  (trained ML brain saved)
         ↓  server.py  (serving)
http://localhost:8000  (website running)
         ↓  you open browser
Select EV + Charger + Sliders
         ↓  click Run Prediction
JavaScript → POST /api/predict → server.py
         ↓
15 features built from your inputs
         ↓
Gradient Boosting model predicts
         ↓
Result: Time · Energy · Cost shown on screen

---

## 🚀 Installation and Setup

### Prerequisites
- Python 3.8 or higher installed
- pip working
- Git installed

### Step 1 — Clone the Repository
```bash
git clone https://github.com/yourusername/ev-charge-predictor.git
cd ev-charge-predictor
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Clean the Dataset
```bash
python download_dataset.py
```
This reads `data/ev_charging_patterns.csv` and creates `data/ev_dataset.csv`

If the raw CSV is missing, run this first:
```bash
python create_real_dataset.py
python download_dataset.py
```

### Step 4 — Train the ML Model
```bash
python train_model.py
```
This reads the clean CSV, trains Gradient Boosting,
and saves models to `models/` folder.
Run time: ~30 seconds.

### Step 5 — Start the Server
```bash
python server.py
```

### Step 6 — Open in Browser
```
http://localhost:8000
```


## 📈 Results

```
Dataset       : 1,000 real EV charging sessions
Best Model    : Gradient Boosting Regressor
Accuracy      : R² = 0.9943 (Charging Time Prediction)
               R² = 0.9972 (Energy Consumption Prediction)
MAE           : 0.20 hours (~12 minutes average error)
Train/Test    : 800 records train / 200 records test
Features      : 15 (8 from dataset + 7 engineered)
```

---

## 🔮 Future Improvements

- [ ] LSTM model for time-series battery prediction
- [ ] Real charging station map integration
- [ ] Mobile responsive design
- [ ] Deploy on cloud (Railway / Render / AWS)
- [ ] IoT integration with ESP32 + sensors
- [ ] Battery degradation prediction module
- [ ] Multi-language support

---

## 👤 Author

**Jayaram Kandukuri**
B.Tech [ECE] | [SRM University AP]
📧 jayaramknss@gmail.com


🌐 [Link](https://ev-charge-predictor.onrender.com/)
