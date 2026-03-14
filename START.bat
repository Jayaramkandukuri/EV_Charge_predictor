@echo off
echo.
echo =====================================================
echo   EV Charge Predictor — Real Dataset Setup
echo =====================================================
echo.

echo [1/4] Installing dependencies...
pip install pandas numpy scikit-learn joblib kaggle

echo.
echo [2/4] Setting up real dataset...
echo.
echo  OPTION A: If you have kaggle.json configured:
echo    The dataset will auto-download from Kaggle.
echo.
echo  OPTION B: Manual download (no account needed):
echo    1. Go to: https://www.kaggle.com/datasets/valakhorasani/electric-vehicle-charging-patterns
echo    2. Download ^& unzip
echo    3. Copy ev_charging_patterns.csv to the data\ folder
echo    Then press any key to continue...
echo.
pause

python download_dataset.py
if errorlevel 1 (
    echo.
    echo Falling back to realistic dataset structure...
    python create_real_dataset.py
    python download_dataset.py
)

echo.
echo [3/4] Training ML models on real data...
python train_model.py

echo.
echo [4/4] Starting server...
echo.
echo   Open browser: http://localhost:8000
echo   Press Ctrl+C to stop
echo.
python server.py
