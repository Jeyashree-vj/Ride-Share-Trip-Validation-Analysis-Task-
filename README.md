# Ride-Share Trip Validation & Trust Scoring

This project builds a hybrid rule-based + ML model to validate ride-share trips and assign a trust score (0–100). It flags GPS anomalies, surge pricing disputes, cancellations, and no-shows — with full explainability using SHAP.

## Features
- Trip-by-trip anomaly detection
- Custom Trust Score logic
- XGBoost regression model with hyperparameter tuning
- SHAP explainability (beeswarm + waterfall)
- Visualizations for surge and GPS anomalies

##  Files
- `src/Ride Share Trip Validation Analysis Task code.py`: Full pipeline code
- `data/rideshare_trips.csv`: Sample dataset
- `output/`: Model results and plots

## Model Performance
- RMSE: `9.78`
- Best Parameters: `{'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1}`

## How to Run
```bash
pip install -r requirements.txt
python src/Ride Share Trip Validation Analysis Task code.py
