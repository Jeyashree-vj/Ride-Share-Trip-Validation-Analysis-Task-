
# 0. Setup

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import shap


# 1. Load Data

df = pd.read_csv("/content/rideshare_trips.csv", on_bad_lines='skip')


# 2. Feature Engineering

df['speed_kmh'] = df['distance_km'] / (df['duration_mins'] / 60)
df['pickup_time'] = pd.to_datetime(df['pickup_time'], errors='coerce')
df['pickup_hour'] = df['pickup_time'].dt.hour
df['pickup_day'] = df['pickup_time'].dt.dayofweek
df['is_peak_hour'] = df['pickup_hour'].between(7, 9) | df['pickup_hour'].between(17, 19)
df['trip_density'] = df['distance_km'] / df['duration_mins']
df['fare_calc'] = df['base_fare'] * df['surge_multiplier']
df['fare_mismatch'] = (df['final_fare'] != df['fare_calc']).astype(int)
df['off_peak_surge'] = ((df['surge_multiplier'] > 1) & ~df['is_peak_hour']).astype(int)
df['cancelled'] = df['status'].str.contains('cancelled', case=False, na=False).astype(int)
df['missing_gps'] = ((df['dropoff_lat'].isnull()) | (df['dropoff_lon'].isnull())).astype(int)
df['no_rider'] = df['rider_id'].isnull().astype(int)


# 3. Trip Trust Score Logic
df['trust_score'] = 100

# Smooth penalties based on severity
df['trust_score'] -= np.clip((df['speed_kmh'] - 80) / 2, 0, 30)
df['trust_score'] -= np.clip((0.2 - df['distance_km']) * 100, 0, 20)
df['trust_score'] -= np.clip((5 - df['duration_mins']) * 5, 0, 25)
df['trust_score'] -= np.clip((df['surge_multiplier'] - 1) * 5, 0, 15)
df['trust_score'] -= df['fare_mismatch'] * 10
df['trust_score'] -= df['off_peak_surge'] * 10
df['trust_score'] -= df['cancelled'] * df['missing_gps'] * 20
df['trust_score'] -= df['cancelled'] * (df['final_fare'] <= 0).astype(int) * 10
df['trust_score'] -= df['no_rider'] * 20



# 4. Prepare Model Data

features = ['distance_km', 'duration_mins', 'speed_kmh', 'surge_multiplier',
            'fare_mismatch', 'off_peak_surge', 'cancelled', 'missing_gps', 'no_rider',
            'pickup_day', 'is_peak_hour', 'trip_density']

df_model = df[features + ['trust_score']].dropna()
X = df_model[features].astype(float)
y = df_model['trust_score']


# 5. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 6. Hyperparameter Tuning

params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1]
}

grid = GridSearchCV(XGBRegressor(), params, cv=3)
grid.fit(X_train, y_train)

# 7. Evaluate Model

best_model = grid.best_estimator_
preds = best_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))


print("Best Parameters:", grid.best_params_)
print("RMSE:", rmse )



# 8. SHAP Explainability

explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_test)

shap.plots.beeswarm(shap_values)
shap.plots.waterfall(shap_values[0])


# 9. Visualizations

plt.figure(figsize=(8,6))
sns.histplot(df['surge_multiplier'].dropna(), bins=10, kde=True)
plt.title('Surge Multiplier Distribution')
plt.xlabel('Surge Multiplier')
plt.ylabel('Number of Trips')
plt.show()

plt.figure(figsize=(8,6))
sns.scatterplot(x='distance_km', y='speed_kmh', data=df, hue=(df['speed_kmh'] > 120))
plt.title('Speed vs Distance (GPS Anomalies)')
plt.xlabel('Distance (km)')
plt.ylabel('Speed (km/h)')
plt.show()


# 10. Final Summary

df_output = df[['trip_id', 'trust_score']].sort_values(by='trust_score')
print("=== Top 10 Lowest Trust Score Trips ===")
print(df_output.head(10))

print("\n=== Actionable Recommendations ===")
print("- Review trips flagged with unrealistic speed or minimal movement for possible GPS errors or fraud.")
print("- Investigate trips with excessive surge multipliers, especially off-peak.")
print("- Ensure all cancelled trips have complete cancellation details and proper fee application.")
print("- Monitor repeated cancellations between the same rider-driver pairs for possible no-show patterns.")
print("- Address missing GPS, distance, and duration fields for data integrity.")


# 11. Print Anomaly Tables

# GPS anomalies
df['gps_flag'] = np.where(df['speed_kmh'] > 120, 'Unrealistic speed',
                  np.where((df['distance_km'] < 0.2) & (df['duration_mins'] > 10), 'No movement',
                  np.where((df['duration_mins'] < 2) & (df['distance_km'] > 5), 'Duration mismatch', None)))
gps_anomalies = df[df['gps_flag'].notnull()][['trip_id', 'distance_km', 'duration_mins', 'speed_kmh', 'gps_flag']]
print("\n=== GPS Anomalies ===")
print(gps_anomalies.head())

# Surge disputes
df['surge_dispute_flag'] = np.where(df['surge_multiplier'] > 3, 'Excessive surge',
                             np.where(df['fare_mismatch'] == 1, 'Fare mismatch',
                             np.where(df['off_peak_surge'] == 1, 'Off-peak surge', None)))
surge_disputes = df[df['surge_dispute_flag'].notnull()][['trip_id', 'pickup_hour', 'surge_multiplier', 'final_fare', 'fare_calc', 'surge_dispute_flag']]
print("\n=== Surge Pricing Disputes ===")
print(surge_disputes.head())

# Cancellations
df['cancellation_fee_applied'] = ~df['final_fare'].isnull() & (df['final_fare'] > 0)
df['cancel_flag'] = np.where((df['cancelled']) & ((df['dropoff_lat'].isnull()) | (df['dropoff_lon'].isnull())), 'Incomplete info',
                     np.where((df['cancelled']) & ~df['cancellation_fee_applied'], 'Fee missing', None))
cancellations = df[df['cancelled'] == 1][['trip_id', 'cancellation_by', 'final_fare', 'cancel_flag']]
print("\n=== Cancellations ===")
print(cancellations.head())

# No-shows
df['no_show_flag'] = np.where(df['no_rider'] == 1, 'No rider ID',
                       np.where((df['distance_km'] < 0.1) & (df['duration_mins'] < 2), 'Minimal movement', None))
no_shows = df[df['no_show_flag'].notnull()][['trip_id', 'driver_id', 'rider_id', 'distance_km', 'duration_mins', 'no_show_flag']]
print("\n=== No-Shows ===")
print(no_shows.head())
