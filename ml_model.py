import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Load your vehicle count data from CSV
data = pd.read_csv("vehicle_counts.csv")

# Define Features (X) and Target (y)
X = data[["Cars", "Motorcycles", "Buses/Trucks"]]
y = data["predicted_green_light_time"]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train the RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate using Mean Absolute Error (MAE)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.2f} seconds")

# Save the trained model
with open("traffic_signal_model.pkl", "wb") as f:
    pickle.dump(model, f)
    print("Model saved to traffic_signal_model.pkl")

# Make predictions on the entire dataset and update the CSV
data['predicted_green_light_time'] = model.predict(X_scaled)
data.to_csv("vehicle_counts.csv", index=False)
print("CSV file updated with predicted green light times")
