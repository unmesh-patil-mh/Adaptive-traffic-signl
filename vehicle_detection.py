import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Configuration constants
TRAINING_DATA_PATH = "training_data.csv"
VEHICLE_MODEL_PATH = "yolo-weights/yolo11x.pt"
AMBULANCE_MODEL_PATH = "best.pt"
OUTPUT_CSV = "final_lane_output.csv"
CYCLE_TIME = 120
MIN_GREEN_TIME = 10
MAX_GREEN_TIME = 90

def load_and_train_model():
    """Load training data and train Random Forest model"""
    print("Loading training data from", TRAINING_DATA_PATH)
    if not os.path.exists(TRAINING_DATA_PATH):
        raise FileNotFoundError(f"Training data not found at {TRAINING_DATA_PATH}")
    
    training_df = pd.read_csv(TRAINING_DATA_PATH)
    
    # Prepare features and targets
    features = training_df[[
        f'{vt}_lane{i}'  
        for vt in ['Cars', 'Motorcycles', 'BusesTrucks']
        for i in range(1,5)]]
    targets = training_df[[f'GreenTime_lane{i}' for i in range(1,5)]]
    
    # Train/test split and model training
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Validate model
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Trained Model - Mean Absolute Error: {mae:.2f} seconds")
    
    return rf_model

def detect_vehicles(image_path, vehicle_model, ambulance_model):
    """Detect vehicles and ambulances in a single image"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Initialize counts
    counts = {"car": 0, "motorcycle": 0, "bus/truck": 0, "ambulance": 0}
    emergency = False
    
    # Vehicle detection
    vehicle_classes = {"car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
    results = vehicle_model(img)
    for box in results[0].boxes:
        cls = int(box.cls[0])
        if cls == vehicle_classes["car"]:
            counts["car"] += 1
        elif cls == vehicle_classes["motorcycle"]:
            counts["motorcycle"] += 1
        elif cls in [vehicle_classes["bus"], vehicle_classes["truck"]]:
            counts["bus/truck"] += 1
    
    # Ambulance detection
    ambulance_results = ambulance_model(img)
    for box in ambulance_results[0].boxes:
        if float(box.conf[0]) > 0.61:
            counts["ambulance"] += 1
            emergency = True
    
    return counts, emergency

def process_lanes(lane_images, rf_model):
    """Process all lanes and generate predictions"""
    # Load models
    vehicle_model = YOLO(VEHICLE_MODEL_PATH)
    ambulance_model = YOLO(AMBULANCE_MODEL_PATH)
    
    # Initialize data storage
    lane_data = {}
    
    # Process each lane
    for lane, image_path in lane_images.items():
        print(f"Processing {lane}...")
        counts, emergency = detect_vehicles(image_path, vehicle_model, ambulance_model)
        lane_data[lane] = {
            "counts": counts,
            "emergency": emergency
        }
    
    # Prepare features for prediction
    features = []
    lane_order = ["lane_1", "lane_2", "lane_3", "lane_4"]
    for lane in lane_order:
        counts = lane_data[lane]["counts"]
        features.extend([counts["car"], counts["motorcycle"], counts["bus/truck"]])
    
    # Make predictions
    predicted_times = rf_model.predict(np.array(features).reshape(1, -1))[0]
    predicted_times = np.clip(predicted_times, MIN_GREEN_TIME, MAX_GREEN_TIME)
    
    # Prepare output data
    output_data = []
    for i, lane in enumerate(lane_order):
        output_data.append({
            "Lane": lane,
            "Cars": lane_data[lane]["counts"]["car"],
            "Motorcycle": lane_data[lane]["counts"]["motorcycle"],
            "Trucks_Buses": lane_data[lane]["counts"]["bus/truck"],
            "Ambulance": lane_data[lane]["counts"]["ambulance"],
            "Emergency": "Yes" if lane_data[lane]["emergency"] else "No",
            "Green_Light_Time": predicted_times[i]
        })
    
    # Save to CSV
    pd.DataFrame(output_data).to_csv(OUTPUT_CSV, index=False, float_format="%.2f")
    print(f"\nFinal CSV generated: {OUTPUT_CSV}")
    
    return output_data

if __name__ == "__main__":
    # Example usage
    lane_images = {
        "lane_1": "input_images/lane1.jpg",
        "lane_2": "input_images/lane2.jpg",
        "lane_3": "input_images/lane3.jpg",
        "lane_4": "input_images/lane4.jpg"
    }
    
    # Train model
    rf_model = load_and_train_model()
    
    # Process lanes and generate predictions
    results = process_lanes(lane_images, rf_model)
    
    # Print results
    print("\nPredicted Green Light Times (seconds):")
    for lane in results:
        print(f"{lane['Lane']}: {lane['Green_Light_Time']} seconds")