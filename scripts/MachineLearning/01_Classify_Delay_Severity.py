import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import pickle
import os

# --- Configuration ---
DATA_PATH = 'data/processed/flights_cleaned_us_only.csv' #
MODEL_DIR = 'app/models'
os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_preprocess_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # 1. Define Target: Severe Delay (Cascade)
    # We want to predict if the arrival delay will be severe (> 60 min)
    df['is_severe_delay'] = (df['arr_delay_minutes'] > 60).astype(int)
    
    # 2. Rename 'dep_delay_minutes' to 'current_delay_minutes'
    # This aligns the training data with the telemetry stream in main.py
    if 'dep_delay_minutes' in df.columns:
        df = df.rename(columns={'dep_delay_minutes': 'current_delay_minutes'})
    
    # Fill NaNs
    df['current_delay_minutes'] = df['current_delay_minutes'].fillna(0)
    
    # 3. Feature Engineering: Top Airports
    # Instead of dropping Origin/Dest, let's keep the top 20 busiest airports
    top_airports = df['ORIGIN_AIRPORT_ID'].value_counts().head(20).index.tolist()
    
    # Create a feature for "Is Origin a Major Hub?"
    # (Simplified approach: One-Hot Encoding top airports)
    for airport_id in top_airports:
        df[f'origin_{airport_id}'] = (df['ORIGIN_AIRPORT_ID'] == airport_id).astype(int)
        
    # 4. Feature Engineering: Time
    df['scheduled_dep_utc'] = pd.to_datetime(df['scheduled_dep_utc'])
    df['hour_of_day'] = df['scheduled_dep_utc'].dt.hour
    df['day_of_week'] = df['scheduled_dep_utc'].dt.dayofweek
    df['month'] = df['scheduled_dep_utc'].dt.month
    
    # 5. Feature Engineering: Carriers
    # One-hot encode common carriers
    common_carriers = ['AA', 'DL', 'UA', 'WN', 'B6']
    for carrier in common_carriers:
        if 'MKT_UNIQUE_CARRIER' in df.columns:
             df[f'carrier_{carrier}'] = (df['MKT_UNIQUE_CARRIER'] == carrier).astype(int)

    # 6. Select Features
    # Ensure these match exactly what main.py expects/generates
    feature_cols = [
        'current_delay_minutes',
        'hour_of_day', 
        'day_of_week', 
        'month'
    ]
    
    # Add the dynamic airport/carrier columns
    feature_cols.extend([c for c in df.columns if c.startswith('origin_')])
    feature_cols.extend([c for c in df.columns if c.startswith('carrier_')])
    
    print(f"Training with {len(feature_cols)} features")
    return df, feature_cols

def train_model():
    df, feature_cols = load_and_preprocess_data()
    
    X = df[feature_cols]
    y = df['is_severe_delay']
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training Random Forest...")
    # KEY CHANGE: class_weight='balanced'
    # This tells the model to pay more attention to the rare "Severe Delay" cases
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=15,          # Limit depth to prevent overfitting
        class_weight='balanced', 
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluation
    print("\n--- Model Evaluation ---")
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
    
    # Sanity Check: Test an 85 minute delay
    print("\n--- Sanity Check: 85 min delay ---")
    # Create a dummy row with 0s
    test_row = np.zeros((1, len(feature_cols)))
    # Set current_delay_minutes (index 0) to 85
    test_row[0, 0] = 85 
    test_row_scaled = scaler.transform(test_row)
    pred_prob = model.predict_proba(test_row_scaled)[0, 1]
    print(f"Predicted Probability for 85 min delay: {pred_prob:.1%}")
    
    # Save Artifacts
    print(f"\nSaving artifacts to {MODEL_DIR}...")
    with open(f'{MODEL_DIR}/cascade_classifier_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open(f'{MODEL_DIR}/cascade_classifier_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(f'{MODEL_DIR}/cascade_classifier_features.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    with open(f'{MODEL_DIR}/cascade_classifier_metadata.pkl', 'wb') as f:
        pickle.dump({'n_total': len(df), 'description': 'Updated model with balanced weights'}, f)
        
    print("Done!")

if __name__ == "__main__":
    train_model()