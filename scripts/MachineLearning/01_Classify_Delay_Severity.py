import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import ast


def load_data():
    """Load all CSV files."""
    print("Loading data...")
    delay_predictions = pd.read_csv('results/delay_predictions.csv')
    flight_assets = pd.read_csv('data/processed/flight_assets_table.csv')
    telemetry = pd.read_csv('data/processed/telemetry_stream.csv')
    airport_mapping = pd.read_csv('data/airport_mapping_complete.csv')
    flight_weather = pd.read_csv('data/processed/flights_weather_table.csv')

    delay_predictions['impacted_events'] = delay_predictions['impacted_events'].apply(ast.literal_eval)
    flight_weather['timestamp'] = pd.to_datetime(flight_weather['timestamp']).dt.tz_localize('UTC')
    delay_predictions['trigger_timestamp'] = pd.to_datetime(delay_predictions['trigger_timestamp'])

    telemetry['location'] = np.where(
        telemetry['event_type'] == 'DEPARTURE', 
        telemetry['origin'], 
        telemetry['destination']
    )

    return delay_predictions, flight_assets, telemetry, airport_mapping, flight_weather


def create_target_variable(delay_predictions, threshold=1):
    """Create binary target variable for cascade events."""
    delay_predictions['is_cascade'] = ((delay_predictions['impacted_events'].apply(len)) >= threshold).astype(int)
    return delay_predictions


def engineer_temporal_features(telemetry):
    """Extract temporal features from telemetry data."""
    telemetry['timestamp'] = pd.to_datetime(telemetry['timestamp'])
    telemetry['scheduled_time'] = pd.to_datetime(telemetry['scheduled_time'])
    telemetry['hour_of_day'] = telemetry['timestamp'].dt.hour
    telemetry['day_of_week'] = telemetry['timestamp'].dt.day_name()
    telemetry['month'] = telemetry['timestamp'].dt.strftime('%b')
    telemetry['is_weekend'] = telemetry['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)
    
    return telemetry


def merge_datasets(telemetry, flight_assets, airport_mapping, flight_weather, delay_predictions):
    """Merge all datasets to create feature-rich training data."""
    print("Engineering features...")

    telemetry = telemetry.drop(columns=['event_id']) # drop arbitrary columns
    flight_assets = flight_assets.drop(columns=['N-NUMBER', 
                                                'MFR MDL CODE', 
                                                'CODE']) #Delete duplicates and arbitrary numbers
    
    airport_mapping = airport_mapping.drop(columns=['DESCRIPTION', 
                                                    'ICAO_CODE', 
                                                    'MATCH_METHOD', 
                                                    'CONFIDENCE']) # Delete arbitrary columns
    
    flight_weather = flight_weather.drop(columns=['avg_weather_delay_min', 
                                                  'max_weather_delay_min', 
                                                  'flights_affected', 
                                                  'flights_with_weather_delay', 
                                                  'pct_flights_delayed', 
                                                  'weather_cancellations']) # only keep the timestamp and weather severity index
    
    delay_predictions = delay_predictions.drop(columns=['impacted_events',
                                                        'event_delays',
                                                        'num_impacted',
                                                        'trigger_event_num',
                                                        'trigger_delay',
                                                        'affected_event_count']) # Drop information that new delays won't have
    # Merge asset features
    df = telemetry.merge(flight_assets, left_on='asset_id', right_on='TAIL_NUM', how='left')
    
    # Merge airport features for origin
    df = df.merge(airport_mapping, left_on='origin', right_on='AIRPORT_ID', how='left', suffixes=('', '_origin'))
    
    # Merge weather data
    df = pd.merge_asof(
        df.sort_values('timestamp'),
        flight_weather[flight_weather['data_source'].notna()].sort_values('timestamp'),
        left_on='timestamp',
        right_on='timestamp',
        by='location',
        direction='nearest',
        tolerance=pd.Timedelta('1h')
    )
    
    # Merge with delay predictions to get target
    df = pd.merge_asof(
        df.sort_values('timestamp'),
        delay_predictions[['trigger_timestamp', 'trigger_carrier', 'trigger_flight_number', 'is_cascade']].sort_values('trigger_timestamp'),
        left_on='timestamp',
        right_on='trigger_timestamp',
        left_by=['carrier', 'flight_number'],
        right_by=['trigger_carrier', 'trigger_flight_number'],
        direction='nearest',
        tolerance=pd.Timedelta('5min')
    )
    
    # Drop rows without target
    df = df.dropna(subset=['is_cascade']).reset_index()
    df = df.drop(columns=['state',
                          'asset_id',
                          'ORIGIN_AIRPORT_ID',
                          'OP_UNIQUE_CARRIER',
                          'date_range',
                          'AIRPORT_ID',
                          'STATUS_REASON',
                          'diversions',
                          'data_source',
                          'trigger_timestamp',
                          'trigger_carrier',
                          'trigger_flight_number',
                          'origin',
                          'destination',
                          'TAIL_NUM'
                          ])
    print(f"Dataset shape: {df.shape}")
    print(f"Cascade events: {df['is_cascade'].sum()} ({100*df['is_cascade'].mean():.1f}%)")
    
    return df

def encode_cyclical_features(df, feature_columns):
    """Encode temporal features to preserve cyclical nature."""
    
    # Day of week (0-6)
    if 'day_of_week' in df.columns:
        # Convert day names to numbers if needed
        day_map = {'MONDAY': 0, 'TUESDAY': 1, 'WEDNESDAY': 2, 'THURSDAY': 3,
                   'FRIDAY': 4, 'SATURDAY': 5, 'SUNDAY': 6}
        if df['day_of_week'].dtype == 'object':
            df['day_of_week_num'] = df['day_of_week'].map(day_map)
        else:
            df['day_of_week_num'] = df['day_of_week']
        
        # Cyclical encoding: converts to sin/cos so Monday is "close" to Sunday
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week_num'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week_num'] / 7)
        feature_columns.extend(['day_of_week_sin', 'day_of_week_cos'])
    
    # Month (1-12)
    if 'month' in df.columns:
        month_map = {'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                     'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12}
        if df['month'].dtype == 'object':
            df['month_num'] = df['month'].map(month_map)
        else:
            df['month_num'] = df['month']
        
        # Cyclical encoding: December is "close" to January
        df['month_sin'] = np.sin(2 * np.pi * df['month_num'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month_num'] / 12)
        feature_columns.extend(['month_sin', 'month_cos'])
    
    # Hour of day (0-23) - already numeric, just encode cyclically
    if 'hour_of_day' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        feature_columns.extend(['hour_sin', 'hour_cos'])
    
    return df, feature_columns

def convert_timedelta_features(df, feature_columns):
    """Convert any timedelta columns to numeric (total seconds or hours)."""
    for col in feature_columns:
        if col in df.columns:
            # Check if column is timedelta or string representation of timedelta
            if pd.api.types.is_timedelta64_dtype(df[col]):
                # Convert to total hours (or seconds if you prefer)
                df[col] = df[col].dt.total_seconds() / 3600  # hours
            elif df[col].dtype == 'object':
                # Try to parse as timedelta if it looks like one
                try:
                    df[col] = pd.to_timedelta(df[col]).dt.total_seconds() / 3600
                except:
                    pass  # Not a timedelta, leave as is
    
    return df

def create_features(df):
    """Create and encode features for modeling."""
    feature_columns = []
    
    df, feature_columns = encode_cyclical_features(df, feature_columns)
    
    # Keep is_weekend as is (already binary)
    if 'is_weekend' in df.columns:
        feature_columns.append('is_weekend')
    
    # Asset features
    if 'avg_flights_per_day' in df.columns:
        feature_columns.append('avg_flights_per_day')
    if 'avg_hours_per_day' in df.columns:
        feature_columns.append('avg_hours_per_day')
    if 'total_flights' in df.columns:
        feature_columns.append('total_flights')
    if 'NO-SEATS' in df.columns:
        feature_columns.append('NO-SEATS')  # aircraft size
    if 'total_flight_time' in df.columns:
        feature_columns.append('total_flight_time')  # cumulative wear
    
    # Weather features
    if 'weather_severity_index' in df.columns:
        df['weather_severity_index'] = df['weather_severity_index'].fillna(0)
        feature_columns.append('weather_severity_index')
    
    if 'CARRIER_DELAY' in df.columns:
        feature_columns.append('CARRIER_DELAY')
    if 'WEATHER_DELAY' in df.columns:
        feature_columns.append('WEATHER_DELAY')
    if 'NAS_DELAY' in df.columns:
        feature_columns.append('NAS_DELAY')
    if 'SECURITY_DELAY' in df.columns:
        feature_columns.append('SECURITY_DELAY')
    if 'LATE_AIRCRAFT_DELAY' in df.columns:
        feature_columns.append('LATE_AIRCRAFT_DELAY')

    # Categorical features - one-hot encode
    if 'carrier' in df.columns:
        carrier_dummies = pd.get_dummies(df['carrier'], prefix='carrier', drop_first=True)
        df = pd.concat([df, carrier_dummies], axis=1)
        feature_columns.extend(carrier_dummies.columns.tolist())
    
    if 'location' in df.columns:
        # Use only top airports to avoid sparse features
        top_airports = df['location'].value_counts().head(200).index
        df['location_grouped'] = df['location'].apply(lambda x: x if x in top_airports else 'OTHER')
        airport_dummies = pd.get_dummies(df['location_grouped'], prefix='airport', drop_first=True)
        df = pd.concat([df, airport_dummies], axis=1)
        feature_columns.extend(airport_dummies.columns.tolist())
    
    if 'event_type' in df.columns:
        event_dummies = pd.get_dummies(df['event_type'], prefix='event', drop_first=True)
        df = pd.concat([df, event_dummies], axis=1)
        feature_columns.extend(event_dummies.columns.tolist())

    df = convert_timedelta_features(df, feature_columns)

    print(f"\nFeature count: {len(feature_columns)}")
    
    return df, feature_columns

def clean_data(X):
    """Clean data by handling inf and NaN values."""
    print("\n=== Cleaning Data ===")
    
    # Check for inf values
    inf_mask = np.isinf(X)
    if inf_mask.any().any():
        print(f"Found {inf_mask.sum().sum()} infinity values")
        inf_cols = X.columns[inf_mask.any()].tolist()
        print(f"Columns with inf: {inf_cols}")
        
        # Replace inf with NaN, then fill
        X = X.replace([np.inf, -np.inf], np.nan)
    
    # Check for NaN values
    nan_counts = X.isna().sum()
    if nan_counts.any():
        print(f"\nColumns with NaN values:")
        for col in nan_counts[nan_counts > 0].index:
            print(f"  {col}: {nan_counts[col]} ({100*nan_counts[col]/len(X):.1f}%)")
        
        # Fill NaN with 0 (or median for some columns)
        X = X.fillna(0)
    
    # Check for extremely large values
    for col in X.columns:
        max_val = X[col].max()
        if abs(max_val) > 1e10:
            print(f"\nWARNING: Column '{col}' has very large values (max={max_val:.2e})")
            # Cap extreme values at 99th percentile
            cap_value = X[col].quantile(0.99)
            X[col] = X[col].clip(upper=cap_value)
            print(f"  Capped at 99th percentile: {cap_value:.2f}")
    
    print(f"\nFinal data shape: {X.shape}")
    print("Data cleaning complete.")
    
    return X

def prepare_train_val_test_split(X, y):
    """Split data into train, validation, and test sets."""
    # Split data: 70% train, 15% validation, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp)
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def train_model(X_train_scaled, y_train):
    """Train logistic regression model."""
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        C=1.0  # Regularization strength
    )
    model.fit(X_train_scaled, y_train)
    
    return model


def evaluate_model(model, X_scaled, y_true, dataset_name):
    """Evaluate model performance on a dataset."""
    print(f"\n=== {dataset_name.upper()} PERFORMANCE ===")
    
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]
    
    print(classification_report(y_true, y_pred, target_names=['No Cascade', 'Cascade']))
    roc_auc = roc_auc_score(y_true, y_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    return y_pred, y_proba, roc_auc


def plot_confusion_matrix(y_true, y_pred, filename='results/MachineLearning/confusion_matrix.png'):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nConfusion matrix saved to '{filename}'")


def plot_roc_curve(y_true, y_proba, filename='results/MachineLearning/roc_curve.png'):
    """Plot and save ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Cascade Likelihood Classifier')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"ROC curve saved to '{filename}'")


def analyze_feature_importance(model, feature_columns, filename='results/MachineLearning/feature_importance.png'):
    """Analyze and plot feature importance."""
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print("\n=== TOP 15 MOST IMPORTANT FEATURES ===")
    print(feature_importance.head(15).to_string(index=False))
    
    plt.figure(figsize=(10, 8))
    top_features = feature_importance.head(20)
    plt.barh(range(len(top_features)), top_features['coefficient'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Coefficient Value')
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nFeature importance plot saved to '{filename}'")


def save_model_artifacts(model, scaler, feature_columns, model_metadata):
    """Save trained model, scaler, feature list, and metadata."""
    with open('cascade_classifier_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('cascade_classifier_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('cascade_classifier_features.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    with open('cascade_classifier_metadata.pkl', 'wb') as f:
        pickle.dump(model_metadata, f)
    
    print("\nModel saved to 'cascade_classifier_model.pkl'")
    print("Scaler saved to 'cascade_classifier_scaler.pkl'")
    print("Features saved to 'cascade_classifier_features.pkl'")
    print("Metadata saved to 'cascade_classifier_metadata.pkl'")


def main():
    """Main execution function."""
    # Load data
    delay_predictions, flight_assets, telemetry, airport_mapping, flight_weather = load_data()
    
    # Create target variable
    delay_predictions = create_target_variable(delay_predictions, threshold=1)
    
    # Engineer temporal features
    telemetry = engineer_temporal_features(telemetry)
    
    # Merge datasets
    df = merge_datasets(telemetry, flight_assets, airport_mapping, flight_weather, delay_predictions)
    
    # Create features
    df, feature_columns = create_features(df)
    
    # Prepare features and target
    X = df[feature_columns].fillna(0)
    y = df['is_cascade']
    print(f"Sample size: {len(X)}")
    
    #Clean Data
    X = clean_data(X)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_train_val_test_split(X, y)
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)
    
    # Train model
    model = train_model(X_train_scaled, y_train)
    
    # Evaluate on validation set
    evaluate_model(model, X_val_scaled, y_val, 'validation')
    
    # Evaluate on test set
    y_test_pred, y_test_proba, roc_auc = evaluate_model(model, X_test_scaled, y_test, 'test')
    
    # Plot results
    plot_confusion_matrix(y_test, y_test_pred)
    plot_roc_curve(y_test, y_test_proba)
    analyze_feature_importance(model, feature_columns)
    
    model_metadata = {
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'n_total': len(X_train) + len(X_val) + len(X_test),
        'cascade_pct_train': float(y_train.sum() / len(y_train) * 100),
        'cascade_pct_val': float(y_val.sum() / len(y_val) * 100),
        'cascade_pct_test': float(y_test.sum() / len(y_test) * 100),
        'n_features': len(feature_columns),
        'training_date': datetime.now().isoformat(),
        'threshold': 1  # Your cascade threshold
    }
    
    # Save model artifacts with metadata
    save_model_artifacts(model, scaler, feature_columns, model_metadata)
    
    print("\nTraining complete!")

    test_results = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_test_pred,
        'y_proba': y_test_proba
    })
    test_results.to_csv('results/MachineLearning/test_results.csv', index=False)

if __name__ == "__main__":
    main()