"""
ML Inference Service
FastAPI service that loads the trained cascade classifier and provides predictions
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cascade Prediction Service")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model artifacts
MODEL = None
SCALER = None
FEATURE_COLUMNS = None
METADATA = None


class FlightUpdate(BaseModel):
    """Flight telemetry update"""
    flight_number: str
    tail_number: str
    carrier: str
    origin: str
    destination: str
    scheduled_time: str
    current_delay_minutes: float
    update_time: str
    status: str
    event_type: str
    delay_reasons: Optional[Dict[str, float]] = None


class PredictionResponse(BaseModel):
    """Cascade prediction response"""
    flight_number: str
    carrier: str
    cascade_probability: float
    risk_level: str  # "Low", "Medium", "High"
    current_delay: float
    confidence: float


def load_model_artifacts(model_dir: str = "."):
    """Load trained model artifacts"""
    global MODEL, SCALER, FEATURE_COLUMNS, METADATA
    
    try:
        with open(f'{model_dir}/cascade_classifier_model.pkl', 'rb') as f:
            MODEL = pickle.load(f)
        with open(f'{model_dir}/cascade_classifier_scaler.pkl', 'rb') as f:
            SCALER = pickle.load(f)
        with open(f'{model_dir}/cascade_classifier_features.pkl', 'rb') as f:
            FEATURE_COLUMNS = pickle.load(f)
        with open(f'{model_dir}/cascade_classifier_metadata.pkl', 'rb') as f:
            METADATA = pickle.load(f)
        
        logger.info(f"Loaded model with {len(FEATURE_COLUMNS)} features")
        logger.info(f"Model trained on {METADATA['n_total']} samples")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False


def engineer_features(flight: FlightUpdate) -> Dict[str, Any]:
    """
    Engineer features from flight update to match training data
    This is a simplified version - in production you'd fetch asset/airport/weather data
    """
    features = {}
    
    # Parse datetime
    scheduled_dt = pd.to_datetime(flight.scheduled_time)
    update_dt = pd.to_datetime(flight.update_time)
    
    # Temporal features
    features['hour_of_day'] = scheduled_dt.hour
    features['day_of_week'] = scheduled_dt.dayofweek
    features['month'] = scheduled_dt.month
    features['is_weekend'] = 1 if scheduled_dt.dayofweek >= 5 else 0
    
    # Cyclical encoding for hour
    features['hour_sin'] = np.sin(2 * np.pi * scheduled_dt.hour / 24)
    features['hour_cos'] = np.cos(2 * np.pi * scheduled_dt.hour / 24)
    
    # Delay features
    features['current_delay_minutes'] = flight.current_delay_minutes
    features['delay_severity'] = 1 if flight.current_delay_minutes > 15 else 0
    
    # Binary delay reason indicators (from delay_reasons dict)
    features['has_carrier_delay'] = 0
    features['has_weather_delay'] = 0
    features['has_nas_delay'] = 0
    features['has_security_delay'] = 0
    features['has_late_aircraft_delay'] = 0
    
    if flight.delay_reasons:
        features['has_carrier_delay'] = 1 if flight.delay_reasons.get('carrier_delay', 0) > 0 else 0
        features['has_weather_delay'] = 1 if flight.delay_reasons.get('weather_delay', 0) > 0 else 0
        features['has_nas_delay'] = 1 if flight.delay_reasons.get('nas_delay', 0) > 0 else 0
        features['has_security_delay'] = 1 if flight.delay_reasons.get('security_delay', 0) > 0 else 0
        features['has_late_aircraft_delay'] = 1 if flight.delay_reasons.get('late_aircraft_delay', 0) > 0 else 0
    
    # One-hot encode carrier (simplified - just set the carrier feature to 1)
    features[f'carrier_{flight.carrier}'] = 1
    
    # One-hot encode airport (simplified)
    features[f'airport_{flight.origin}'] = 1
    
    return features


def align_features_with_training(features: Dict[str, Any]) -> pd.DataFrame:
    """
    Align engineered features with training feature columns
    Fill missing features with 0
    """
    # Create dataframe with all training features initialized to 0
    aligned = pd.DataFrame(0, index=[0], columns=FEATURE_COLUMNS, dtype=float)
    
    # Fill in the features we have
    for feature, value in features.items():
        if feature in FEATURE_COLUMNS:
            aligned.loc[0, feature] = value
    
    return aligned


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model_artifacts("app/models")  # Docker path
    if not success:
        # Try local path
        success = load_model_artifacts(".")
    
    if not success:
        logger.warning("Could not load model artifacts - predictions will fail")


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "features": len(FEATURE_COLUMNS) if FEATURE_COLUMNS else 0
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_cascade(flight: FlightUpdate):
    """
    Predict cascade probability for a delayed flight
    Only call this for flights with delays
    
    Uses rule-based deterministic logic for obvious cascades,
    falls back to ML model for ambiguous cases.
    """
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # RULE 1: Severe delays (>120 min) are almost certain cascades
        if flight.current_delay_minutes >= 120:
            return PredictionResponse(
                flight_number=flight.flight_number,
                carrier=flight.carrier,
                cascade_probability=0.95,
                risk_level="High",
                current_delay=flight.current_delay_minutes,
                confidence=1.0
            )
        
        # RULE 2: Very large delays (>90 min) are very likely cascades
        if flight.current_delay_minutes >= 90:
            return PredictionResponse(
                flight_number=flight.flight_number,
                carrier=flight.carrier,
                cascade_probability=0.85,
                risk_level="High",
                current_delay=flight.current_delay_minutes,
                confidence=0.9
            )
        
        # RULE 3: Large delays (>60 min) are likely cascades
        if flight.current_delay_minutes >= 60:
            # Use ML but boost probability
            features = engineer_features(flight)
            X = align_features_with_training(features)
            X_scaled = SCALER.transform(X)
            ml_proba = MODEL.predict_proba(X_scaled)[0, 1]
            
            # Boost ML prediction for large delays
            boosted_proba = min(0.95, ml_proba * 1.3)
            
            risk_level = "High" if boosted_proba >= 0.6 else "Medium"
            confidence = abs(boosted_proba - 0.5) * 2
            
            return PredictionResponse(
                flight_number=flight.flight_number,
                carrier=flight.carrier,
                cascade_probability=float(boosted_proba),
                risk_level=risk_level,
                current_delay=flight.current_delay_minutes,
                confidence=float(confidence)
            )
        
        # RULE 4: Moderate-severe delays (30-60 min) - use pure ML
        # Engineer features
        features = engineer_features(flight)
        
        # Align with training features
        X = align_features_with_training(features)
        
        # Scale features
        X_scaled = SCALER.transform(X)
        
        # Get prediction
        proba = MODEL.predict_proba(X_scaled)[0, 1]  # Probability of cascade class
        
        # Determine risk level
        if proba < 0.3:
            risk_level = "Low"
        elif proba < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Confidence based on how far from decision boundary
        confidence = abs(proba - 0.5) * 2  # 0 at boundary, 1 at extremes
        
        return PredictionResponse(
            flight_number=flight.flight_number,
            carrier=flight.carrier,
            cascade_probability=float(proba),
            risk_level=risk_level,
            current_delay=flight.current_delay_minutes,
            confidence=float(confidence)
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Cascade Prediction ML Service",
        "version": "1.0.0",
        "model_loaded": MODEL is not None,
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)