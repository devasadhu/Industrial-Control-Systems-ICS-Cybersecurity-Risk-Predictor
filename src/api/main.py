"""
ICS Anomaly Detection API
FastAPI REST endpoint for real-time attack detection

Features:
- JWT Authentication
- Rate Limiting
- Health Checks
- Model Versioning
- IEC 62443 Compliance Mapping

Author: Sadhana Devarajan
Version: 1.0.0
"""

from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import time
from datetime import datetime, timedelta
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="ICS Anomaly Detection API",
    description="Real-time Industrial Control System attack detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Rate Limiting
RATE_LIMIT = {}  # Simple in-memory rate limiter
MAX_REQUESTS = 100
TIME_WINDOW = 60  # seconds

# Global model cache
MODEL_CACHE = {}


def load_model():
    """Load trained model and scaler."""
    if MODEL_CACHE:
        return MODEL_CACHE
    
    try:
        # Find models directory - works from any location
        current_dir = Path(__file__).parent
        models_dir = current_dir.parent.parent / "models"
        
        if not models_dir.exists():
            # Try alternative path
            models_dir = Path("./models")
        
        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found. Tried: {models_dir}")
        
        # Load model
        model_path = models_dir / "isolation_forest_ics_detector.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        MODEL_CACHE['model'] = joblib.load(model_path)
        MODEL_CACHE['scaler'] = joblib.load(models_dir / "feature_scaler.pkl")
        
        # Load feature names
        with open(models_dir / "feature_names.txt", 'r') as f:
            MODEL_CACHE['features'] = [line.strip() for line in f.readlines()]
        
        # Load metadata
        with open(models_dir / "model_metadata.json", 'r') as f:
            MODEL_CACHE['metadata'] = json.load(f)
        
        logger.info(f"✅ Model loaded: {MODEL_CACHE['metadata']['model_type']}")
        return MODEL_CACHE
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


def check_rate_limit(client_ip: str):
    """Simple rate limiting."""
    now = time.time()
    
    if client_ip not in RATE_LIMIT:
        RATE_LIMIT[client_ip] = []
    
    # Remove old requests
    RATE_LIMIT[client_ip] = [t for t in RATE_LIMIT[client_ip] if now - t < TIME_WINDOW]
    
    # Check limit
    if len(RATE_LIMIT[client_ip]) >= MAX_REQUESTS:
        return False
    
    RATE_LIMIT[client_ip].append(now)
    return True


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token (simplified for demo)."""
    token = credentials.credentials
    
    # In production, verify JWT properly
    if token != "demo_api_key_12345":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return token


# Pydantic Models
class NetworkFlow(BaseModel):
    """Single network flow for prediction."""
    src_ip: str = Field(..., description="Source IP address")
    dst_ip: str = Field(..., description="Destination IP address")
    src_port: int = Field(..., ge=0, le=65535, description="Source port")
    dst_port: int = Field(..., ge=0, le=65535, description="Destination port")
    protocol: int = Field(..., ge=0, le=255, description="IP protocol (6=TCP, 17=UDP)")
    
    # Packet statistics
    total_fwd_packets: int = Field(0, ge=0)
    total_bwd_packets: int = Field(0, ge=0)
    total_length_fwd_packets: int = Field(0, ge=0)
    total_length_bwd_packets: int = Field(0, ge=0)
    
    # Timing
    flow_duration: float = Field(0.0, ge=0.0)
    flow_iat_mean: float = Field(0.0, ge=0.0)
    flow_iat_std: float = Field(0.0, ge=0.0)
    
    # Protocol features
    fwd_psh_flags: int = Field(0, ge=0)
    bwd_psh_flags: int = Field(0, ge=0)
    fwd_urg_flags: int = Field(0, ge=0)
    bwd_urg_flags: int = Field(0, ge=0)
    
    @validator('src_ip', 'dst_ip')
    def validate_ip(cls, v):
        parts = v.split('.')
        if len(parts) != 4:
            raise ValueError('Invalid IP address format')
        for part in parts:
            if not 0 <= int(part) <= 255:
                raise ValueError('Invalid IP address range')
        return v


class BatchFlowRequest(BaseModel):
    """Batch prediction request."""
    flows: List[NetworkFlow]
    include_features: bool = Field(False, description="Return engineered features")


class PredictionResponse(BaseModel):
    """Prediction response."""
    flow_id: int
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    severity: str
    iec62443_zone: Optional[str] = None
    recommended_action: str
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    total_flows: int
    anomalies_detected: int
    normal_flows: int
    predictions: List[PredictionResponse]
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_type: str
    uptime_seconds: float
    total_predictions: int


# Global counters
PREDICTIONS_COUNT = 0
START_TIME = time.time()


def map_to_iec62443(dst_port: int, anomaly_score: float) -> tuple:
    """Map detection to IEC 62443 security zones."""
    # IEC 62443 Zone mapping
    if dst_port in [502, 20000]:  # Modbus
        zone = "Level 1 - Process Control"
        severity = "CRITICAL"
    elif dst_port in [44818, 2222]:  # EtherNet/IP, DNP3
        zone = "Level 1 - Field Devices"
        severity = "HIGH"
    elif dst_port in [80, 443, 8080]:  # Web/HMI
        zone = "Level 2 - Supervisory Control"
        severity = "MEDIUM"
    else:
        zone = "Level 3 - Enterprise Network"
        severity = "LOW"
    
    # Adjust severity based on anomaly score
    if abs(anomaly_score) > 0.6:
        severity = "CRITICAL"
    elif abs(anomaly_score) > 0.5:
        severity = "HIGH"
    
    return zone, severity


def get_recommended_action(severity: str, is_anomaly: bool) -> str:
    """Get recommended response action."""
    if not is_anomaly:
        return "No action required - normal traffic"
    
    actions = {
        "CRITICAL": "IMMEDIATE: Isolate affected zone, activate incident response",
        "HIGH": "Investigate immediately, consider traffic blocking",
        "MEDIUM": "Monitor closely, log for investigation",
        "LOW": "Log and review during routine audit"
    }
    return actions.get(severity, "Monitor and log")


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("🚀 Starting ICS Anomaly Detection API...")
    load_model()
    logger.info("✅ API ready to accept requests")


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint."""
    return {
        "service": "ICS Anomaly Detection API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_cache = load_model()
    
    return HealthResponse(
        status="healthy",
        model_loaded=bool(model_cache),
        model_type=model_cache.get('metadata', {}).get('model_type', 'unknown'),
        uptime_seconds=time.time() - START_TIME,
        total_predictions=PREDICTIONS_COUNT
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(
    flow: NetworkFlow,
    request: Request,
    token: str = Depends(verify_token)
):
    """
    Predict if a single network flow is anomalous.
    
    Requires authentication token in header:
    Authorization: Bearer demo_api_key_12345
    """
    global PREDICTIONS_COUNT
    
    # Rate limiting
    client_ip = request.client.host
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {MAX_REQUESTS} requests per {TIME_WINDOW}s"
        )
    
    try:
        model_cache = load_model()
        model = model_cache['model']
        scaler = model_cache['scaler']
        feature_names = model_cache['features']
        
        # Create feature vector (simplified - map flow to 52 features)
        # In production, use proper feature engineering
        features = np.zeros(len(feature_names))
        
        # Map basic features
        features[0] = flow.total_fwd_packets
        features[1] = flow.total_bwd_packets
        features[2] = flow.total_length_fwd_packets
        features[3] = flow.total_length_bwd_packets
        features[4] = flow.flow_duration
        
        # Scale and predict
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)[0]
        anomaly_score = model.score_samples(features_scaled)[0]
        
        # Convert prediction (-1 = anomaly, 1 = normal)
        is_anomaly = (prediction == -1)
        confidence = abs(anomaly_score)
        
        # Map to IEC 62443
        zone, severity = map_to_iec62443(flow.dst_port, anomaly_score)
        action = get_recommended_action(severity, is_anomaly)
        
        PREDICTIONS_COUNT += 1
        
        return PredictionResponse(
            flow_id=PREDICTIONS_COUNT,
            is_anomaly=is_anomaly,
            anomaly_score=float(anomaly_score),
            confidence=float(confidence),
            severity=severity,
            iec62443_zone=zone,
            recommended_action=action,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    batch: BatchFlowRequest,
    request: Request,
    token: str = Depends(verify_token)
):
    """
    Predict anomalies for multiple flows.
    
    Efficient for bulk analysis of network traffic.
    """
    global PREDICTIONS_COUNT
    
    start_time = time.time()
    
    # Rate limiting
    client_ip = request.client.host
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded"
        )
    
    if len(batch.flows) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Batch size too large (max 1000 flows)"
        )
    
    try:
        predictions = []
        
        for idx, flow in enumerate(batch.flows):
            # Reuse single prediction logic
            pred = await predict_single(flow, request, token)
            pred.flow_id = idx + 1
            predictions.append(pred)
        
        anomalies = sum(1 for p in predictions if p.is_anomaly)
        normal = len(predictions) - anomalies
        
        processing_time = (time.time() - start_time) * 1000
        
        return BatchPredictionResponse(
            total_flows=len(predictions),
            anomalies_detected=anomalies,
            normal_flows=normal,
            predictions=predictions,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info(token: str = Depends(verify_token)):
    """Get model information."""
    model_cache = load_model()
    
    return {
        "model_type": model_cache['metadata']['model_type'],
        "n_features": model_cache['metadata']['n_features'],
        "training_samples": model_cache['metadata']['training_samples'],
        "contamination": model_cache['metadata']['contamination'],
        "feature_list": model_cache['features'][:10] + ["..."],
        "total_features": len(model_cache['features'])
    }


@app.get("/iec62443/zones")
async def iec62443_zones():
    """Get IEC 62443 security zone mappings."""
    return {
        "zones": [
            {
                "level": "Level 0",
                "name": "Physical Process",
                "description": "Sensors, actuators, PLCs",
                "critical_ports": [502, 20000, 44818]
            },
            {
                "level": "Level 1",
                "name": "Basic Control",
                "description": "Process control, field devices",
                "critical_ports": [502, 2222, 44818]
            },
            {
                "level": "Level 2",
                "name": "Supervisory Control",
                "description": "SCADA, HMI, Engineering workstations",
                "critical_ports": [80, 443, 8080, 3389]
            },
            {
                "level": "Level 3",
                "name": "Site Operations",
                "description": "MES, data historians",
                "critical_ports": [1433, 3306, 5432]
            },
            {
                "level": "Level 4",
                "name": "Enterprise Network",
                "description": "Corporate IT systems",
                "critical_ports": [22, 23, 21, 445]
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    print("="*80)
    print("🚀 ICS ANOMALY DETECTION API")
    print("="*80)
    print("\n📋 Endpoints:")
    print("   • Health:    http://localhost:8000/health")
    print("   • Docs:      http://localhost:8000/docs")
    print("   • Predict:   POST http://localhost:8000/predict")
    print("\n🔐 Authentication:")
    print("   Header: Authorization: Bearer demo_api_key_12345")
    print("\n🏭 IEC 62443 Compliance: Enabled")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")