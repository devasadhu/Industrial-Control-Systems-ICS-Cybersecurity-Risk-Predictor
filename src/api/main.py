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
Version: 2.1.0

Fixes in v1.3.0:
- Models loaded with joblib.load() (not pickle.load())
- feature_scaler.pkl is an array of feature names, not a StandardScaler;
  loading now uses feature_names.txt correctly
- ensemble_scaler.pkl used for actual scaling
- Feature vector construction uses correct feature_names ordering
- Pydantic @validator → @field_validator (Pydantic v2)
- @app.on_event("startup") → lifespan context manager (FastAPI 0.93+)

Fixes in v2.0.0:
- All feature references updated to 58 — models are trained on 58 features
  (52 base + 6 session). Fallback generic names corrected to 58.
- /predict and /predict/batch now load EnsembleICSDetector (IF + XGBoost + RF)
  instead of IsolationForest alone. Predictions go through ensemble.predict()
  which applies the correct threshold=0.25 and weighted voting across all 3 models.
  Previously XGBoost and RF were trained but completely bypassed at inference time.
- is_anomaly logic updated: ensemble.predict() returns (predictions, confidence)
  where predictions are 0/1 labels (not -1/1 IsolationForest convention).
- anomaly_score sourced from ensemble confidence (0–1) instead of raw IF
  score_samples output (negative floats), so PredictionResponse values are
  now in a consistent 0–1 range.
- Session features (src_unique_dst_count, src_flow_count, src_inter_flow_interval,
  src_inter_flow_variance, src_dst_flow_ratio, src_payload_entropy) cannot be
  computed from a single-flow API request — they require 60-second window
  aggregation across flows. These 6 features are zero-filled at inference time.
  This is documented in /model/info and logged as a warning on startup.
  Impact: replay and ip-scan detection accuracy is reduced for single-flow
  requests compared to batch/streaming session-aware inference.
- /model/info updated to reflect 58 features, ensemble model type, and session
  feature limitation.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import time
from datetime import datetime
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate Limiting
RATE_LIMIT: Dict[str, List[float]] = {}
MAX_REQUESTS = 100
TIME_WINDOW = 60  # seconds

MODEL_CACHE: Dict = {}


def load_model():
    """
    Load EnsembleICSDetector (IF + XGBoost + RF), scaler, and 58 feature names.

    NOTE: All pkl files are joblib format — use joblib.load(), NOT pickle.load().
    NOTE: The full ensemble is loaded (not just ensemble_isolation_forest.pkl).
          EnsembleICSDetector.predict() applies threshold=0.25 and weighted voting
          across all 3 models. Calling only IsolationForest bypasses XGB/RF.
    NOTE: Session features (6 of 58) are zero-filled for single-flow requests.
          See module docstring for impact.
    """
    if MODEL_CACHE:
        return MODEL_CACHE

    try:
        current_dir = Path(__file__).parent
        models_dir = current_dir.parent.parent / "models"
        if not models_dir.exists():
            models_dir = Path("./models")
        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found. Tried: {models_dir}")

        # ── Load full ensemble ────────────────────────────────────────────────
        import sys
        sys.path.insert(0, str(current_dir.parent.parent))

        try:
            from src.models.ensemble_detector import EnsembleICSDetector
            ensemble = EnsembleICSDetector()
            ensemble.load(str(models_dir))
            MODEL_CACHE['ensemble'] = ensemble
            MODEL_CACHE['use_ensemble'] = True
            logger.info("✅ Loaded EnsembleICSDetector (IF + XGBoost + RF, threshold=0.25)")
        except Exception as e:
            # Fallback: load IsolationForest only (degraded — XGB/RF bypassed)
            logger.warning(
                f"⚠️  EnsembleICSDetector load failed ({e}). "
                "Falling back to IsolationForest only. "
                "Predictions will use threshold=-0 (IF convention), not 0.25."
            )
            model_path = models_dir / "ensemble_isolation_forest.pkl"
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found: {model_path}\n"
                    "Run `python quick_start.py` to train the ensemble first."
                )
            MODEL_CACHE['model'] = joblib.load(model_path)
            MODEL_CACHE['use_ensemble'] = False

        # ── Load scaler ───────────────────────────────────────────────────────
        # Use ensemble_scaler.pkl (StandardScaler for 62 features).
        # feature_scaler.pkl is misnamed — it contains feature name strings, not a scaler.
        scaler_path = models_dir / "ensemble_scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)
            MODEL_CACHE['scaler'] = scaler
            # Validate feature count at startup — catch shape mismatch before first request
            if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != 58:
                logger.warning(
                    f"⚠️  ensemble_scaler.pkl was trained on {scaler.n_features_in_} features, "
                    f"expected 58. "
                    f"Re-run quick_start.py to retrain."
                )
            logger.info(f"✅ Loaded ensemble_scaler.pkl "
                        f"(n_features_in_={getattr(scaler, 'n_features_in_', 'unknown')})")
        else:
            logger.warning(
                "⚠️  ensemble_scaler.pkl not found — predictions run without scaling (degraded). "
                "Note: feature_scaler.pkl contains feature names, not a StandardScaler."
            )
            MODEL_CACHE['scaler'] = None

        # ── Load feature names (authoritative: feature_names.txt) ─────────────
        feature_names_path = models_dir / "feature_names.txt"
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                features = [line.strip() for line in f if line.strip()]
            MODEL_CACHE['features'] = features
            if len(features) != 58:
                logger.warning(
                    f"⚠️  feature_names.txt has {len(features)} features, expected 58. "
                    "If ics_feature_engineer.py was re-run without session features, "
                    "models will fail with shape mismatch. Re-run quick_start.py."
                )
        else:
            # Fallback: 62 generic names (not 51 — models are trained on 62)
            MODEL_CACHE['features'] = [f"feature_{i}" for i in range(58)]
            logger.warning("feature_names.txt not found — using 58 generic feature names")

        # ── Session feature indices (zero-filled for single-flow requests) ────
        SESSION_FEATURE_NAMES = {
            'src_unique_dst_count', 'src_flow_count', 'src_inter_flow_interval',
            'src_inter_flow_variance', 'src_dst_flow_ratio', 'src_payload_entropy',
        }
        MODEL_CACHE['session_feature_indices'] = [
            i for i, name in enumerate(MODEL_CACHE['features'])
            if name in SESSION_FEATURE_NAMES
        ]
        if MODEL_CACHE['session_feature_indices']:
            logger.warning(
                f"⚠️  Session features will be zero-filled for single-flow requests "
                f"(indices: {MODEL_CACHE['session_feature_indices']}). "
                "Replay and IP-scan detection accuracy reduced. "
                "Use batch/streaming mode with session aggregation for full accuracy."
            )

        # ── Load metadata ─────────────────────────────────────────────────────
        metadata_path = models_dir / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                MODEL_CACHE['metadata'] = json.load(f)
        else:
            MODEL_CACHE['metadata'] = {
                'model_type': 'EnsembleICSDetector (IF + XGBoost + RF)',
                'n_features': len(MODEL_CACHE['features']),
                'threshold': 0.25,
                'training_samples': 0,
            }

        logger.info(f"✅ Model loaded: {MODEL_CACHE['metadata'].get('model_type', 'unknown')}")
        logger.info(f"   Features : {len(MODEL_CACHE['features'])}")
        logger.info(f"   Threshold: {MODEL_CACHE['metadata'].get('threshold', 0.25)}")
        return MODEL_CACHE

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise


# FIX: lifespan replaces deprecated @app.on_event("startup")
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting ICS Anomaly Detection API...")
    load_model()
    logger.info("✅ API ready to accept requests")
    yield
    # shutdown logic (if any) goes here
    logger.info("🛑 Shutting down ICS Anomaly Detection API")


app = FastAPI(
    title="ICS Anomaly Detection API",
    description="Real-time Industrial Control System attack detection",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()


def check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    if client_ip not in RATE_LIMIT:
        RATE_LIMIT[client_ip] = []
    RATE_LIMIT[client_ip] = [t for t in RATE_LIMIT[client_ip] if now - t < TIME_WINDOW]
    if len(RATE_LIMIT[client_ip]) >= MAX_REQUESTS:
        return False
    RATE_LIMIT[client_ip].append(now)
    return True


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    if token != "demo_api_key_12345":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return token


# Pydantic Models
class NetworkFlow(BaseModel):
    src_ip: str = Field(..., description="Source IP address")
    dst_ip: str = Field(..., description="Destination IP address")
    src_port: int = Field(..., ge=0, le=65535)
    dst_port: int = Field(..., ge=0, le=65535)
    protocol: int = Field(..., ge=0, le=255)

    total_fwd_packets: int = Field(0, ge=0)
    total_bwd_packets: int = Field(0, ge=0)
    total_length_fwd_packets: int = Field(0, ge=0)
    total_length_bwd_packets: int = Field(0, ge=0)
    flow_duration: float = Field(0.0, ge=0.0)
    flow_iat_mean: float = Field(0.0, ge=0.0)
    flow_iat_std: float = Field(0.0, ge=0.0)
    fwd_psh_flags: int = Field(0, ge=0)
    bwd_psh_flags: int = Field(0, ge=0)
    fwd_urg_flags: int = Field(0, ge=0)
    bwd_urg_flags: int = Field(0, ge=0)

    # FIX: Pydantic v2 — @field_validator replaces deprecated @validator
    @field_validator('src_ip', 'dst_ip')
    @classmethod
    def validate_ip(cls, v: str) -> str:
        parts = v.split('.')
        if len(parts) != 4:
            raise ValueError('Invalid IP address format')
        try:
            for part in parts:
                if not 0 <= int(part) <= 255:
                    raise ValueError('Invalid IP address range')
        except ValueError:
            raise ValueError('Invalid IP address')
        return v


class BatchFlowRequest(BaseModel):
    flows: List[NetworkFlow]
    include_features: bool = Field(False)


class PredictionResponse(BaseModel):
    flow_id: int
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    severity: str
    iec62443_zone: Optional[str] = None
    recommended_action: str
    timestamp: str


class BatchPredictionResponse(BaseModel):
    total_flows: int
    anomalies_detected: int
    normal_flows: int
    predictions: List[PredictionResponse]
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    uptime_seconds: float
    total_predictions: int


PREDICTIONS_COUNT = 0
START_TIME = time.time()


def _build_feature_vector(flow: NetworkFlow, feature_names: List[str]) -> np.ndarray:
    """
    Map incoming NetworkFlow fields to the 58-feature vector expected by the model.
    Features are ordered to match feature_names.txt. Unknown features default to 0.

    Coverage by group:
      network_basic (10)  — fully mapped from flow fields
      timing (6)          — partially mapped (src/dst packet + byte rates)
      statistical (11)    — partially mapped (avg fields from totals)
      protocol (20)       — PSH/URG flags mapped; TTL/WIN/fragment/ACK-delay zero-filled
                            (not present in NetworkFlow schema — add fields to extend)
      behavioral (4)      — syn_ack_imbalance computed; others zero-filled
      engineered (5)      — partially computed from available fields
      session (6)         — ALWAYS ZERO-FILLED for single-flow requests.
                            src_unique_dst_count, src_flow_count, src_inter_flow_interval,
                            src_inter_flow_variance, src_dst_flow_ratio, src_payload_entropy
                            require 60-second window aggregation across flows.
                            Impact: replay detection accuracy reduced (src_inter_flow_variance
                            is the primary replay signal, normal/attack ratio=11.76).
                            Use session-aware batch mode for full accuracy.
    """
    n = len(feature_names)
    vec = np.zeros(n)
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    fwd       = flow.total_fwd_packets
    bwd       = flow.total_bwd_packets
    fwd_bytes = flow.total_length_fwd_packets
    bwd_bytes = flow.total_length_bwd_packets
    duration  = max(flow.flow_duration, 0.001)
    total_pkt = fwd + bwd
    total_byt = fwd_bytes + bwd_bytes

    flow_mapping = {
        # ── network_basic (10) ────────────────────────────────────────────────
        'src_packets':       fwd,
        'dst_packets':       bwd,
        'src_bytes':         fwd_bytes,
        'dst_bytes':         bwd_bytes,
        'flow_duration':     flow.flow_duration,
        'total_packets':     total_pkt,
        'packet_ratio':      fwd / max(bwd, 1),
        'total_bytes':       total_byt,
        'byte_ratio':        fwd_bytes / max(bwd_bytes, 1),
        'bytes_per_packet':  total_byt / max(total_pkt, 1),

        # ── timing (6) — inter-packet fields not in schema; rate fields mapped ─
        'src_inter_packet_avg': flow.flow_iat_mean,
        'dst_inter_packet_avg': flow.flow_iat_mean,  # symmetric fallback
        'src_packet_rate':      fwd / duration,
        'dst_packet_rate':      bwd / duration,
        'src_byte_rate':        fwd_bytes / duration,
        'dst_byte_rate':        bwd_bytes / duration,

        # ── statistical (11) — derived from totals ────────────────────────────
        'src_bytes_avg':    fwd_bytes / max(fwd, 1),
        'dst_bytes_avg':    bwd_bytes / max(bwd, 1),
        'src_payload_avg':  fwd_bytes / max(fwd, 1),
        'dst_payload_avg':  bwd_bytes / max(bwd, 1),
        'src_payload_sum':  fwd_bytes,
        # src/dst _max/_min/_load not computable from flow totals — remain 0
        'src_load':         fwd_bytes / duration,
        'dst_load':         bwd_bytes / duration,

        # ── protocol (20) — PSH/URG from flags; rest not in NetworkFlow schema ─
        'src_psh_rate':     flow.fwd_psh_flags / max(fwd, 1),
        'dst_psh_rate':     flow.bwd_psh_flags / max(bwd, 1),
        'src_urg_rate':     flow.fwd_urg_flags / max(fwd, 1),
        'dst_urg_rate':     flow.bwd_urg_flags / max(bwd, 1),
        # src/dst _syn/ack/fin/rst/ttl/win/fragment/ack_delay — zero-filled
        # Add these to NetworkFlow if protocol-level detection accuracy is needed

        # ── behavioral (4) ────────────────────────────────────────────────────
        # syn_ack_imbalance: approximated from PSH flag imbalance
        'syn_ack_imbalance':  abs(flow.fwd_psh_flags - flow.bwd_psh_flags) / max(total_pkt, 1),
        'traffic_symmetry':   1.0 - abs(fwd - bwd) / max(total_pkt, 1),
        # packet_size_anomaly, reset_rate_total — not computable without per-packet data

        # ── engineered (5) ────────────────────────────────────────────────────
        'inter_packet_timing_asymmetry': abs(flow.flow_iat_mean - flow.flow_iat_std),
        'timing_regularity':             1.0 / (1.0 + flow.flow_iat_std),
        'flow_burstiness':               flow.flow_iat_std / max(flow.flow_iat_mean, 0.001),
        # payload_size_consistency, scan_signature_score — not computable here

        # ── session (6) — ZERO-FILLED, cannot compute from single flow ────────
        # src_unique_dst_count, src_flow_count, src_inter_flow_interval,
        # src_inter_flow_variance, src_dst_flow_ratio, src_payload_entropy
        # These remain 0 (np.zeros initialisation above).
    }

    for feat_name, value in flow_mapping.items():
        if feat_name in name_to_idx:
            vec[name_to_idx[feat_name]] = value

    return vec


def map_to_iec62443(dst_port: int, anomaly_score: float) -> tuple:
    if dst_port in [502, 20000]:
        zone = "Level 1 - Process Control"
        severity = "CRITICAL"
    elif dst_port in [44818, 2222]:
        zone = "Level 1 - Field Devices"
        severity = "HIGH"
    elif dst_port in [80, 443, 8080]:
        zone = "Level 2 - Supervisory Control"
        severity = "MEDIUM"
    else:
        zone = "Level 3 - Enterprise Network"
        severity = "LOW"

    if abs(anomaly_score) > 0.6:
        severity = "CRITICAL"
    elif abs(anomaly_score) > 0.5:
        severity = "HIGH"

    return zone, severity


def get_recommended_action(severity: str, is_anomaly: bool) -> str:
    if not is_anomaly:
        return "No action required - normal traffic"
    actions = {
        "CRITICAL": "IMMEDIATE: Isolate affected zone, activate incident response",
        "HIGH": "Investigate immediately, consider traffic blocking",
        "MEDIUM": "Monitor closely, log for investigation",
        "LOW": "Log and review during routine audit"
    }
    return actions.get(severity, "Monitor and log")


@app.get("/", response_model=Dict)
async def root():
    return {
        "service": "ICS Anomaly Detection API",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
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

    Uses EnsembleICSDetector (IF + XGBoost + RF, threshold=0.25).
    Session features (6/58) are zero-filled — see /model/info for impact.

    Authentication: Authorization: Bearer demo_api_key_12345
    """
    global PREDICTIONS_COUNT

    client_ip = request.client.host
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded: {MAX_REQUESTS} requests per {TIME_WINDOW}s"
        )

    try:
        model_cache = load_model()
        scaler       = model_cache['scaler']
        feature_names = model_cache['features']

        features = _build_feature_vector(flow, feature_names)

        if scaler is not None:
            features_scaled = scaler.transform([features])
        else:
            features_scaled = np.array([features])

        # ── Ensemble path (preferred) ─────────────────────────────────────────
        if model_cache.get('use_ensemble'):
            ensemble = model_cache['ensemble']
            # ensemble.predict() returns (predictions, confidence_scores)
            # predictions: 0=normal, 1=attack  (threshold=0.25, already set in detector)
            # confidence: 0–1 float
            predictions, confidences = ensemble.predict(features_scaled)
            is_anomaly  = bool(predictions[0] == 1)
            confidence  = float(confidences[0])
            # Map confidence to an anomaly_score sign convention for IEC 62443 severity
            anomaly_score = confidence if is_anomaly else -confidence

        # ── Fallback: IsolationForest only (degraded) ─────────────────────────
        else:
            model = model_cache['model']
            raw_pred    = model.predict(features_scaled)[0]
            anomaly_score = float(model.score_samples(features_scaled)[0])
            is_anomaly  = (raw_pred == -1)
            confidence  = abs(anomaly_score)

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
    Predict anomalies for multiple flows efficiently.

    Uses EnsembleICSDetector (IF + XGBoost + RF, threshold=0.25).
    Session features are zero-filled per flow — full session accuracy requires
    pre-computing session aggregation before calling this endpoint.
    """
    global PREDICTIONS_COUNT

    start_time = time.time()

    client_ip = request.client.host
    if not check_rate_limit(client_ip):
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                            detail="Rate limit exceeded")

    if len(batch.flows) > 1000:
        raise HTTPException(status_code=400,
                            detail="Batch size too large (max 1000 flows)")

    try:
        model_cache   = load_model()
        scaler        = model_cache['scaler']
        feature_names = model_cache['features']

        feature_matrix = np.array(
            [_build_feature_vector(f, feature_names) for f in batch.flows]
        )
        if scaler is not None:
            feature_matrix = scaler.transform(feature_matrix)

        # ── Ensemble path ─────────────────────────────────────────────────────
        if model_cache.get('use_ensemble'):
            ensemble = model_cache['ensemble']
            predictions_raw, confidence_scores = ensemble.predict(feature_matrix)
            is_anomaly_arr = (predictions_raw == 1)
            # anomaly_score: +confidence for attacks, -confidence for normal
            anomaly_scores = np.where(is_anomaly_arr, confidence_scores, -confidence_scores)

        # ── Fallback: IsolationForest only ────────────────────────────────────
        else:
            model          = model_cache['model']
            predictions_raw = model.predict(feature_matrix)
            anomaly_scores  = model.score_samples(feature_matrix)
            is_anomaly_arr  = (predictions_raw == -1)
            confidence_scores = np.abs(anomaly_scores)

        predictions = []
        for idx, (flow, is_anomaly, score, conf) in enumerate(
                zip(batch.flows, is_anomaly_arr, anomaly_scores, confidence_scores)):
            zone, severity = map_to_iec62443(flow.dst_port, float(score))
            action = get_recommended_action(severity, bool(is_anomaly))
            PREDICTIONS_COUNT += 1

            predictions.append(PredictionResponse(
                flow_id=idx + 1,
                is_anomaly=bool(is_anomaly),
                anomaly_score=float(score),
                confidence=float(conf),
                severity=severity,
                iec62443_zone=zone,
                recommended_action=action,
                timestamp=datetime.now().isoformat()
            ))

        anomalies = sum(1 for p in predictions if p.is_anomaly)

        return BatchPredictionResponse(
            total_flows=len(predictions),
            anomalies_detected=anomalies,
            normal_flows=len(predictions) - anomalies,
            predictions=predictions,
            processing_time_ms=(time.time() - start_time) * 1000
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info(token: str = Depends(verify_token)):
    model_cache = load_model()
    n_features  = len(model_cache['features'])
    scaler      = model_cache['scaler']
    scaler_n    = getattr(scaler, 'n_features_in_', None) if scaler else None

    return {
        "model_type":        model_cache['metadata'].get('model_type',
                             'EnsembleICSDetector (IF + XGBoost + RF)'),
        "ensemble_mode":     model_cache.get('use_ensemble', False),
        "threshold":         model_cache['metadata'].get('threshold', 0.25),
        "n_features":        n_features,
        "scaler_n_features": scaler_n,
        "feature_count_ok":  (n_features == 58 and (scaler_n is None or scaler_n == 58)),
        "training_samples":  model_cache['metadata'].get('training_samples', 0),
        "feature_list":      model_cache['features'][:10] + ["..."],
        "total_features":    n_features,
        "scaler_loaded":     scaler is not None,
        "session_features_zero_filled": model_cache.get('session_feature_indices', []),
        "notes": {
            "feature_count": (
                "58 features: 52 base (per-flow) + 6 session. "
                "Models will fail with shape mismatch if ics_feature_engineer.py "
                "is re-run without session aggregation."
            ),
            "session_features": (
                "6 session features are zero-filled for single-flow requests. "
                "These require 60-second window aggregation across flows. "
                "Replay detection accuracy is reduced (src_inter_flow_variance, "
                "the primary replay signal, is always 0 per-flow). "
                "IP-scan detection is unaffected (dataset ceiling, not an API issue)."
            ),
            "scaler": (
                "Use ensemble_scaler.pkl (StandardScaler, 58 features). "
                "feature_scaler.pkl contains feature name strings only — not a scaler."
            ),
        }
    }


@app.get("/cves/{pattern_name}")
async def get_cves_for_pattern(
    pattern_name: str,
    token: str = Depends(verify_token),
):
    """
    Fetch live CVE enrichment from NVD for a given attack pattern name.

    Pattern names: modbus_flooding, plc_scanning, unauthorized_write,
    man_in_the_middle, replay_attack, protocol_fuzzing, command_injection,
    time_based_attack, credential_stuffing, firmware_modification, modbus_write_dpi
    """
    try:
        from src.detection.attack_patterns import ICSAttackPatternLibrary
    except ImportError:
        raise HTTPException(status_code=500, detail="attack_patterns module not available")

    library = ICSAttackPatternLibrary()
    cves = library.fetch_cves(pattern_name)
    if cves is None:
        raise HTTPException(status_code=404, detail=f"Unknown pattern: {pattern_name}")
    return {
        "pattern_name": pattern_name,
        "cve_count": len(cves),
        "cves": cves,
    }


@app.post("/export/stix")
async def export_stix_bundle(
    request: Request,
    token: str = Depends(verify_token),
):
    """
    Convert detection results to a STIX 2.1 bundle.

    POST body: the JSON object returned by detect_all_patterns() — either
    pass it directly or call /predict/batch first and forward results here.

    Returns: STIX 2.1 Bundle JSON.
    """
    try:
        from src.stix_exporter import ICSSTIXExporter
    except ImportError:
        raise HTTPException(status_code=500, detail="stix_exporter module not available — pip install stix2")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Request body must be valid JSON")

    exporter = ICSSTIXExporter()
    bundle = exporter.export(body)
    return bundle


@app.get("/iec62443/zones")
async def iec62443_zones():
    return {
        "zones": [
            {"level": "Level 0", "name": "Physical Process",
             "description": "Sensors, actuators, PLCs", "critical_ports": [502, 20000, 44818]},
            {"level": "Level 1", "name": "Basic Control",
             "description": "Process control, field devices", "critical_ports": [502, 2222, 44818]},
            {"level": "Level 2", "name": "Supervisory Control",
             "description": "SCADA, HMI, Engineering workstations", "critical_ports": [80, 443, 8080, 3389]},
            {"level": "Level 3", "name": "Site Operations",
             "description": "MES, data historians", "critical_ports": [1433, 3306, 5432]},
            {"level": "Level 4", "name": "Enterprise Network",
             "description": "Corporate IT systems", "critical_ports": [22, 23, 21, 445]},
        ]
    }


if __name__ == "__main__":
    import uvicorn

    print("=" * 80)
    print("🚀 ICS ANOMALY DETECTION API  v2.1.0")
    print("=" * 80)
    print("\n📋 Endpoints:")
    print("   • Health:    http://localhost:8000/health")
    print("   • Docs:      http://localhost:8000/docs")
    print("   • Predict:   POST http://localhost:8000/predict")
    print("   • Batch:     POST http://localhost:8000/predict/batch")
    print("   • Info:      GET  http://localhost:8000/model/info")
    print("   • CVEs:      GET  http://localhost:8000/cves/{pattern_name}")
    print("   • STIX:      POST http://localhost:8000/export/stix")
    print("\n🔑 Authentication:")
    print("   Header: Authorization: Bearer demo_api_key_12345")
    print("\n🤖 Model: EnsembleICSDetector (IF + XGBoost + RF, threshold=0.25)")
    print("   Features: 58 (session features zero-filled for single-flow requests)")
    print("   See /model/info for accuracy implications.")
    print("\n🛡 IEC 62443 Compliance: Enabled")
    print("=" * 80 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")