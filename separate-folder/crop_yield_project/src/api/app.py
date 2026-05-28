"""
FastAPI Production Server
Endpoints: /predict, /predict/batch, /health, /metadata, /crops, /districts
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
import joblib, json, os, time, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Load artifacts ────────────────────────────────────────────────────────────
BASE   = os.path.join(os.path.dirname(__file__), "../../")
MODEL_DIR = os.path.join(BASE, "models")

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

model    = tf.keras.models.load_model(f"{MODEL_DIR}/best_model.keras")
scaler   = joblib.load(f"{MODEL_DIR}/feature_scaler.pkl")
encoders = joblib.load(f"{MODEL_DIR}/label_encoders.pkl")

with open(f"{MODEL_DIR}/metadata.json") as f:
    META = json.load(f)

FEATURE_COLS = META["feature_cols"]

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title        = "🌾 Crop Yield Prediction API",
    description  = "AI-powered crop yield prediction for Indian agriculture",
    version      = "1.0.0",
    docs_url     = "/docs",
    redoc_url    = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Request / Response models ─────────────────────────────────────────────────

class PredictRequest(BaseModel):
    state:              str   = Field(...,  example="Gujarat")
    district:           str   = Field(...,  example="Anand")
    crop:               str   = Field(...,  example="Wheat")
    season:             str   = Field(...,  example="Rabi")
    year:               int   = Field(...,  example=2024)
    annual_rainfall_mm: float = Field(...,  example=650.0)
    tmax_celsius:       float = Field(...,  example=34.0)
    tmin_celsius:       float = Field(...,  example=15.0)
    humidity_pct:       float = Field(...,  example=55.0)
    solar_radiation:    float = Field(18.0, example=18.0)
    soil_type:          str   = Field(...,  example="Loam")
    soil_ph:            float = Field(6.5,  example=6.5)
    soil_n_kgha:        float = Field(280.0,example=280.0)
    soil_p_kgha:        float = Field(30.0, example=30.0)
    soil_k_kgha:        float = Field(200.0,example=200.0)
    organic_carbon_pct: float = Field(0.8,  example=0.8)
    elevation_m:        int   = Field(150,  example=150)
    fertilizer_n_kgha:  float = Field(120.0,example=120.0)
    fertilizer_p_kgha:  float = Field(50.0, example=50.0)
    fertilizer_k_kgha:  float = Field(50.0, example=50.0)
    pesticide_kgha:     float = Field(1.5,  example=1.5)
    irrigation_type:    str   = Field(...,  example="Canal")
    area_ha:            float = Field(500.0,example=500.0)
    ndvi:               float = Field(0.6,  example=0.6)


class ShapValue(BaseModel):
    feature:     str
    value:       float
    contribution:float
    direction:   str   # "positive" | "negative"

class PredictResponse(BaseModel):
    predicted_yield_kg_ha:   float
    confidence_interval_low: float
    confidence_interval_high:float
    yield_category:          str   # Low / Medium / High / Very High
    shap_top5:               List[ShapValue]
    advisory:                str
    model_version:           str
    latency_ms:              float


# ── Helper functions ──────────────────────────────────────────────────────────

def safe_encode(enc_name: str, value: str) -> int:
    le = encoders[enc_name]
    if value not in le.classes_:
        # Return closest known class index 0 with a warning flag
        return 0
    return int(le.transform([value])[0])


def build_feature_vector(req: PredictRequest) -> np.ndarray:
    rain_dev    = 0.0   # simplified; real pipeline would compute rolling mean
    npk_ratio   = req.fertilizer_n_kgha / (req.fertilizer_p_kgha + req.fertilizer_k_kgha + 1)
    total_npk   = req.fertilizer_n_kgha + req.fertilizer_p_kgha + req.fertilizer_k_kgha
    temp_range  = req.tmax_celsius - req.tmin_celsius
    mean_temp   = (req.tmax_celsius + req.tmin_celsius) / 2
    heat_stress = 1 if req.tmax_celsius > 35 else 0
    drought_flg = 1 if req.annual_rainfall_mm < 400 else 0
    is_irrigated= 0 if req.irrigation_type == "Rainfed" else 1
    soil_fert   = (req.soil_n_kgha/450 + req.soil_p_kgha/60 +
                   req.soil_k_kgha/350 + req.organic_carbon_pct/1.8) / 4
    year_trend  = (req.year - 1992) / 32
    rain_ndvi   = req.annual_rainfall_mm * req.ndvi / 1000
    gdd         = mean_temp * 120

    feat = [
        safe_encode("state",          req.state),
        safe_encode("district",       req.district),
        safe_encode("crop",           req.crop),
        safe_encode("season",         req.season),
        safe_encode("soil_type",      req.soil_type),
        safe_encode("irrigation_type",req.irrigation_type),
        req.annual_rainfall_mm,
        req.tmax_celsius,
        req.tmin_celsius,
        req.humidity_pct,
        req.solar_radiation,
        gdd,
        rain_dev,
        heat_stress,
        drought_flg,
        req.soil_ph,
        req.soil_n_kgha,
        req.soil_p_kgha,
        req.soil_k_kgha,
        req.organic_carbon_pct,
        req.elevation_m,
        soil_fert,
        req.fertilizer_n_kgha,
        req.fertilizer_p_kgha,
        req.fertilizer_k_kgha,
        req.pesticide_kgha,
        req.area_ha,
        is_irrigated,
        npk_ratio,
        total_npk,
        temp_range,
        mean_temp,
        year_trend,
        rain_ndvi,
        req.ndvi,
    ]
    return np.array(feat, dtype=np.float32).reshape(1, -1)


def compute_simple_shap(feat_raw: np.ndarray) -> List[ShapValue]:
    """Simple gradient-based feature importance (SHAP approximation)."""
    import tensorflow as tf as _tf
    feat_tensor = _tf.Variable(feat_raw, dtype=tf.float32)
    feat_scaled = scaler.transform(feat_raw)
    feat_t      = tf.Variable(feat_scaled.astype(np.float32))

    with tf.GradientTape() as tape:
        pred = model(feat_t, training=False)
    grads = tape.gradient(pred, feat_t).numpy().flatten()

    importances = np.abs(grads * feat_scaled.flatten())
    top5_idx    = np.argsort(importances)[::-1][:5]

    results = []
    for i in top5_idx:
        raw_grad = float(grads[i] * feat_scaled.flatten()[i])
        results.append(ShapValue(
            feature      = FEATURE_COLS[i] if i < len(FEATURE_COLS) else f"feat_{i}",
            value        = float(feat_raw.flatten()[i]),
            contribution = float(abs(raw_grad)),
            direction    = "positive" if raw_grad > 0 else "negative",
        ))
    return results


def yield_category(y_kg: float, crop: str) -> str:
    thresholds = {
        "Rice":2500,"Wheat":3000,"Maize":2200,"Sugarcane":50000,
        "Potato":15000,"Tomato":18000,"Banana":25000,
    }
    base = thresholds.get(crop, 2000)
    if y_kg < base * 0.6:  return "Low"
    if y_kg < base * 0.9:  return "Medium"
    if y_kg < base * 1.2:  return "High"
    return "Very High"


def generate_advisory(req: PredictRequest, yield_kg: float) -> str:
    msgs = []
    if req.annual_rainfall_mm < 400:
        msgs.append("⚠️ Low rainfall — consider supplemental irrigation")
    if req.tmax_celsius > 36:
        msgs.append("🌡️ Heat stress risk — use heat-tolerant varieties")
    if req.fertilizer_n_kgha < 60:
        msgs.append("🌱 Low nitrogen — increase urea application by ~30 kg/ha")
    if req.ndvi < 0.4:
        msgs.append("📡 Low NDVI — monitor crop health and check for disease")
    if req.irrigation_type == "Rainfed" and req.annual_rainfall_mm < 600:
        msgs.append("💧 Rainfed + low rainfall — consider drip irrigation to improve yield by 15-20%")
    if not msgs:
        msgs.append("✅ Conditions look good — maintain current crop management practices")
    return " | ".join(msgs)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model": "ResidualANN", "version": "1.0.0"}


@app.get("/metadata")
def metadata():
    return {
        "crops":      META["crops"],
        "states":     META["states"],
        "districts":  META["districts"],
        "seasons":    META["seasons"],
        "soil_types": META["soil_types"],
        "irrigation": META["irrigation"],
        "n_features": META["n_features"],
    }


@app.get("/crops")
def list_crops():
    return {"crops": META["crops"]}


@app.get("/districts")
def list_districts():
    return {"districts": META["districts"]}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t0 = time.time()
    try:
        feat_raw  = build_feature_vector(req)
        feat_sc   = scaler.transform(feat_raw).astype(np.float32)
        log_pred  = float(model.predict(feat_sc, verbose=0)[0][0])
        yield_kg  = float(np.expm1(log_pred))
        yield_kg  = max(0, yield_kg)

        # Confidence interval: ±15% (proxy; real CI needs MC dropout)
        ci_low  = round(yield_kg * 0.85, 1)
        ci_high = round(yield_kg * 1.15, 1)

        # Simple feature importances
        try:
            shap5 = compute_simple_shap(feat_raw)
        except Exception:
            shap5 = []

        return PredictResponse(
            predicted_yield_kg_ha   = round(yield_kg, 1),
            confidence_interval_low = ci_low,
            confidence_interval_high= ci_high,
            yield_category          = yield_category(yield_kg, req.crop),
            shap_top5               = shap5,
            advisory                = generate_advisory(req, yield_kg),
            model_version           = "ResidualANN-v1.0",
            latency_ms              = round((time.time() - t0) * 1000, 1),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(requests: List[PredictRequest]):
    results = []
    for req in requests[:50]:   # cap at 50 per batch
        try:
            feat_raw = build_feature_vector(req)
            feat_sc  = scaler.transform(feat_raw).astype(np.float32)
            log_pred = float(model.predict(feat_sc, verbose=0)[0][0])
            yield_kg = max(0, float(np.expm1(log_pred)))
            results.append({
                "crop": req.crop, "district": req.district,
                "season": req.season, "year": req.year,
                "predicted_yield_kg_ha": round(yield_kg, 1),
                "yield_category": yield_category(yield_kg, req.crop),
            })
        except Exception as e:
            results.append({"error": str(e)})
    return {"results": results, "count": len(results)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
