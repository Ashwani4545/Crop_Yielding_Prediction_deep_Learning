"""
CLI Prediction Script
Usage:
  python src/ml/predict.py --crop Wheat --state Gujarat --district Anand \
    --season Rabi --year 2024 --rainfall 650 --tmax 33 --tmin 14 \
    --n 120 --p 50 --k 50 --irrigation Canal --soil Loam
"""

import argparse
import numpy as np
import joblib, json, sys, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

BASE     = os.path.join(os.path.dirname(__file__), "../../")
MDL_DIR  = os.path.join(BASE, "models")

model    = tf.keras.models.load_model(f"{MDL_DIR}/best_model.keras")
scaler   = joblib.load(f"{MDL_DIR}/feature_scaler.pkl")
encoders = joblib.load(f"{MDL_DIR}/label_encoders.pkl")
with open(f"{MDL_DIR}/metadata.json") as f:
    META = json.load(f)

FEATURE_COLS = META["feature_cols"]


def safe_enc(col, val):
    le = encoders[col]
    return int(le.transform([val])[0]) if val in le.classes_ else 0


def predict(args):
    rain    = args.rainfall
    tmax    = args.tmax
    tmin    = args.tmin
    n_fert  = args.n
    p_fert  = args.p
    k_fert  = args.k

    feat = [
        safe_enc("state",          args.state),
        safe_enc("district",       args.district),
        safe_enc("crop",           args.crop),
        safe_enc("season",         args.season),
        safe_enc("soil_type",      args.soil),
        safe_enc("irrigation_type",args.irrigation),
        rain, tmax, tmin,
        args.humidity, args.solar, (tmax+tmin)/2*120,
        0.0,                                   # rain_dev_pct
        1 if tmax > 35 else 0,                 # heat_stress
        1 if rain < 400 else 0,                # drought_flag
        args.ph, args.soil_n, args.soil_p, args.soil_k,
        args.oc, args.elevation,
        (args.soil_n/450 + args.soil_p/60 + args.soil_k/350 + args.oc/1.8)/4,
        n_fert, p_fert, k_fert,
        args.pesticide, args.area,
        0 if args.irrigation == "Rainfed" else 1,
        n_fert / (p_fert + k_fert + 1),
        n_fert + p_fert + k_fert,
        tmax - tmin, (tmax + tmin)/2,
        (args.year - 1992) / 32,
        rain * args.ndvi / 1000,
        args.ndvi,
    ]

    X      = np.array(feat, dtype=np.float32).reshape(1, -1)
    X_sc   = scaler.transform(X).astype(np.float32)
    log_y  = float(model.predict(X_sc, verbose=0)[0][0])
    yield_ = max(0, float(np.expm1(log_y)))

    print("\n" + "="*50)
    print("  🌾 CROP YIELD PREDICTION")
    print("="*50)
    print(f"  Crop      : {args.crop}")
    print(f"  Location  : {args.district}, {args.state}")
    print(f"  Season    : {args.season} {args.year}")
    print(f"  Irrigation: {args.irrigation}")
    print("-"*50)
    print(f"  Predicted Yield : {yield_:,.0f} kg/ha")
    print(f"  Confidence Band : {yield_*0.85:,.0f} – {yield_*1.15:,.0f} kg/ha")
    if yield_ >= 1000:
        print(f"  In tonnes/ha    : {yield_/1000:.2f} t/ha")
    print("="*50)

    # Quick advisory
    if tmax > 36:
        print("  ⚠️  Heat stress risk detected (Tmax > 36°C)")
    if rain < 400:
        print("  ⚠️  Low rainfall — consider supplemental irrigation")
    if n_fert < 60:
        print("  🌱 Low nitrogen dose — consider increasing urea")
    print()
    return yield_


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Crop Yield Prediction CLI")
    p.add_argument("--crop",       default="Wheat")
    p.add_argument("--state",      default="Gujarat")
    p.add_argument("--district",   default="Anand")
    p.add_argument("--season",     default="Rabi")
    p.add_argument("--year",       type=int,   default=2024)
    p.add_argument("--rainfall",   type=float, default=650)
    p.add_argument("--tmax",       type=float, default=33)
    p.add_argument("--tmin",       type=float, default=14)
    p.add_argument("--humidity",   type=float, default=55)
    p.add_argument("--solar",      type=float, default=18)
    p.add_argument("--soil",       default="Loam")
    p.add_argument("--ph",         type=float, default=6.5)
    p.add_argument("--soil_n",     type=float, default=280)
    p.add_argument("--soil_p",     type=float, default=30)
    p.add_argument("--soil_k",     type=float, default=200)
    p.add_argument("--oc",         type=float, default=0.8)
    p.add_argument("--elevation",  type=int,   default=150)
    p.add_argument("--n",          type=float, default=120)
    p.add_argument("--p",          type=float, default=50)
    p.add_argument("--k",          type=float, default=50)
    p.add_argument("--pesticide",  type=float, default=1.5)
    p.add_argument("--irrigation", default="Canal")
    p.add_argument("--area",       type=float, default=500)
    p.add_argument("--ndvi",       type=float, default=0.6)

    args = p.parse_args()
    predict(args)
