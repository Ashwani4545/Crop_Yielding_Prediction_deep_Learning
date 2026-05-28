"""
Preprocessing Pipeline
- Cleans raw data
- Engineers 15+ features
- Encodes categoricals
- Splits train/val/test
- Saves processed arrays + encoders
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib, os, json

RAW_PATH  = "data/raw/crop_yield_india.csv"
PROC_DIR  = "data/processed"
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(RAW_PATH)
print(f"Loaded {len(df):,} rows")

# ── 2. Drop exact duplicates ──────────────────────────────────────────────────
df.drop_duplicates(inplace=True)

# ── 3. Handle missing values ──────────────────────────────────────────────────
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# ── 4. Feature Engineering ────────────────────────────────────────────────────

# 4a. Rainfall deviation from district 10-yr mean
df["rain_dev_pct"] = df.groupby(["district", "crop"])["annual_rainfall_mm"].transform(
    lambda x: (x - x.rolling(10, min_periods=3).mean()) / (x.rolling(10, min_periods=3).mean() + 1) * 100
).fillna(0).round(2)

# 4b. NPK ratio
df["npk_ratio"]    = (df["fertilizer_n_kgha"] /
                      (df["fertilizer_p_kgha"] + df["fertilizer_k_kgha"] + 1)).round(3)
df["total_npk"]    = (df["fertilizer_n_kgha"] +
                      df["fertilizer_p_kgha"] +
                      df["fertilizer_k_kgha"]).round(1)

# 4c. Temperature range
df["temp_range"]   = (df["tmax_celsius"] - df["tmin_celsius"]).round(1)
df["mean_temp"]    = ((df["tmax_celsius"] + df["tmin_celsius"]) / 2).round(2)

# 4d. Heat stress flag (days > 35°C proxy)
df["heat_stress"]  = (df["tmax_celsius"] > 35).astype(int)

# 4e. Drought stress flag
df["drought_flag"] = (df["annual_rainfall_mm"] < 400).astype(int)

# 4f. Irrigation binary (irrigated vs rainfed)
df["is_irrigated"] = (df["irrigation_type"] != "Rainfed").astype(int)

# 4g. Soil fertility index (normalised sum)
df["soil_fertility"] = (
    (df["soil_n_kgha"] / 450 + df["soil_p_kgha"] / 60 +
     df["soil_k_kgha"] / 350 + df["organic_carbon_pct"] / 1.8) / 4
).round(3)

# 4h. Year trend (scaled)
df["year_trend"]   = ((df["year"] - 1992) / 32).round(4)

# 4i. Rainfall × NDVI interaction (proxy for canopy + water availability)
df["rain_ndvi"]    = (df["annual_rainfall_mm"] * df["ndvi"] / 1000).round(3)

# 4j. Log transform skewed target
df["log_yield"]    = np.log1p(df["yield_kg_ha"]).round(4)

print("Feature engineering done. Shape:", df.shape)

# ── 5. Encode categoricals ────────────────────────────────────────────────────
cat_cols    = ["state", "district", "crop", "season", "soil_type", "irrigation_type"]
encoders    = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col])
    encoders[col]    = le
    print(f"  {col}: {len(le.classes_)} classes")

joblib.dump(encoders, "models/label_encoders.pkl")

# ── 6. Define feature matrix ──────────────────────────────────────────────────
FEATURE_COLS = [
    # Encoded categoricals
    "state_enc", "district_enc", "crop_enc", "season_enc",
    "soil_type_enc", "irrigation_type_enc",
    # Weather
    "annual_rainfall_mm", "tmax_celsius", "tmin_celsius",
    "humidity_pct", "solar_radiation", "gdd",
    "rain_dev_pct", "heat_stress", "drought_flag",
    # Soil
    "soil_ph", "soil_n_kgha", "soil_p_kgha", "soil_k_kgha",
    "organic_carbon_pct", "elevation_m", "soil_fertility",
    # Inputs
    "fertilizer_n_kgha", "fertilizer_p_kgha", "fertilizer_k_kgha",
    "pesticide_kgha", "area_ha", "is_irrigated",
    "npk_ratio", "total_npk",
    # Derived
    "temp_range", "mean_temp", "year_trend", "rain_ndvi",
    # Remote sensing
    "ndvi",
]

TARGET_COL = "log_yield"   # predict log(yield), inverse at serving time

X = df[FEATURE_COLS].values.astype(np.float32)
y = df[TARGET_COL].values.astype(np.float32)

# ── 7. Train / Val / Test split (70/15/15) ────────────────────────────────────
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.176, random_state=42)
# 0.176 of 0.85 ≈ 0.15 of total

print(f"\nSplits → Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

# ── 8. Scale features ─────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)
joblib.dump(scaler, "models/feature_scaler.pkl")

# ── 9. Save processed arrays ──────────────────────────────────────────────────
np.save(f"{PROC_DIR}/X_train.npy", X_train_s)
np.save(f"{PROC_DIR}/X_val.npy",   X_val_s)
np.save(f"{PROC_DIR}/X_test.npy",  X_test_s)
np.save(f"{PROC_DIR}/y_train.npy", y_train)
np.save(f"{PROC_DIR}/y_val.npy",   y_val)
np.save(f"{PROC_DIR}/y_test.npy",  y_test)

# Save processed CSV for notebook EDA
df.to_csv(f"{PROC_DIR}/crop_yield_processed.csv", index=False)

# Save feature metadata
meta = {
    "feature_cols":  FEATURE_COLS,
    "target_col":    TARGET_COL,
    "n_features":    len(FEATURE_COLS),
    "n_train":       int(len(X_train)),
    "n_val":         int(len(X_val)),
    "n_test":        int(len(X_test)),
    "crops":         list(encoders["crop"].classes_),
    "states":        list(encoders["state"].classes_),
    "districts":     list(encoders["district"].classes_),
    "seasons":       list(encoders["season"].classes_),
    "soil_types":    list(encoders["soil_type"].classes_),
    "irrigation":    list(encoders["irrigation_type"].classes_),
}
with open("models/metadata.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n✅ Preprocessing complete. Features: {len(FEATURE_COLS)}")
print("Saved: data/processed/, models/label_encoders.pkl, models/feature_scaler.pkl, models/metadata.json")
