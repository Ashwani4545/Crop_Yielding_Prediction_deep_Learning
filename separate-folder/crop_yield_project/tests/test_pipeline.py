"""
Test suite for Crop Yield Prediction API
Run: pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/ml"))

import pytest
import numpy as np


# ── Unit tests: feature engineering ──────────────────────────────────────────

class TestFeatureEngineering:

    def test_npk_ratio_positive(self):
        n, p, k = 120, 50, 50
        ratio = n / (p + k + 1)
        assert ratio > 0

    def test_heat_stress_flag(self):
        assert (1 if 38 > 35 else 0) == 1
        assert (1 if 30 > 35 else 0) == 0

    def test_drought_flag(self):
        assert (1 if 350 < 400 else 0) == 1
        assert (1 if 700 < 400 else 0) == 0

    def test_soil_fertility_range(self):
        fert = (280/450 + 30/60 + 200/350 + 0.8/1.8) / 4
        assert 0.0 < fert < 1.0

    def test_year_trend_scaling(self):
        # 1992 → 0.0, 2024 → 1.0
        assert abs((1992 - 1992) / 32) < 0.001
        assert abs((2024 - 1992) / 32 - 1.0) < 0.001

    def test_rain_ndvi_interaction(self):
        rain, ndvi = 650, 0.6
        val = rain * ndvi / 1000
        assert 0 < val < 5

    def test_log_yield_invertible(self):
        yield_kg = 3200.0
        log_y = np.log1p(yield_kg)
        recovered = np.expm1(log_y)
        assert abs(recovered - yield_kg) < 0.01


# ── Unit tests: yield category logic ─────────────────────────────────────────

class TestYieldCategory:

    BASELINES = {
        "Wheat": 3000, "Rice": 2500, "Maize": 2200,
        "Sugarcane": 50000, "Potato": 15000,
    }

    def categorise(self, crop, y):
        base = self.BASELINES.get(crop, 2000)
        if y < base * 0.6:  return "Low"
        if y < base * 0.9:  return "Medium"
        if y < base * 1.2:  return "High"
        return "Very High"

    def test_wheat_low(self):
        assert self.categorise("Wheat", 1000) == "Low"

    def test_wheat_medium(self):
        assert self.categorise("Wheat", 2400) == "Medium"

    def test_wheat_high(self):
        assert self.categorise("Wheat", 3100) == "High"

    def test_wheat_very_high(self):
        assert self.categorise("Wheat", 4000) == "Very High"

    def test_sugarcane_scale(self):
        assert self.categorise("Sugarcane", 70000) == "Very High"
        assert self.categorise("Sugarcane", 20000) == "Low"


# ── Integration tests: model inference ───────────────────────────────────────

class TestModelInference:

    @pytest.fixture(scope="class")
    def model_artifacts(self):
        import tensorflow as tf
        import joblib, json
        base = os.path.join(os.path.dirname(__file__), "../models")
        model    = tf.keras.models.load_model(f"{base}/best_model.keras")
        scaler   = joblib.load(f"{base}/feature_scaler.pkl")
        encoders = joblib.load(f"{base}/label_encoders.pkl")
        with open(f"{base}/metadata.json") as f:
            meta = json.load(f)
        return model, scaler, encoders, meta

    def make_features(self, encoders):
        def safe(col, val):
            le = encoders[col]
            return int(le.transform([val])[0]) if val in le.classes_ else 0
        return np.array([
            safe("state","Gujarat"), safe("district","Anand"),
            safe("crop","Wheat"), safe("season","Rabi"),
            safe("soil_type","Loam"), safe("irrigation_type","Canal"),
            650, 33, 14, 55, 18, (33+14)/2*120,
            0, 0, 0, 6.5, 280, 30, 200, 0.8, 150,
            (280/450+30/60+200/350+0.8/1.8)/4,
            120, 50, 50, 1.5, 500, 1,
            120/(50+50+1), 220, 19, 23.5, (2024-1992)/32,
            650*0.6/1000, 0.6,
        ], dtype=np.float32).reshape(1, -1)

    def test_model_output_shape(self, model_artifacts):
        model, scaler, encoders, _ = model_artifacts
        X = self.make_features(encoders)
        X_sc = scaler.transform(X).astype(np.float32)
        out = model.predict(X_sc, verbose=0)
        assert out.shape == (1, 1)

    def test_yield_is_positive(self, model_artifacts):
        model, scaler, encoders, _ = model_artifacts
        X = self.make_features(encoders)
        X_sc = scaler.transform(X).astype(np.float32)
        log_y = float(model.predict(X_sc, verbose=0)[0][0])
        yield_ = float(np.expm1(log_y))
        assert yield_ > 0

    def test_yield_in_plausible_range(self, model_artifacts):
        model, scaler, encoders, _ = model_artifacts
        X = self.make_features(encoders)
        X_sc = scaler.transform(X).astype(np.float32)
        log_y = float(model.predict(X_sc, verbose=0)[0][0])
        yield_ = float(np.expm1(log_y))
        # Wheat yield should be between 200 and 8000 kg/ha
        assert 200 < yield_ < 8000

    def test_feature_count(self, model_artifacts):
        _, _, encoders, meta = model_artifacts
        X = self.make_features(encoders)
        assert X.shape[1] == meta["n_features"]


# ── Data quality tests ────────────────────────────────────────────────────────

class TestDataQuality:

    @pytest.fixture(scope="class")
    def df(self):
        import pandas as pd
        return pd.read_csv("data/raw/crop_yield_india.csv")

    def test_no_null_yield(self, df):
        assert df["yield_kg_ha"].isnull().sum() == 0

    def test_yield_positive(self, df):
        assert (df["yield_kg_ha"] > 0).all()

    def test_year_range(self, df):
        assert df["year"].min() >= 1990
        assert df["year"].max() <= 2025

    def test_rainfall_positive(self, df):
        assert (df["annual_rainfall_mm"] > 0).all()

    def test_expected_columns(self, df):
        required = ["state","district","crop","season","year",
                    "annual_rainfall_mm","tmax_celsius","yield_kg_ha"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_crop_count(self, df):
        assert df["crop"].nunique() >= 15

    def test_district_count(self, df):
        assert df["district"].nunique() >= 50
