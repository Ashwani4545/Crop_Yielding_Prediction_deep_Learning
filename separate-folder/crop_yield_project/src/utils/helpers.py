"""
Shared utility functions — used by both ML pipeline and API server
"""

import numpy as np
import json, os

# ── Crop baseline yields (kg/ha) for advisory thresholds ─────────────────────
CROP_BASELINES = {
    "Rice": 2800, "Wheat": 3200, "Maize": 2400, "Sorghum": 1200,
    "Pearl Millet": 1000, "Chickpea": 900, "Pigeonpea": 800,
    "Groundnut": 1400, "Soybean": 1300, "Mustard": 1100,
    "Sunflower": 1000, "Cotton": 400, "Sugarcane": 65000,
    "Potato": 18000, "Onion": 14000, "Tomato": 20000,
    "Turmeric": 4500, "Banana": 30000, "Lentil": 750, "Barley": 2200,
}

# ── Season → typical sowing window ───────────────────────────────────────────
SOWING_WINDOWS = {
    "Kharif": "June – July",
    "Rabi":   "October – November",
    "Annual": "Year-round",
}

# ── Irrigation multipliers (yield boost over rainfed) ────────────────────────
IRRIG_BOOST = {
    "Rainfed": 1.00, "Canal": 1.15,
    "Borewell": 1.12, "Drip": 1.20, "Sprinkler": 1.18,
}


def yield_percentile(crop: str, yield_kg: float) -> float:
    """Return approx percentile rank (0–100) of yield_kg for given crop."""
    base = CROP_BASELINES.get(crop, 2000)
    # Simulate a normal distribution around base yield ±30%
    mu, sigma = base, base * 0.30
    from scipy.stats import norm
    return float(round(norm.cdf(yield_kg, mu, sigma) * 100, 1))


def format_yield(yield_kg: float, crop: str) -> str:
    """Return human-readable yield string with appropriate unit."""
    if yield_kg >= 10000:
        return f"{yield_kg/1000:.2f} tonnes/ha"
    return f"{yield_kg:.0f} kg/ha"


def season_from_month(month: int) -> str:
    if 6 <= month <= 9:
        return "Kharif"
    elif 10 <= month <= 2:
        return "Rabi"
    return "Annual"


def load_metadata(model_dir: str = "models") -> dict:
    path = os.path.join(model_dir, "metadata.json")
    with open(path) as f:
        return json.load(f)


def log_prediction(record: dict, log_path: str = "logs/predictions.jsonl"):
    """Append a prediction record to JSONL log for monitoring."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
