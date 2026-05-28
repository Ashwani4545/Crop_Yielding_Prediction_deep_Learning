"""
Dataset Generator — ICRISAT-style synthetic agricultural dataset
Covers 20 crops × 30 districts × 30 years × 2 seasons = ~36,000 rows
Features: soil, weather, fertilizer, irrigation, yield
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)

# ── Constants ────────────────────────────────────────────────────────────────

STATES_DISTRICTS = {
    "Gujarat":      ["Ahmedabad","Anand","Banaskantha","Gandhinagar","Junagadh","Kutch","Mehsana","Rajkot","Surat","Vadodara"],
    "Punjab":       ["Amritsar","Bathinda","Fatehgarh Sahib","Gurdaspur","Ludhiana","Mansa","Patiala","Rupnagar","Sangrur","SAS Nagar"],
    "Maharashtra":  ["Ahmednagar","Aurangabad","Kolhapur","Latur","Nagpur","Nashik","Pune","Sangli","Satara","Solapur"],
    "Uttar Pradesh":["Agra","Aligarh","Allahabad","Bareilly","Ghaziabad","Gorakhpur","Kanpur","Lucknow","Mathura","Varanasi"],
    "Rajasthan":    ["Ajmer","Alwar","Barmer","Bikaner","Jaipur","Jaisalmer","Jodhpur","Kota","Nagaur","Udaipur"],
    "Madhya Pradesh":["Bhopal","Gwalior","Indore","Jabalpur","Rewa","Sagar","Satna","Ujjain","Vidisha","Ratlam"],
    "Karnataka":    ["Bagalkote","Belgaum","Bellary","Bidar","Bijapur","Dharwad","Gulbarga","Hassan","Mysore","Tumkur"],
    "Andhra Pradesh":["Anantapur","Chittoor","East Godavari","Guntur","Krishna","Kurnool","Nellore","Prakasam","Srikakulam","Visakhapatnam"],
    "Haryana":      ["Ambala","Bhiwani","Faridabad","Fatehabad","Hisar","Jind","Karnal","Kurukshetra","Rohtak","Sirsa"],
    "Bihar":        ["Araria","Bhagalpur","Darbhanga","Gaya","Muzaffarpur","Nalanda","Patna","Purnea","Rohtas","Samastipur"],
}

CROPS = {
    # crop: (base_yield_kg_ha, season, soil_pref, rain_need_mm)
    "Rice":         (2800, "Kharif",  "Clay",        1200),
    "Wheat":        (3200, "Rabi",    "Loam",         450),
    "Maize":        (2400, "Kharif",  "Sandy Loam",   600),
    "Sorghum":      (1200, "Kharif",  "Sandy Loam",   400),
    "Pearl Millet": (1000, "Kharif",  "Sandy",        350),
    "Chickpea":     (900,  "Rabi",    "Loam",         350),
    "Pigeonpea":    (800,  "Kharif",  "Loam",         700),
    "Groundnut":    (1400, "Kharif",  "Sandy Loam",   500),
    "Soybean":      (1300, "Kharif",  "Clay Loam",    600),
    "Mustard":      (1100, "Rabi",    "Sandy Loam",   300),
    "Sunflower":    (1000, "Rabi",    "Loam",         450),
    "Cotton":       (400,  "Kharif",  "Black",        700),
    "Sugarcane":    (65000,"Annual",  "Clay Loam",    1500),
    "Potato":       (18000,"Rabi",    "Sandy Loam",   500),
    "Onion":        (14000,"Rabi",    "Loam",         400),
    "Tomato":       (20000,"Kharif",  "Loam",         600),
    "Turmeric":     (4500, "Kharif",  "Clay Loam",    1200),
    "Banana":       (30000,"Annual",  "Clay Loam",    1200),
    "Lentil":       (750,  "Rabi",    "Loam",         300),
    "Barley":       (2200, "Rabi",    "Sandy Loam",   350),
}

SOIL_TYPES   = ["Sandy", "Sandy Loam", "Loam", "Clay Loam", "Clay", "Black", "Red", "Alluvial"]
IRRIGATIONS  = ["Rainfed", "Canal", "Borewell", "Drip", "Sprinkler"]
YEARS        = list(range(1992, 2024))

# ── State-level climate baselines ─────────────────────────────────────────────

STATE_CLIMATE = {
    "Gujarat":          {"rain": 700,  "tmax": 36, "tmin": 18, "humid": 55},
    "Punjab":           {"rain": 600,  "tmax": 33, "tmin": 10, "humid": 60},
    "Maharashtra":      {"rain": 900,  "tmax": 34, "tmin": 18, "humid": 65},
    "Uttar Pradesh":    {"rain": 850,  "tmax": 35, "tmin": 12, "humid": 62},
    "Rajasthan":        {"rain": 380,  "tmax": 40, "tmin": 14, "humid": 40},
    "Madhya Pradesh":   {"rain": 950,  "tmax": 35, "tmin": 14, "humid": 58},
    "Karnataka":        {"rain": 800,  "tmax": 32, "tmin": 17, "humid": 62},
    "Andhra Pradesh":   {"rain": 950,  "tmax": 34, "tmin": 19, "humid": 68},
    "Haryana":          {"rain": 550,  "tmax": 34, "tmin": 10, "humid": 55},
    "Bihar":            {"rain": 1100, "tmax": 34, "tmin": 13, "humid": 70},
}

# ── Yield simulation ──────────────────────────────────────────────────────────

def simulate_yield(crop, state, year, rainfall, tmax, n_kg, irrigation, soil_type):
    base, _, soil_pref, rain_need = CROPS[crop]

    # Rainfall effect: optimal = rain_need, penalise deviation
    rain_ratio = min(rainfall / rain_need, 1.4)
    rain_factor = 1.0 - 0.5 * max(0, 1.0 - rain_ratio) - 0.2 * max(0, rain_ratio - 1.3)

    # Temperature stress (heat stress > 35°C)
    temp_factor = 1.0 - max(0, (tmax - 35) * 0.03)

    # Nitrogen fertilizer effect (diminishing returns)
    n_factor = 0.6 + 0.4 * min(n_kg / 120, 1.0)

    # Irrigation bonus
    irrig_mult = {"Rainfed": 1.0, "Canal": 1.15, "Borewell": 1.12,
                  "Drip": 1.20, "Sprinkler": 1.18}[irrigation]

    # Soil match bonus
    soil_bonus = 1.05 if soil_type == soil_pref else 0.95

    # Technology trend (+0.8% yield gain per year)
    trend = 1.0 + (year - 1992) * 0.008

    # Climate change penalty after 2010
    climate_pen = 1.0 - max(0, (year - 2010) * 0.003)

    yield_val = (base * rain_factor * temp_factor * n_factor *
                 irrig_mult * soil_bonus * trend * climate_pen)

    # Add realistic noise (±12%)
    noise = np.random.normal(1.0, 0.12)
    return max(0, round(yield_val * noise, 1))

# ── Generate rows ─────────────────────────────────────────────────────────────

rows = []
for state, districts in STATES_DISTRICTS.items():
    clim = STATE_CLIMATE[state]
    for district in districts:
        # District-level soil (fixed)
        soil_type = np.random.choice(SOIL_TYPES)
        ph        = round(np.random.uniform(5.5, 8.0), 1)
        n_soil    = round(np.random.uniform(150, 450), 1)   # kg/ha available N
        p_soil    = round(np.random.uniform(10, 60), 1)
        k_soil    = round(np.random.uniform(100, 350), 1)
        oc        = round(np.random.uniform(0.3, 1.8), 2)   # organic carbon %
        elevation = int(np.random.uniform(50, 900))

        for year in YEARS:
            # Yearly climate variation
            rainfall  = max(100, round(np.random.normal(clim["rain"], clim["rain"]*0.22)))
            tmax      = round(np.random.normal(clim["tmax"], 2.0), 1)
            tmin      = round(np.random.normal(clim["tmin"], 1.5), 1)
            humidity  = round(np.random.normal(clim["humid"], 8), 1)
            solar_rad = round(np.random.uniform(14, 22), 1)  # MJ/m²/day

            for crop, (_, season, _, _) in CROPS.items():
                # Not every crop grown in every district every year
                if np.random.random() > 0.55:
                    continue

                irrigation = np.random.choice(
                    IRRIGATIONS,
                    p=[0.40, 0.20, 0.25, 0.08, 0.07]
                )
                fertilizer_n = round(np.random.uniform(20, 200), 1)
                fertilizer_p = round(np.random.uniform(10, 80), 1)
                fertilizer_k = round(np.random.uniform(10, 80), 1)
                pesticide    = round(np.random.uniform(0.2, 4.5), 2)
                area_ha      = round(np.random.uniform(50, 12000), 0)

                # Growing Degree Days proxy
                gdd = round((tmax + tmin) / 2 * 120, 0)

                # NDVI proxy (0.3–0.9)
                ndvi = round(np.random.uniform(0.3, 0.9), 3)

                yield_val = simulate_yield(
                    crop, state, year, rainfall,
                    tmax, fertilizer_n, irrigation, soil_type
                )

                rows.append({
                    "state":           state,
                    "district":        district,
                    "crop":            crop,
                    "season":          season,
                    "year":            year,
                    # Weather
                    "annual_rainfall_mm":  rainfall,
                    "tmax_celsius":        tmax,
                    "tmin_celsius":        tmin,
                    "humidity_pct":        humidity,
                    "solar_radiation":     solar_rad,
                    "gdd":                 gdd,
                    # Soil
                    "soil_type":           soil_type,
                    "soil_ph":             ph,
                    "soil_n_kgha":         n_soil,
                    "soil_p_kgha":         p_soil,
                    "soil_k_kgha":         k_soil,
                    "organic_carbon_pct":  oc,
                    "elevation_m":         elevation,
                    # Inputs
                    "fertilizer_n_kgha":   fertilizer_n,
                    "fertilizer_p_kgha":   fertilizer_p,
                    "fertilizer_k_kgha":   fertilizer_k,
                    "pesticide_kgha":      pesticide,
                    "irrigation_type":     irrigation,
                    "area_ha":             area_ha,
                    # Remote sensing
                    "ndvi":                ndvi,
                    # Target
                    "yield_kg_ha":         yield_val,
                })

df = pd.DataFrame(rows)
print(f"Generated {len(df):,} rows | {df['crop'].nunique()} crops | "
      f"{df['district'].nunique()} districts | {df['year'].nunique()} years")
print(df.dtypes)
print(df.describe())

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/crop_yield_india.csv", index=False)
print("\nSaved → data/raw/crop_yield_india.csv")
