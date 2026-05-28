"""
Model Architectures:
1. Deep ANN with Residual Connections (primary)
2. Wide & Deep Network
3. Ensemble wrapper

All models predict log(yield_kg_ha); invert with expm1() at serving time.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import json, os

with open("models/metadata.json") as f:
    META = json.load(f)

N_FEATURES = META["n_features"]


# ── 1. Deep Residual ANN (Primary Production Model) ───────────────────────────

def build_residual_ann(
    n_features   = N_FEATURES,
    hidden_units = [256, 256, 128, 128, 64, 64],
    dropout_rate = 0.25,
    l2_reg       = 1e-4,
    learning_rate= 1e-3,
):
    """
    Deep ANN with skip connections (residual blocks).
    Each block: Dense → BatchNorm → ReLU → Dropout → Dense → Add(skip)
    """
    reg = regularizers.l2(l2_reg)

    inp = keras.Input(shape=(n_features,), name="features")
    x   = layers.Dense(256, kernel_regularizer=reg, name="stem")(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Activation("relu")(x)

    # Residual blocks
    block_sizes = [(256, 256), (128, 128), (64, 64)]
    for i, (d1, d2) in enumerate(block_sizes):
        # Shortcut projection if dimension changes
        shortcut = layers.Dense(d2, use_bias=False, name=f"skip_{i}")(x) if x.shape[-1] != d2 else x

        x = layers.Dense(d1, kernel_regularizer=reg, name=f"block{i}_dense1")(x)
        x = layers.BatchNormalization(name=f"block{i}_bn1")(x)
        x = layers.Activation("relu", name=f"block{i}_relu1")(x)
        x = layers.Dropout(dropout_rate, name=f"block{i}_drop1")(x)

        x = layers.Dense(d2, kernel_regularizer=reg, name=f"block{i}_dense2")(x)
        x = layers.BatchNormalization(name=f"block{i}_bn2")(x)

        x = layers.Add(name=f"block{i}_add")([x, shortcut])
        x = layers.Activation("relu", name=f"block{i}_out")(x)
        x = layers.Dropout(dropout_rate / 2, name=f"block{i}_drop2")(x)

    # Head
    x   = layers.Dense(32, activation="relu", kernel_regularizer=reg, name="head")(x)
    out = layers.Dense(1, name="output")(x)

    model = keras.Model(inp, out, name="ResidualANN")
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate, clipnorm=1.0),
        loss      = "huber",          # robust to outliers
        metrics   = ["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


# ── 2. Wide & Deep Network ────────────────────────────────────────────────────

def build_wide_deep(
    n_features   = N_FEATURES,
    deep_units   = [128, 64, 32],
    dropout_rate = 0.2,
    learning_rate= 8e-4,
):
    """
    Wide component: direct linear connections (memorisation)
    Deep component: stacked dense layers (generalisation)
    """
    inp   = keras.Input(shape=(n_features,), name="features")

    # Wide branch
    wide  = layers.Dense(n_features, use_bias=False, name="wide")(inp)

    # Deep branch
    deep  = inp
    for i, u in enumerate(deep_units):
        deep = layers.Dense(u, activation="relu", name=f"deep_{i}")(deep)
        deep = layers.BatchNormalization()(deep)
        deep = layers.Dropout(dropout_rate)(deep)

    combined = layers.Concatenate(name="combine")([wide, deep])
    out      = layers.Dense(1, name="output")(combined)

    model = keras.Model(inp, out, name="WideDeep")
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate),
        loss      = "mse",
        metrics   = ["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


# ── 3. Lightweight model (for TFLite export) ──────────────────────────────────

def build_lite_model(n_features=N_FEATURES):
    model = keras.Sequential([
        keras.Input(shape=(n_features,)),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.15),
        layers.Dense(32, activation="relu"),
        layers.Dense(1),
    ], name="LiteANN")
    model.compile(
        optimizer = keras.optimizers.Adam(1e-3),
        loss      = "huber",
        metrics   = ["mae", tf.keras.metrics.RootMeanSquaredError(name="rmse")],
    )
    return model


if __name__ == "__main__":
    m = build_residual_ann()
    m.summary()
    print(f"\nTotal parameters: {m.count_params():,}")
