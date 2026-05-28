"""
Training Script
- Trains ResidualANN on processed data
- Saves best checkpoint
- Full evaluation with R², RMSE, MAE, MAPE
- Exports TFLite lite model
- Saves training history JSON
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import json, os, time
import joblib

# ── Paths ─────────────────────────────────────────────────────────────────────
PROC_DIR    = "data/processed"
MODEL_DIR   = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
X_train = np.load(f"{PROC_DIR}/X_train.npy")
X_val   = np.load(f"{PROC_DIR}/X_val.npy")
X_test  = np.load(f"{PROC_DIR}/X_test.npy")
y_train = np.load(f"{PROC_DIR}/y_train.npy")
y_val   = np.load(f"{PROC_DIR}/y_val.npy")
y_test  = np.load(f"{PROC_DIR}/y_test.npy")

print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

with open(f"{MODEL_DIR}/metadata.json") as f:
    META = json.load(f)

# ── Import model builders ─────────────────────────────────────────────────────
import sys; sys.path.insert(0, "src/ml")
from model import build_residual_ann, build_lite_model

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath        = f"{MODEL_DIR}/best_model.keras",
        monitor         = "val_mae",
        save_best_only  = True,
        verbose         = 1,
    ),
    keras.callbacks.EarlyStopping(
        monitor   = "val_mae",
        patience  = 20,
        restore_best_weights = True,
        verbose   = 1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor  = "val_loss",
        factor   = 0.5,
        patience = 8,
        min_lr   = 1e-6,
        verbose  = 1,
    ),
    keras.callbacks.CSVLogger(f"{MODEL_DIR}/training_log.csv"),
]

# ── Train Primary Model ───────────────────────────────────────────────────────
print("\n" + "="*60)
print("Training ResidualANN")
print("="*60)

model = build_residual_ann()
model.summary()

t0 = time.time()
history = model.fit(
    X_train, y_train,
    validation_data = (X_val, y_val),
    epochs          = 150,
    batch_size      = 256,
    callbacks       = callbacks,
    verbose         = 1,
)
train_time = time.time() - t0
print(f"\nTraining time: {train_time:.1f}s")

# ── Evaluation ────────────────────────────────────────────────────────────────
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate(model, X, y_log, split_name):
    y_pred_log = model.predict(X, verbose=0).flatten()
    # Invert log transform
    y_true = np.expm1(y_log)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.clip(y_pred, 0, None)

    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    bias = np.mean(y_pred - y_true)

    print(f"\n── {split_name} Evaluation ──────────────────────")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAE  : {mae:.1f} kg/ha")
    print(f"  RMSE : {rmse:.1f} kg/ha")
    print(f"  MAPE : {mape:.2f}%")
    print(f"  Bias : {bias:.1f} kg/ha")
    return {"r2": round(r2,4), "mae": round(mae,1), "rmse": round(rmse,1),
            "mape": round(mape,2), "bias": round(bias,1)}

val_metrics  = evaluate(model, X_val,  y_val,  "Validation")
test_metrics = evaluate(model, X_test, y_test, "Test (held-out)")

# ── Save model & results ──────────────────────────────────────────────────────
model.save(f"{MODEL_DIR}/crop_yield_model.keras")
model.save(f"{MODEL_DIR}/crop_yield_model.h5")

results = {
    "model":        "ResidualANN",
    "n_features":   META["n_features"],
    "train_samples":META["n_train"],
    "val_metrics":  val_metrics,
    "test_metrics": test_metrics,
    "train_time_s": round(train_time, 1),
    "epochs_run":   len(history.history["loss"]),
}
with open(f"{MODEL_DIR}/evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Save history
hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
with open(f"{MODEL_DIR}/training_history.json", "w") as f:
    json.dump(hist_dict, f, indent=2)

# ── Train & Export Lite Model ─────────────────────────────────────────────────
print("\n" + "="*60)
print("Training LiteANN (for TFLite export)")
print("="*60)

lite_model = build_lite_model()
lite_model.fit(
    X_train, y_train,
    validation_data = (X_val, y_val),
    epochs          = 80,
    batch_size      = 256,
    callbacks       = [
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
    ],
    verbose         = 0,
)
evaluate(lite_model, X_test, y_test, "LiteModel Test")

# Convert to TFLite
converter        = tf.lite.TFLiteConverter.from_keras_model(lite_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model     = converter.convert()
tflite_path      = f"{MODEL_DIR}/crop_yield_lite.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
size_kb = os.path.getsize(tflite_path) / 1024
print(f"\n✅ TFLite model saved: {tflite_path} ({size_kb:.1f} KB)")

print("\n✅ Training complete! All artifacts saved to models/")
print(json.dumps(results, indent=2))
