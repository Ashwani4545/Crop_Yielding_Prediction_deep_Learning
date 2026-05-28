# ─────────────────────────────────────────────────────────────────────────────
# KrishiAI — Makefile
# Usage: make <target>
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help setup data preprocess train test api frontend docker clean

PYTHON = python3
PIP    = pip3

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  🌾  KrishiAI — Crop Yield Prediction System"
	@echo ""
	@echo "  Usage: make <target>"
	@echo ""
	@echo "  Setup & Data"
	@echo "  ────────────────────────────────────"
	@echo "  setup        Install all Python dependencies"
	@echo "  data         Generate synthetic training dataset"
	@echo "  preprocess   Clean data & engineer features"
	@echo ""
	@echo "  ML Pipeline"
	@echo "  ────────────────────────────────────"
	@echo "  train        Train the deep learning model"
	@echo "  evaluate     Run evaluation on test set"
	@echo "  predict      Run a sample CLI prediction"
	@echo ""
	@echo "  Testing"
	@echo "  ────────────────────────────────────"
	@echo "  test         Run full pytest suite"
	@echo "  test-fast    Run unit tests only (no model load)"
	@echo ""
	@echo "  Deployment"
	@echo "  ────────────────────────────────────"
	@echo "  api          Start FastAPI server (port 8000)"
	@echo "  frontend     Serve frontend (port 3000)"
	@echo "  docker       Build & start Docker stack"
	@echo "  docker-stop  Stop Docker stack"
	@echo ""
	@echo "  Utilities"
	@echo "  ────────────────────────────────────"
	@echo "  clean        Remove generated files"
	@echo "  pipeline     Run full pipeline: data→preprocess→train→test"
	@echo ""

# ── Setup ────────────────────────────────────────────────────────────────────
setup:
	@echo "📦 Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "✅ Dependencies installed"

# ── Data ─────────────────────────────────────────────────────────────────────
data:
	@echo "🌱 Generating dataset..."
	$(PYTHON) scripts/generate_dataset.py
	@echo "✅ Dataset saved to data/raw/"

# ── Preprocess ───────────────────────────────────────────────────────────────
preprocess:
	@echo "⚙️  Running preprocessing pipeline..."
	$(PYTHON) src/ml/preprocess.py
	@echo "✅ Processed data saved to data/processed/"

# ── Train ────────────────────────────────────────────────────────────────────
train:
	@echo "🧠 Training deep learning model..."
	$(PYTHON) src/ml/train.py
	@echo "✅ Model saved to models/"

# ── Evaluate ─────────────────────────────────────────────────────────────────
evaluate:
	@echo "📊 Running evaluation..."
	$(PYTHON) -c "
import numpy as np, tensorflow as tf, json
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings; warnings.filterwarnings('ignore')
model  = tf.keras.models.load_model('models/best_model.keras')
X_test = np.load('data/processed/X_test.npy')
y_test = np.load('data/processed/y_test.npy')
y_pred = np.expm1(model.predict(X_test, verbose=0).flatten())
y_true = np.expm1(y_test)
print(f'  R²:   {r2_score(y_true,y_pred):.4f}')
print(f'  MAE:  {mean_absolute_error(y_true,y_pred):.1f} kg/ha')
print(f'  RMSE: {np.sqrt(mean_squared_error(y_true,y_pred)):.1f} kg/ha')
print(f'  MAPE: {np.mean(np.abs((y_true-y_pred)/(y_true+1)))*100:.2f}%')
"

# ── Sample prediction ─────────────────────────────────────────────────────────
predict:
	@echo "🔮 Running sample prediction..."
	$(PYTHON) src/ml/predict.py \
		--crop Wheat --state Gujarat --district Anand \
		--season Rabi --year 2024 --rainfall 650 \
		--tmax 33 --tmin 14 --n 120 --p 50 --k 50 \
		--irrigation Canal --soil Loam

# ── Test ─────────────────────────────────────────────────────────────────────
test:
	@echo "🧪 Running tests..."
	$(PYTHON) -m pytest tests/ -v --tb=short

test-fast:
	@echo "🧪 Running unit tests..."
	$(PYTHON) -m pytest tests/ -v --tb=short -k "not Inference and not Quality"

# ── API ──────────────────────────────────────────────────────────────────────
api:
	@echo "🚀 Starting FastAPI server on http://localhost:8000 ..."
	@echo "   Docs: http://localhost:8000/docs"
	cd src/api && uvicorn app:app --reload --host 0.0.0.0 --port 8000

# ── Frontend ─────────────────────────────────────────────────────────────────
frontend:
	@echo "🌐 Serving frontend on http://localhost:3000 ..."
	$(PYTHON) -m http.server 3000 --directory frontend

# ── Docker ───────────────────────────────────────────────────────────────────
docker:
	@echo "🐳 Building and starting Docker stack..."
	cd deployment/docker && docker compose up --build -d
	@echo "✅ API:      http://localhost:8000"
	@echo "✅ Frontend: http://localhost:3000"
	@echo "✅ API Docs: http://localhost:8000/docs"

docker-stop:
	@echo "🛑 Stopping Docker stack..."
	cd deployment/docker && docker compose down

# ── Full pipeline ─────────────────────────────────────────────────────────────
pipeline: data preprocess train test
	@echo ""
	@echo "🎉 Full pipeline complete!"
	@echo "   Run 'make api' to start the server"
	@echo "   Run 'make frontend' to serve the dashboard"

# ── Clean ────────────────────────────────────────────────────────────────────
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf data/raw/*.csv data/processed/*.npy data/processed/*.csv
	rm -rf models/*.keras models/*.h5 models/*.tflite models/*.pkl
	rm -rf models/training_log.csv models/training_history.json
	rm -rf logs/ .pytest_cache/ __pycache__/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Clean done"
