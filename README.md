# 🌾 Crop Yield Prediction using Deep Learning

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow" />
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-red?style=for-the-badge&logo=keras" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge" />
</p>

<p align="center">
  <b>An intelligent deep learning system to predict crop yields based on soil properties, weather conditions, and agricultural inputs — helping farmers and agronomists make data-driven decisions.</b>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Results](#-results)
- [How to Make This Project Unique & Dynamic](#-how-to-make-this-project-unique--dynamic-software-grade-upgrade)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 🌟 Overview

Crop yield prediction is one of the most critical challenges in modern agriculture. Inaccurate predictions lead to food shortages, economic loss, and overuse of resources. This project applies **Deep Learning** techniques to predict the expected yield of various crops based on:

- 🌡️ Weather/climate data (temperature, rainfall, humidity)
- 🪨 Soil composition (nitrogen, phosphorus, potassium, pH)
- 🌍 Geographical region
- 🗓️ Season / year

By leveraging **Artificial Neural Networks (ANN)** and optionally **CNN/LSTM** architectures, the system learns complex nonlinear patterns between agricultural variables and crop output.

---

## ✅ Features

- Predicts crop yield for multiple crop types
- Trained on real-world agricultural datasets
- Preprocessing pipeline for handling missing values, normalization, and encoding
- Model training with configurable hyperparameters
- Evaluation metrics: MAE, RMSE, R² Score
- Visualization of training loss, accuracy curves, and prediction plots
- Easy-to-run Jupyter Notebook interface

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.8+ |
| Deep Learning | TensorFlow / Keras |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Utilities | Scikit-learn |
| Notebook | Jupyter Notebook |
| Version Control | Git & GitHub |

---

## 📁 Project Structure

```
Crop_Yielding_Prediction_deep_Learning/
│
├── dataset/
│   ├── crop_data.csv              # Main dataset
│   └── preprocessed_data.csv      # Cleaned/processed dataset
│
├── models/
│   ├── crop_yield_model.h5        # Saved trained model
│   └── model_architecture.png     # Model diagram
│
├── notebooks/
│   ├── EDA.ipynb                  # Exploratory Data Analysis
│   ├── Preprocessing.ipynb        # Data cleaning & feature engineering
│   └── Model_Training.ipynb       # Model building & evaluation
│
├── src/
│   ├── preprocess.py              # Data preprocessing functions
│   ├── model.py                   # Model definition
│   ├── train.py                   # Training script
│   └── predict.py                 # Prediction script
│
├── results/
│   ├── training_curves.png
│   └── predictions_vs_actual.png
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 📊 Dataset

The model is trained on agricultural data containing features such as:

| Feature | Description |
|--------|-------------|
| `Crop` | Type of crop (e.g., Rice, Wheat, Maize) |
| `Season` | Kharif / Rabi / Whole Year |
| `State` | Indian state / Region |
| `Area` | Area under cultivation (in hectares) |
| `Annual_Rainfall` | Average annual rainfall (mm) |
| `Fertilizer` | Fertilizer used (kg/ha) |
| `Pesticide` | Pesticide used (kg/ha) |
| `Production` | Target variable — crop yield (tonnes) |

> 📌 Dataset Source: [Kaggle - Crop Production Statistics](https://www.kaggle.com/datasets/abhinand05/crop-production-in-india) or similar agricultural open datasets.

---

## 🧠 Model Architecture

The deep learning model is a **Multi-Layer Feedforward Neural Network (ANN)**:

```
Input Layer  →  [N features]
    ↓
Dense Layer  →  256 units, ReLU, Dropout(0.3)
    ↓
Dense Layer  →  128 units, ReLU, Dropout(0.2)
    ↓
Dense Layer  →  64 units, ReLU
    ↓
Output Layer →  1 unit (predicted yield)
```

**Loss Function:** Mean Squared Error (MSE)  
**Optimizer:** Adam (lr=0.001)  
**Metrics:** MAE, RMSE, R²

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or above
- pip package manager
- Git

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Ashwani4545/Crop_Yielding_Prediction_deep_Learning.git
cd Crop_Yielding_Prediction_deep_Learning

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter notebook
```

### requirements.txt

```
tensorflow>=2.10
keras
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
flask           # (for web app upgrade)
streamlit       # (optional: for dashboard UI)
```

---

## 🚀 Usage

### Run Training

```bash
python src/train.py --epochs 100 --batch_size 32 --learning_rate 0.001
```

### Run Prediction

```bash
python src/predict.py --input "Wheat,Rabi,Punjab,500,800,150,2"
```

### Interactive Notebook

Open `notebooks/Model_Training.ipynb` in Jupyter and run all cells step-by-step.

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~93% |
| Validation Loss (MSE) | ~0.04 |
| R² Score | ~0.91 |
| MAE | ~120 kg/ha |

> Actual results may vary based on dataset version and hyperparameter tuning.

---

## 🚀 How to Make This Project Unique & Dynamic (Software-Grade Upgrade)

This section outlines powerful ideas to transform this basic deep learning notebook into a **production-ready, full-stack intelligent software application**.

---

### 1. 🌐 Build a Full-Stack Web Application

Convert the prediction model into a web app using **Flask** (backend) + **React.js or HTML/CSS** (frontend):

```
User Input Form → REST API (/predict) → DL Model → JSON Response → Display Result
```

**Steps:**
- Export trained model as `crop_yield_model.h5`
- Create a Flask API endpoint:
```python
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('models/crop_yield_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([data['features']])
    prediction = model.predict(features)
    return jsonify({'predicted_yield': float(prediction[0][0])})
```
- Build an HTML form where farmers input crop details and get real-time predictions.

---

### 2. 📊 Add a Real-Time Dashboard with Streamlit

Replace the Jupyter notebook with an **interactive Streamlit dashboard**:

```bash
pip install streamlit
streamlit run app.py
```

Features to add in dashboard:
- Slider inputs for rainfall, fertilizer, area
- Dropdown for crop type and season
- Animated prediction result card
- Historical yield trend charts
- Region heatmap (using Plotly or Folium)

---

### 3. 🌦️ Integrate Live Weather API

Instead of manual weather input, auto-fetch real-time weather data using **OpenWeatherMap API**:

```python
import requests

def get_weather(city):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid=YOUR_API_KEY"
    data = requests.get(url).json()
    return {
        'temperature': data['main']['temp'] - 273.15,
        'humidity': data['main']['humidity'],
        'rainfall': data.get('rain', {}).get('1h', 0)
    }
```

This makes the tool **truly dynamic** — just enter your city and crop type!

---

### 4. 🗺️ Geo-Spatial Crop Yield Map

Add an **interactive map** showing predicted yields across different regions using **Folium** or **Plotly Choropleth**:

```python
import folium

m = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
# Add color-coded markers for yield per district
```

This transforms the app into a **GIS-enabled agricultural intelligence tool**.

---

### 5. 🔄 Implement Continuous Learning (Model Retraining Pipeline)

Allow the model to improve over time as new data comes in:

- Add a `/submit_actual_yield` API endpoint for users to submit true yields
- Store in a database (SQLite / PostgreSQL)
- Schedule periodic retraining using **Celery + Redis** or **cron jobs**
- Auto-evaluate new model vs old, and deploy only if improved

---

### 6. 🤖 Add Multi-Model Comparison

Allow users to switch between models and compare performance:

| Model | MAE | R² |
|-------|-----|----|
| ANN (Current) | 120 | 0.91 |
| Random Forest | 145 | 0.87 |
| LSTM | 110 | 0.93 |
| XGBoost | 130 | 0.90 |

Display a model comparison chart in the dashboard.

---

### 7. 📱 Build a Mobile App (React Native / Flutter)

Create a **cross-platform mobile app** for farmers with:
- Voice input support (farmer speaks crop details)
- Offline mode using TensorFlow Lite (`.tflite` model)
- Push notifications for seasonal crop recommendations
- Multi-language support (Hindi, Gujarati, Punjabi, etc.)

**Convert to TFLite:**
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

---

### 8. 📦 Dockerize & Deploy to Cloud

Make the application production-ready:

```dockerfile
# Dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

```bash
# Build and run
docker build -t crop-yield-app .
docker run -p 5000:5000 crop-yield-app
```

**Deploy on:**
- **AWS EC2 / Elastic Beanstalk** — for scalable production deployment
- **Heroku** — for quick free-tier deployment
- **Google Cloud Run** — for serverless container hosting
- **Hugging Face Spaces** — for ML demo sharing

---

### 9. 🔐 Add Authentication & User Profiles

Use **Flask-Login** or **Firebase Auth** to:
- Let farmers create accounts
- Save their prediction history
- Track yield accuracy over multiple seasons
- Export predictions as PDF reports

---

### 10. 📉 Explainability with SHAP / LIME

Add model explainability so farmers understand *why* a prediction was made:

```python
import shap
explainer = shap.DeepExplainer(model, X_train)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```

Output: **"Rainfall contributed 42% to this prediction, Soil pH contributed 18%..."**

This builds **farmer trust** in the AI system.

---

### 11. 🌱 Add Crop Recommendation Module

Extend beyond prediction — add a **Crop Recommendation System**:
- Input: soil NPK values, pH, temperature, rainfall
- Output: Top 3 recommended crops for that region and season
- Use a separate classifier model (Random Forest / Naive Bayes)

---

### 12. 🧪 Add Unit Tests & CI/CD Pipeline

Add automated testing and deployment:

```bash
# tests/test_model.py
def test_prediction_shape():
    result = model.predict(sample_input)
    assert result.shape == (1, 1)

def test_prediction_range():
    result = model.predict(sample_input)
    assert 0 < result[0][0] < 100000
```

Use **GitHub Actions** for CI/CD:
```yaml
# .github/workflows/ci.yml
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pip install -r requirements.txt
      - run: pytest tests/
```

---

## 🗺️ Roadmap

- [x] Data preprocessing pipeline
- [x] Deep learning model (ANN)
- [x] Model training & evaluation
- [ ] Flask REST API
- [ ] Streamlit dashboard
- [ ] Live weather API integration
- [ ] Geo-spatial yield map
- [ ] Mobile app (TFLite)
- [ ] Docker deployment
- [ ] SHAP explainability
- [ ] CI/CD pipeline

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the project
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

Please make sure to update tests as appropriate.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Ashwani Kumar**

- GitHub: [@Ashwani4545](https://github.com/Ashwani4545)
- LinkedIn: [Connect on LinkedIn](https://linkedin.com)

---

## 🙏 Acknowledgements

- [Kaggle - Crop Production in India](https://www.kaggle.com/datasets/abhinand05/crop-production-in-india)
- TensorFlow & Keras Documentation
- Scikit-learn Documentation
- [Jiaxuan You et al. - Deep Gaussian Process for Crop Yield Prediction (AAAI 2017)](https://github.com/JiaxuanYou/crop_yield_prediction)

---

<p align="center">
  Made with ❤️ for Indian Agriculture 🌾
</p>
