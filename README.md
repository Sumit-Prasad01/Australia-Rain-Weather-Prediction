# Australia Weather Rain Prediction – End-to-End MLOps with GitHub Actions

An **end-to-end Machine Learning & MLOps project** for **Rain Prediction in Australia**, covering the complete lifecycle from **data processing and model training** to **CI/CD deployment using GitHub Actions** and **Kubernetes**.

This project is built with a strong **production and DevOps mindset**, making it ideal for ML Engineers and Data Scientists preparing for real-world systems.

---

## 📌 Project Overview

The objective of this project is to:
- Predict whether it will rain tomorrow based on historical Australian weather data
- Build a clean, modular ML training pipeline
- Develop a **Flask-based web application** for predictions
- Containerize the application using **Docker**
- Deploy the system using **Kubernetes**
- Automate CI/CD using **GitHub Actions only**

> ⚠️ Note: This repository **intentionally uses GitHub Actions as the sole CI/CD tool**.

---

## 🧠 Tech Stack

| Category | Tools |
|--------|------|
| Programming | Python |
| Machine Learning | Scikit-learn |
| Web Framework | Flask |
| Containerization | Docker |
| CI/CD | GitHub Actions |
| Orchestration | Kubernetes |
| Cloud Ready | GCP-compatible |
| Version Control | Git & GitHub |

---

## 📂 Project Structure

```
├── .github/workflows/             # GitHub Actions CI/CD pipelines
├── artifacts/                     # Trained models and artifacts
├── config/                        # Configuration files
├── notebooks/                     # Jupyter notebooks for experimentation
├── pipeline/                      # Training and inference pipelines
├── src/                           # Core ML source code
├── static/                        # Static files for Flask app
├── templates/                     # HTML templates
├── utils/                         # Utility functions
├── application.py                 # Flask application entry point
├── main.py                        # Training pipeline entry point
├── kubernetes-deployment.yaml     # Kubernetes deployment manifest
├── Dockerfile                     # Docker image configuration
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project metadata
├── setup.py                       # Package setup
├── README.md                      # Documentation
└── .gitignore
```

---

## ⚙️ Local Setup

### 1️⃣ Clone the Repository
```bash
git clone <repository-url>
cd australia-weather-rain-prediction
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate       # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🧪 Model Training Pipeline

The complete training workflow is executed via:

```bash
python main.py
```

This includes:
- Data ingestion
- Data preprocessing
- Feature engineering
- Model training
- Artifact generation

All outputs are stored in the `artifacts/` directory.

---

## 🌐 Flask Web Application

Run the Flask application locally:
```bash
python application.py
```

Access the UI at:
```
http://localhost:5000
```

---

## 🐳 Docker

### Build Docker Image
```bash
docker build -t australia-weather-rain .
```

### Run Container
```bash
docker run -p 5000:5000 australia-weather-rain
```

---

## 🔁 CI/CD with GitHub Actions

- CI/CD pipelines are defined inside `.github/workflows/`
- Automated workflow includes:
  - Code checkout
  - Dependency installation
  - Model pipeline execution
  - Docker image build
  - Kubernetes deployment

Every push to the main branch triggers the pipeline automatically.

---

## ☸️ Kubernetes Deployment

Apply the Kubernetes deployment:
```bash
kubectl apply -f kubernetes-deployment.yaml
```

Verify deployment:
```bash
kubectl get pods
kubectl get services
```

---

## 🚀 Key Features

- End-to-end ML pipeline
- Production-ready Flask application
- Dockerized deployment
- GitHub Actions–based CI/CD
- Kubernetes orchestration
- Clean and scalable codebase

---




