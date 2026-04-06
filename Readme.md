# 📊 Telco Churn MLOps Monitoring System

An end-to-end MLOps project for **customer churn prediction** using the IBM Telco Churn dataset.  
This project covers the full ML lifecycle: **training → experiment tracking → model registry → API serving → drift monitoring → dashboard visualization → Dockerized deployment**.

---

## 🚀 Features

- Train and compare multiple models:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Select best model using **F1-score**
- Track experiments with **MLflow**
- Register best model in **MLflow Model Registry**
- Serve predictions with **FastAPI**
- Load model from **MLflow Registry** with local artifact fallback
- Generate synthetic batch data to simulate production traffic
- Detect **data drift** with **Evidently**
- Trigger **alerts** when drift or performance degradation occurs
- Visualize monitoring metrics using **Streamlit**
- Run all services with **Docker Compose**

---

## 🧱 Tech Stack

- **Python**
- **Scikit-learn**
- **XGBoost**
- **MLflow**
- **FastAPI**
- **Streamlit**
- **Evidently AI**
- **Docker / Docker Compose**

---

## 📂 Project Structure

```text
app/
  api/
  dashboard/
  monitoring/
  training/
scripts/
data/
artifacts/
screenshots/


⚙️ Local Setup
1. Clone the repo
git clone <YOUR_REPO_URL>
cd mlops-model-monitor
2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
3. Install dependencies
pip install -r requirements.txt
🏋️ Train the Model
python -m app.training.train

This will:

preprocess the dataset
train multiple models
log runs to MLflow
register the best model
save local fallback artifacts
📈 Generate Monitoring Data
Seed reference data
python -m scripts.seed_reference_data
Generate batches
python -m scripts.generate_batch --batch-id 1 --size 200
python -m scripts.generate_batch --batch-id 2 --size 200
python -m scripts.generate_batch --batch-id 3 --size 200 --drift --severity 0.15
python -m scripts.generate_batch --batch-id 4 --size 200 --drift --severity 0.25
python -m scripts.generate_batch --batch-id 5 --size 200 --drift --severity 0.35
Run monitoring
python -m scripts.run_monitoring
🐳 Run with Docker Compose
docker compose up --build
Services
MLflow UI → http://127.0.0.1:5000
FastAPI Docs → http://127.0.0.1:8000/docs
Streamlit Dashboard → http://127.0.0.1:8501
🔌 API Endpoints
GET /health
GET /model-info
POST /predict
Sample prediction response
{
  "prediction": 1,
  "probability": 0.71,
  "churn_label": "Yes",
  "model_source": "mlflow_registry",
  "registered_model_name": "telco_churn_classifier",
  "registered_model_version": "1"
}
📸 Screenshots

Add screenshots here:

MLflow experiment tracking
MLflow model registry
FastAPI docs
Dashboard overview
Drift trends
Alerts log
📌 Highlights
Registry-first model loading from MLflow Model Registry
Automatic fallback to local joblib artifact if registry is unavailable
Synthetic batch generation to simulate real-world drift scenarios
End-to-end monitoring with alerts and dashboards
🔮 Future Improvements
Deploy services publicly on Render / Railway / EC2
Add CI/CD pipeline with GitHub Actions
Add scheduled monitoring jobs
Store monitoring metrics in a database
Add model retraining trigger workflow

---

# 10) Before GitHub push: final local validation

Run this from project root:

```powershell id="c1vtew"
docker compose down
docker compose up -d --build
docker compose ps

Then test:

http://127.0.0.1:5000
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/health
http://127.0.0.1:8501

If all 4 work, repo is clean enough.

11) GitHub push commands (exact)
If repo not initialized yet:
git init
git add .
git commit -m "Initial commit: Telco Churn MLOps Monitoring System"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
If already initialized:
git add .
git commit -m "Finalized MLOps model monitoring project"
git push