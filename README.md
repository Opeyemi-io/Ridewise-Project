# Ridewise-Project

# 🚗 RideWise — Customer Analytics & Churn Prediction

> A machine learning system that helps RideWise **optimize customer lifetime value and reduce churn** through data-driven behavioral insights and predictive analytics.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Serving-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![Status](https://img.shields.io/badge/status-active-success)

---

## 🖼️ Demo & Screenshots

> Replace the placeholders below with real captures once the dashboard and API are live.

| Analytics Dashboard | Churn Risk Scoring |
|---|---|
| ![Dashboard screenshot](docs/images/dashboard.png) | ![Churn scoring screenshot](docs/images/churn_scoring.png) |

<p align="center">
  <img src="docs/images/demo.gif" alt="RideWise demo walkthrough" width="700"/>
</p>

---

## 📌 Overview

**RideWise** is a European mobility-tech platform serving **200,000+ customers** and powering **800,000+ monthly trips** across London, Berlin, Amsterdam, Barcelona, and Milan. Despite strong growth, it loses **~25% of regular users every quarter** and runs promotions with low uptake and unclear ROI.

This project replaces slow, manual customer analysis with an **automated, predictive analytics pipeline** that segments customers, flags churn risk before it happens, and serves insights in real time.

## ❗ The Problem

- 🔻 **25% quarterly churn** among regular users
- 🕵️ Limited understanding of behavioral patterns and customer segments
- 🎯 Promotions with low uptake and **unmeasurable ROI**
- ⏳ Manual analysis takes **weeks per campaign**, throttling agility
- 🚫 No predictive way to identify at-risk customers early
- 📉 No integrated, real-time analytics for operational decisions

## 🎯 Objectives

- Build **churn prediction** models to flag at-risk customers
- Create **customer segmentation** with clustering
- Engineer a **behavioral feature pipeline** (RFM, usage, temporal)
- Deliver **business-friendly model interpretability**
- Deploy **real-time scoring** via API
- Ship an **analytics dashboard** with basic monitoring

## 🗂️ Dataset

Synthetic, privacy-safe data spanning **12 months** across 5 cities.

| Table | Description |
|---|---|
| `customers` | Profiles, signup, city, loyalty & account status |
| `trips` | 800K+ journeys: distance, fare, rating, payment |
| `customer_behavior` | Monthly aggregates: trips, spend, usage ratios, cancellations |
| `promotions` | Campaign type, discount, usage tracking |
| `external_factors` | Weather, holidays, events, competitor pricing |

**Structure:** one-to-many (customer → trips/promos), time-series-ready for churn and trend modeling.

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| 🤖 Machine Learning | scikit-learn, pandas, NumPy |
| 🧬 Feature Engineering | Custom pipelines, RFM analysis, behavioral metrics |
| 🔍 Interpretability | Feature importance, business-rule explanations |
| 📊 Visualization | matplotlib, seaborn, Streamlit |
| 📈 Experiment Tracking | MLflow |
| ⚙️ Backend / Serving | FastAPI |
| 🩺 Monitoring | Logging, basic performance tracking |
| 🚀 Deployment | Docker, FastAPI, Streamlit |

**Models:** K-Means (segmentation) · Logistic Regression & Random Forest (churn, chosen for interpretability), validated via time-based splits and cross-validation.

## 🔄 Workflow

| Step | Stage | Focus |
|---|---|---|
| 1️⃣ | **Data Analysis & EDA** | Explore customer and trip data; analyze churn patterns and behaviors |
| 2️⃣ | **Feature Engineering** | RFM metrics, usage patterns, temporal and behavioral indicators |
| 3️⃣ | **Customer Segmentation** | K-Means clustering into business-meaningful groups |
| 4️⃣ | **Churn Modeling** | Logistic Regression and Random Forest, tuned for interpretability |
| 5️⃣ | **Validation** | Time-based splits, cross-validation, business sense-checking |
| 6️⃣ | **API Development** | FastAPI real-time scoring endpoints + Streamlit dashboard |
| 7️⃣ | **Deployment & Monitoring** | Dockerized service, logging, and segment/performance tracking |

## 📊 Success Metrics

| Focus Area | Target |
|---|---|
| Customer Segmentation | ≥3 distinct, business-meaningful segments |
| Churn Prediction | AUC > 0.80; surface top 15% at-risk customers |
| Latency | < 1 second real-time scoring |
| API Performance | 99% uptime, functional core endpoints |
| Dashboard | 5 key metrics, clear visualizations |
| Insights | Actionable recommendations per segment |

## 🧩 Technical Challenges & Solutions

- **Class imbalance** → balanced sampling, precision/recall focus, threshold tuning
- **Interpretability for business** → feature importance + plain-language segment descriptions
- **Real-time scoring** → lightweight models, efficient API, basic caching
- **Data quality** → robust cleaning, validation checks, simple imputation

## 🚀 Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/ridewise.git
cd ridewise

# 2. Set up environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Run the API
uvicorn app.main:app --reload

# 4. Launch the dashboard
streamlit run dashboard/app.py
```

The FastAPI service exposes real-time customer scoring endpoints; the Streamlit dashboard surfaces segment performance and churn insights.

## 🐳 Deployment

Containerize and run the scoring service with Docker for portable, reproducible deployment across environments:

```bash
# Build the image
docker build -t ridewise-api .

# Run the container
docker run -p 8000:8000 ridewise-api
```

The API is then available at `http://localhost:8000` (interactive docs at `/docs`). Basic logging and segment-stability tracking keep the model monitored in production.

## 📦 Deliverables

- 🧮 Customer Segmentation Model (K-Means + interpretation)
- 🔮 Churn Prediction System (LogReg + Random Forest)
- 🧬 Feature Engineering Pipeline (RFM + behavioral metrics)
- 💬 Model Interpretation Framework
- ⚡ Real-time Scoring API (FastAPI)
- 📊 Customer Analytics Dashboard
- 🩺 Basic Monitoring System
- 📄 Technical Documentation & Business Insights Report
- 🎤 Executive Presentation with ROI analysis

## 📁 Project Structure

```
ridewise/
├── app/                      # FastAPI service
│   ├── main.py               # API entry point & routes
│   ├── schemas.py            # Request/response models
│   └── scoring.py            # Real-time churn scoring logic
├── dashboard/                # Streamlit analytics dashboard
│   └── app.py
├── data/
│   ├── raw/                  # Source tables (customers, trips, etc.)
│   └── processed/            # Cleaned & feature-engineered data
├── notebooks/                # EDA, modeling & experimentation
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_segmentation.ipynb
│   └── 04_churn_modeling.ipynb
├── src/
│   ├── features/             # RFM & behavioral feature pipelines
│   ├── models/               # Training & evaluation
│   └── utils/                # Helpers & config
├── models/                   # Serialized models (.pkl)
├── docs/
│   └── images/               # Screenshots & demo GIF
├── tests/                    # Unit & integration tests
├── requirements.txt
├── Dockerfile                # (optional) containerized deployment
└── README.md
```

## 👥 Contributors

| Name | Role | Contact |
|---|---|---|
| Opeyemi Adegoke | Project Lead / Data Scientist | [LinkedIn](https://linkedin.com/in/<your-handle>) · [GitHub](https://github.com/<your-username>) |

Contributions are welcome! Fork the repo, create a feature branch, and open a pull request. For major changes, please open an issue first to discuss what you'd like to change.

## 📄 License

_No license specified. Add one here if you intend to make the project open source._

---

*Built to demonstrate production-ready customer analytics for mobility applications — segmentation, churn modeling, and ML deployment.*