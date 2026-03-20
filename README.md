# Predictive Maintenance Copilot

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python" />
  </a>
  <a href="https://streamlit.io/">
    <img src="https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  </a>
  <a href="https://www.databricks.com/">
    <img src="https://img.shields.io/badge/Platform-Databricks-EA4335?style=for-the-badge&logo=databricks&logoColor=white" />
  </a>
  <a href="https://delta.io/">
    <img src="https://img.shields.io/badge/Data%20Layer-Delta%20Lake-003B57?style=for-the-badge" />
  </a>
  <a href="https://mlflow.org/">
    <img src="https://img.shields.io/badge/Experiment%20Tracking-MLflow-0194E2?style=for-the-badge" />
  </a>
  <a href="https://plotly.com/python/">
    <img src="https://img.shields.io/badge/Visualization-Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" />
  </a>
  <a href="https://ai.google.dev/">
    <img src="https://img.shields.io/badge/LLM-Gemini%203%20Flash-4285F4?style=for-the-badge&logo=googlebard&logoColor=white" />
  </a>
  <a href="https://github.com/anchitchourasia/maintenance-copilot-app">
    <img src="https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github" />
  </a>
</p>

<p align="center">
  <b>A production-style predictive maintenance dashboard built with Databricks Lakehouse, Machine Learning, Streamlit, Plotly, and Gemini AI.</b>
</p>

---

## Overview

Predictive Maintenance Copilot is an end-to-end AI/ML project that helps identify machine risk levels, prioritize maintenance actions, and present insights through a real-time interactive dashboard.

The project combines:

- Databricks Lakehouse for data engineering
- Delta Lake with Bronze, Silver, and Gold architecture
- Random Forest model for predictive maintenance classification
- MLflow for experiment tracking
- Streamlit + Plotly for live dashboarding
- Gemini AI for grounded maintenance guidance

This project demonstrates how to build a practical AI application that connects data engineering, machine learning, visualization, and LLM-assisted decision support.

---

## Platforms Used

- **Platform:** Databricks
- **Frontend:** Streamlit
- **Programming Language:** Python
- **Database / Query Layer:** Databricks SQL
- **Storage Architecture:** Delta Lake
- **ML Tracking:** MLflow
- **Visualization:** Plotly
- **LLM Integration:** Gemini 3 Flash
- **Version Control:** GitHub

---

## Key Highlights

- Built a production-style predictive maintenance solution on Databricks Lakehouse
- Implemented Medallion Architecture: Bronze -> Silver -> Gold
- Trained a Random Forest model with **AUC = 0.954**
- Created a live Streamlit dashboard with multiple real-time charts
- Added KPI cards for machine volume, risk levels, and maintenance priorities
- Integrated Gemini AI to answer grounded maintenance questions
- Used filtered live Databricks data for dashboard insights
- Designed the system for portfolio, demo, and early production-style use

---

## Architecture

```mermaid
flowchart TD
    A[Raw Dataset] --> B[Bronze Layer<br/>Raw Ingestion]
    B --> C[Silver Layer<br/>Cleaned / Feature Engineered Data]
    C --> D[Gold Layer]
    D --> D1[gold_predictions]
    D --> D2[gold_machine_kpis]
    D --> D3[maintenance_priority]
    D --> E[ML Model<br/>Random Forest + MLflow]
    E --> F[Batch Inference]
    F --> G[Streamlit Dashboard + Gemini AI Advisor]

