# 🧬 Pernicious Anaemia Diagnosis Machine Learning Pipeline

This repository provides an end-to-end machine learning pipeline to simulate patient data, train classification models for **Pernicious Anaemia (PA)** detection, benchmark multiple models, and generate interactive HTML reports — all orchestrated with **Nextflow** and optionally containerised with **Docker**.

** FROM RAW CLINICAL BIOMARKER DATA **

---

## 📁 Project Structure

---

## 🚀 Features

- 📊 Synthetic dataset generation
- 🤖 Logistic Regression, Random Forest, Support Vector Machine and XGBoost model training
- 🧠 SHAP analysis for explainability
- 📉 Visual reports: Confusion Matrix, ROC, PR curves, feature importance
- 📄 Auto-generated HTML reports
- ⚙️ Executable via Bash, Docker, or Nextflow

---

## 🚀 Run the Pipeline 

Follow the steps below to set up and run the pipeline.

### 1. Clone the Repository

```bash
git clone https://github.com/guillermocomesanacimadevila/PA-Predict.git
```

### 2. Run PA-predict

```bash
cd PA-Predict/ && chmod +x pa_predict.sh && ./pa_predict.sh
```
