# 🧬 Pernicious Anaemia Diagnosis Machine Learning Pipeline

This repository provides an end-to-end machine learning pipeline to simulate patient data, train classification models for **Pernicious Anaemia (PA)** detection, benchmark multiple models, and generate interactive HTML reports — all orchestrated with **Nextflow** and optionally containerised with **Docker**.

** FROM RAW CLINICAL BIOMARKER DATA **

---

## 📁 Project Structure

---

## 🚀 Features

- 📊 Synthetic dataset generation
- 🤖 Logistic Regression, Random Forest, and XGBoost model training
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
```bash
cd PA-Predict
```

### 2. Ensure Nextflow is Available

```bash
$($(find / -name nextflow -type f 2>/dev/null | head -n 1))
```

### 3. Run the Pipeline

```bash
chmod +x run_pipeline.sh && ./run_pipeline.sh
```

---
