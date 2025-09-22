# 🧬 Pernicious Anaemia Diagnosis Machine Learning Pipeline

This repository provides an end-to-end machine learning pipeline to train classification models for **Pernicious Anaemia (PA)** detection, benchmark multiple models, and generate an interactive HTML report.

---

## 📁 Project Structure

---

## 🚀 Features

- 🤖 Logistic Regression, Random Forest, Support Vector Machine and XGBoost model training
- 🧠 SHAP analysis for explainability 
- 📉 Visual reports: Confusion Matrix, ROC, PR curves, feature importance, hyperparameter sweeps
- 📄 Auto-generated HTML reports
- ⚙️ Executable via Bash (virtual environment -> Conda)

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

### 3. Every time you want to run PA-predict from a new terminal

```bash
bash -lc 'repo=$(find ~ -type d -name "PA-Predict" 2>/dev/null | head -n1); \
  if [ -z "$repo" ]; then echo "PA-Predict not found. Clone it first."; exit 1; fi; \
  cd "$repo"; chmod +x pa_predict.sh; ./pa_predict.sh --open-report'
```



