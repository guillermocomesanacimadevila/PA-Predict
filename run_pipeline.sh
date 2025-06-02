#!/bin/bash

set -euo pipefail

# ---------------------------------------------
# User-Facing Pipeline Runner for PA ML Project
# ---------------------------------------------

show_help() {
  echo ""
  echo "🚀 PA-ML Pipeline Runner"
  echo ""
  echo "Usage:"
  echo "  ./run_pipeline.sh [--samples N] [--seed N] [--model TYPE] [--docker] [--nextflow]"
  echo ""
  echo "Options:"
  echo "  --samples N       Number of synthetic samples to generate (default: 1000)"
  echo "  --seed N          Random seed for reproducibility (default: 42)"
  echo "  --model TYPE      Model type: 'logistic', 'rf', 'xgb', or 'svm' (default: rf)"
  echo "  --docker          Use Docker (must build container first)"
  echo "  --nextflow        Run pipeline using Nextflow"
  echo "  -h, --help        Show this help message"
  echo ""
  echo "Examples:"
  echo "  ./run_pipeline.sh"
  echo "  ./run_pipeline.sh --samples 2000 --model svm"
  echo "  ./run_pipeline.sh --docker"
  echo ""
}

# ----------------------------
# Defaults
# ----------------------------

SAMPLES=1000
SEED=42
MODEL_TYPE="rf"
USE_DOCKER=false
USE_NEXTFLOW=false

# ----------------------------
# Parse arguments
# ----------------------------

while [[ $# -gt 0 ]]; do
  case $1 in
    --samples)
      SAMPLES="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --model)
      MODEL_TYPE="$2"
      shift 2
      ;;
    --docker)
      USE_DOCKER=true
      shift
      ;;
    --nextflow)
      USE_NEXTFLOW=true
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "❌ Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# ----------------------------
# Setup Paths
# ----------------------------

DATA_DIR="data"
OUTPUT_DIR="output"
DATA_PATH="${DATA_DIR}/simulated_pa_data.csv"
MODEL_PATH="${OUTPUT_DIR}/model.pkl"
FIGS_DIR="${OUTPUT_DIR}/figs"
CSV_REPORT="${OUTPUT_DIR}/model_comparison.csv"
HTML_REPORT="${OUTPUT_DIR}/report.html"
TEMPLATE_PATH="Scripts/report_template.html"

# ----------------------------
# Mode: Docker
# ----------------------------

if $USE_DOCKER; then
  echo "🐳 Running pipeline via Docker..."
  docker run --rm -v "$PWD":/app -w /app pa-ml-pipeline \
    bash -c "
      mkdir -p $DATA_DIR $OUTPUT_DIR $FIGS_DIR &&
      python Scripts/generate_pa_data.py \
        --samples $SAMPLES \
        --seed $SEED \
        --output $DATA_PATH &&
      python Scripts/pa_model_trainer.py \
        --data $DATA_PATH \
        --model $MODEL_TYPE \
        --savefigs \
        --output_model $MODEL_PATH \
        --output_figs_dir $FIGS_DIR &&
      python Scripts/benchmark_models.py \
        --data $DATA_PATH \
        --output_csv $CSV_REPORT &&
      python Scripts/generate_report.py \
        --csv $CSV_REPORT \
        --output $HTML_REPORT \
        --template $TEMPLATE_PATH \
        --figs_dir $FIGS_DIR
    "
  exit 0
fi

# ----------------------------
# Mode: Nextflow
# ----------------------------

if $USE_NEXTFLOW; then
  echo "⚙️  Running pipeline via Nextflow..."
  nextflow run main.nf \
    --samples "$SAMPLES" \
    --seed "$SEED" \
    --model_type "$MODEL_TYPE"
  exit 0
fi

# ----------------------------
# Local Native Execution
# ----------------------------

echo "🧼 Preparing directories..."
mkdir -p "$DATA_DIR" "$OUTPUT_DIR" "$FIGS_DIR"

echo "📊 Generating synthetic dataset..."
python Scripts/generate_pa_data.py \
  --samples "$SAMPLES" \
  --seed "$SEED" \
  --output "$DATA_PATH"

echo "🤖 Training ML model ($MODEL_TYPE)..."
python Scripts/pa_model_trainer.py \
  --data "$DATA_PATH" \
  --model "$MODEL_TYPE" \
  --savefigs \
  --output_model "$MODEL_PATH" \
  --output_figs_dir "$FIGS_DIR"

echo "📈 Benchmarking all models..."
python Scripts/benchmark_models.py \
  --data "$DATA_PATH" \
  --output_csv "$CSV_REPORT"

echo "📝 Generating HTML report with embedded visualizations..."
python Scripts/generate_report.py \
  --csv "$CSV_REPORT" \
  --output "$HTML_REPORT" \
  --template "$TEMPLATE_PATH" \
  --figs_dir "$FIGS_DIR"

# ----------------------------
# Completion
# ----------------------------

echo "✅ Pipeline completed successfully."
echo "🧠 Trained model saved to: $MODEL_PATH"
echo "📊 Model comparison CSV: $CSV_REPORT"
echo "📄 HTML report generated: $HTML_REPORT"
echo "📁 Visualizations saved in: $FIGS_DIR"
