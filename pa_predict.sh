#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────── #
#                  PA-ML Runner (ALL MODELS)                      #
#     CSV (fzf) → EDA → Benchmark (logistic, RF, XGB, SVM) → UI   #
#      Conda env from Setup/environment.yml + fzf (auto/fallback) #
# ──────────────────────────────────────────────────────────────── #
set -euo pipefail

# ===============  Styling  ===============
RED="\033[31m"; YELLOW="\033[33m"; GREEN="\033[32m"; BLUE="\033[34m"; CYAN="\033[36m"; NC="\033[0m"

# ===============  Defaults / Paths  ===============
OUTPUT_DIR="output"
FIGS_DIR="${OUTPUT_DIR}/figs"
CSV_REPORT="${OUTPUT_DIR}/model_comparison.csv"
HTML_REPORT="${OUTPUT_DIR}/report.html"
TEMPLATE_PATH="Scripts/Frontend/index.html"   # <- dashboard

# Conda env
ENV_NAME="pa-predict"
ENV_FILE="Setup/environment.yml"

# Controls
SELECTED_CSV=""
USE_FZF=true
AUTO_OPEN=true   # enable via --open-report

# ===============  Help  ===============
show_help() {
  cat <<'EOF'

🚀 PA-ML Pipeline Runner (ALL MODELS)

Usage:
  ./pa_predict.sh [options]

Options:
  --data PATH.csv         Use this CSV directly (skips interactive picker)
  --output-dir DIR        Output directory (default: output)
  --figs-dir DIR          Figures directory (default: output/figs)
  --template PATH.html    Report template (default: Scripts/Frontend/index.html)
  --no-fzf                Disable fzf; use numbered fallback
  --open-report           Try to open the HTML report when finished
  -h, --help              Show this help

Flow:
  1) Ensure conda exists; create env from Setup/environment.yml if needed
  2) Pick dataset (fzf over ./Data and ./data unless --data provided)
  3) EDA (Table 1, group tests, class balance, missingness, dists, corr)
  4) Benchmark ALL models (logistic, RF, XGB, SVM) + save plots
  5) Generate HTML dashboard with embedded plots & EDA

EOF
}

# ===============  CLI Parsing  ===============
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)         SELECTED_CSV="$2"; shift 2 ;;
    --output-dir)   OUTPUT_DIR="$2"; shift 2 ;;
    --figs-dir)     FIGS_DIR="$2"; shift 2 ;;
    --template)     TEMPLATE_PATH="$2"; shift 2 ;;
    --no-fzf)       USE_FZF=false; shift ;;
    --open-report)  AUTO_OPEN=true; shift ;;
    -h|--help)      show_help; exit 0 ;;
    *) echo -e "${RED}❌ Unknown option: $1${NC}"; show_help; exit 1 ;;
  esac
done

# Re-derive dependent paths if user changed DIRs
CSV_REPORT="${OUTPUT_DIR}/model_comparison.csv"
HTML_REPORT="${OUTPUT_DIR}/report.html"

# ===============  OS / Platform detection  ===============
OS="$(uname -s)"
IS_LINUX=false; IS_MAC=false; IS_WSL=false
case "$OS" in
  Linux*)  IS_LINUX=true ;;
  Darwin*) IS_MAC=true ;;
esac
if grep -qi microsoft /proc/version 2>/dev/null; then IS_WSL=true; fi
echo -e "${CYAN}🔎 System: ${OS}  (WSL: ${IS_WSL})${NC}"

# ===============  Utils  ===============
require_cmd() { command -v "$1" >/dev/null 2>&1 || { echo -e "${RED}❌ Missing command: $1${NC}"; return 1; }; }

open_html() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo -e "${YELLOW}⚠️  Report not found: $f${NC}"
    return
  fi
  if $IS_WSL && command -v wslview >/dev/null 2>&1; then wslview "$f" >/dev/null 2>&1 &
  elif $IS_MAC && command -v open >/dev/null 2>&1; then open "$f" >/dev/null 2>&1 &
  elif $IS_LINUX && command -v xdg-open >/dev/null 2>&1; then xdg-open "$f" >/dev/null 2>&1 &
  else echo -e "${YELLOW}ℹ️  Open manually: file://$f${NC}"; fi
}

install_fzf_if_missing() {
  if command -v fzf >/dev/null 2>&1; then return 0; fi
  echo -e "${YELLOW}⚙️  fzf not found — attempting installation...${NC}"
  if $IS_MAC && command -v brew >/dev/null 2>&1; then
    brew list fzf >/dev/null 2>&1 || brew install fzf
    "$(brew --prefix)/opt/fzf/install" --no-bash --no-fish --no-key-bindings --no-completion >/dev/null 2>&1 || true
  elif $IS_LINUX && command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -qq || true
    sudo apt-get install -y fzf || true
  fi
  # source fallback
  if ! command -v fzf >/dev/null 2>&1; then
    tmpdir="$(mktemp -d)"; git clone --depth 1 https://github.com/junegunn/fzf.git "$tmpdir/fzf" >/dev/null 2>&1 || true
    bash "$tmpdir/fzf/install" --bin --no-update-rc --no-key-bindings --no-completion >/dev/null 2>&1 || true
    mkdir -p "$HOME/.local/bin"
    if [[ -f "$tmpdir/fzf/bin/fzf" ]]; then
      cp "$tmpdir/fzf/bin/fzf" "$HOME/.local/bin/fzf"
      export PATH="$HOME/.local/bin:$PATH"
    fi
    rm -rf "$tmpdir"
  fi
  if command -v fzf >/dev/null 2>&1; then
    echo -e "${GREEN}✅ fzf installed.${NC}"
  else
    echo -e "${YELLOW}⚠️  Could not auto-install fzf. Fallback UI will be used.${NC}"
  fi
}

# ===============  Conda env management via conda run  ===============
ensure_conda_and_env() {
  if ! command -v conda >/dev/null 2>&1; then
    echo -e "${RED}❌ Conda not found in PATH.${NC}"
    echo -e "  Install Miniconda, then re-run:"
    if $IS_MAC; then
      echo "  https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-$(uname -m).sh"
    else
      echo "  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    fi
    exit 1
  fi
  # Enable conda shell funcs; needed for `conda env list`
  source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true

  if [[ ! -f "$ENV_FILE" ]]; then
    echo -e "${RED}❌ Missing environment file: $ENV_FILE${NC}"
    exit 1
  fi

  if ! conda env list | grep -qw "$ENV_NAME"; then
    echo -e "${YELLOW}🔧 Creating conda env '$ENV_NAME' from $ENV_FILE...${NC}"
    conda env create -f "$ENV_FILE" -n "$ENV_NAME" || { echo -e "${RED}❌ Env creation failed.${NC}"; exit 1; }
    echo -e "${GREEN}✅ Env created.${NC}"
  fi

  # Sanity: ensure conda run works
  if ! conda run -n "$ENV_NAME" python -c "import sys; print(sys.version)" >/dev/null 2>&1; then
    echo -e "${YELLOW}ℹ️  Updating conda to enable 'conda run'...${NC}"
    conda update -n base -y conda
  fi
  if ! conda run -n "$ENV_NAME" python -c "import sys; print(sys.version)" >/dev/null 2>&1; then
    echo -e "${RED}❌ 'conda run' failed for env '$ENV_NAME'.${NC}"
    exit 1
  fi
}

# ===============  Preflight (env-level) using conda run  ===============
preflight_env() {
  echo -e "${CYAN}🔎 Preflight checks (inside env via conda run)...${NC}"
  # Python version
  conda run -n "$ENV_NAME" python - <<'PY'
import sys
maj, min = sys.version_info[:2]
assert (maj, min) >= (3,9), f"Python >=3.9 required, found {maj}.{min}"
PY

  # Required packages (include scipy for group tests)
  conda run -n "$ENV_NAME" python - <<'PY'
import importlib, sys
pkgs = ["numpy","pandas","sklearn","xgboost","shap","matplotlib","seaborn","jinja2","joblib","scipy"]
missing = [p for p in pkgs if importlib.util.find_spec(p) is None]
if missing:
    raise SystemExit("Missing: " + ", ".join(missing))
PY
  echo -e "${GREEN}✅ Python deps OK${NC}"

  # Required project files
  for f in "Scripts/benchmark_models.py" "Scripts/generate_report.py" "Scripts/eda_profile.py" "$TEMPLATE_PATH"; do
    [[ -f "$f" ]] || { echo -e "${RED}❌ Missing file: $f${NC}"; exit 1; }
  done
  echo -e "${GREEN}✅ Script/template files OK${NC}"

  # Disk space & writability
  FREE_KB=$(df -Pk "$PWD" | awk 'NR==2{print $4}')
  FREE_MB=$((FREE_KB/1024))
  NEED_MB=200
  if [[ -n "${FREE_MB:-}" && "$FREE_MB" -lt "$NEED_MB" ]]; then
    echo -e "${RED}❌ Not enough disk space: ${FREE_MB}MB free (< ${NEED_MB}MB).${NC}"; exit 1
  fi
  mkdir -p "$OUTPUT_DIR" "$FIGS_DIR" 2>/dev/null || { echo -e "${RED}❌ Cannot create output dirs${NC}"; exit 1; }
  touch "$OUTPUT_DIR/.write_test" 2>/dev/null || { echo -e "${RED}❌ No write permission in $OUTPUT_DIR${NC}"; exit 1; }
  rm -f "$OUTPUT_DIR/.write_test"
  echo -e "${GREEN}✅ Output directory writable${NC}"

  # fzf note
  if $USE_FZF && ! command -v fzf >/dev/null 2>&1; then
    echo -e "${YELLOW}ℹ️  fzf not available — using fallback selection UI.${NC}"
  fi

  # Safe plotting + tame BLAS/OMP for child processes
  export MPLBACKEND=Agg
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
  export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
  export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
  export KMP_DUPLICATE_LIB_OK=TRUE
}

# ===============  CSV Selector (fzf or fallback)  ===============
select_csv() {
  local prompt="${1:-CSV > }"; shift || true
  local roots=("$@"); [[ ${#roots[@]} -eq 0 ]] && roots=("Data" "data")
  local files=() r
  for r in "${roots[@]}"; do
    [[ -d "$r" ]] || continue
    while IFS= read -r f; do files+=("$f"); done < <(find "$r" -type f -iname "*.csv" 2>/dev/null | sort)
  done
  if [[ ${#files[@]} -eq 0 ]]; then
    echo -e "${RED}❌ No CSV files found in: ${roots[*]}${NC}" 1>&2; return 1
  fi
  local file=""
  if $USE_FZF && command -v fzf >/dev/null 2>&1; then
    export TERM=${TERM:-xterm-256color}
    file="$(printf '%s\n' "${files[@]}" | fzf --prompt="$prompt " --height=15 --border --reverse)"
    [[ -z "$file" ]] && { echo -e "${RED}❌ No file selected! Exiting.${NC}" 1>&2; return 1; }
  else
    echo -e "${CYAN}Available CSV files:${NC}"
    local i=1; for f in "${files[@]}"; do printf " %2d) %s\n" "$i" "$f"; ((i++)); done
    echo
    local sel; read -rp "$prompt (number or path): " sel
    if [[ -f "$sel" ]]; then
      file="$sel"
    elif [[ "$sel" =~ ^[0-9]+$ ]] && (( sel>=1 && sel<=${#files[@]} )); then
      file="${files[sel-1]}"
    else
      echo -e "${RED}❌ Invalid choice: ${sel}${NC}" 1>&2; return 1
    fi
  fi
  [[ "$file" == "$PWD"* ]] && file="."${file#$PWD}
  printf '%s\n' "$file"
}

# ===============  Go time  ===============
install_fzf_if_missing
ensure_conda_and_env
preflight_env

# Dataset selection
if [[ -z "$SELECTED_CSV" ]]; then
  SELECTED_CSV="$(select_csv '📂 CSV >' 'Data' 'data')" || exit 1
fi
echo -e "📂 Using dataset: ${GREEN}$SELECTED_CSV${NC}"

# -------- EDA (inside env) --------
echo -e "${CYAN}🧪 Running dataset EDA...${NC}"
conda run -n "$ENV_NAME" python Scripts/eda_profile.py \
  --data "$SELECTED_CSV" \
  --target "Diagnosed_PA" \
  --figs_dir "$FIGS_DIR" \
  --output_json "${OUTPUT_DIR}/eda_summary.json" \
  --table1_csv "${OUTPUT_DIR}/eda_table1.csv" \
  --tests_csv "${OUTPUT_DIR}/eda_group_tests.csv"

# -------- Benchmark ALL models (inside env) --------
echo -e "${CYAN}📈 Benchmarking ALL models (logistic, RF, XGB, SVM)...${NC}"
conda run -n "$ENV_NAME" python Scripts/benchmark_models.py \
  --data "$SELECTED_CSV" \
  --output_csv "$CSV_REPORT" \
  --figs_dir "$FIGS_DIR"

# -------- Generate report (inside env) --------
echo -e "${CYAN}📝 Building HTML report...${NC}"
conda run -n "$ENV_NAME" python Scripts/generate_report.py \
  --csv "$CSV_REPORT" \
  --output "$HTML_REPORT" \
  --template "$TEMPLATE_PATH" \
  --figs_dir "$FIGS_DIR"

# Done
echo -e "${GREEN}✅ Pipeline completed successfully.${NC}"
echo -e "   📊 Metrics CSV : ${CSV_REPORT}"
echo -e "   📁 Figures     : ${FIGS_DIR}"
echo -e "   📄 Report      : ${HTML_REPORT}"
echo -e "   📄 EDA Table 1 : ${OUTPUT_DIR}/eda_table1.csv"
echo -e "   📄 EDA Tests   : ${OUTPUT_DIR}/eda_group_tests.csv"

if $AUTO_OPEN; then
  echo -e "${CYAN}🌐 Opening report...${NC}"
  open_html "$HTML_REPORT"
fi
