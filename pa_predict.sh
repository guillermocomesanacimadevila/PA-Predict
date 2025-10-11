#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────── #
#                  PA-ML Runner (ALL MODELS)                      #
#   Zero-dep bootstrap: system Conda or local Miniconda + fzf     #
#   CSV → EDA → Benchmark (logistic, RF, XGB, SVM) → HTML report  #
# ──────────────────────────────────────────────────────────────── #
set -euo pipefail

# ===============  Styling  ===============
RED="\033[31m"; YELLOW="\033[33m"; GREEN="\033[32m"; BLUE="\033[34m"; CYAN="\033[36m"; NC="\033[0m"

# ===============  Defaults / Paths  ===============
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${REPO_ROOT}/output"
FIGS_DIR="${OUTPUT_DIR}/figs"
CSV_REPORT="${OUTPUT_DIR}/model_comparison.csv"
HTML_REPORT="${OUTPUT_DIR}/report.html"
TEMPLATE_PATH="${REPO_ROOT}/Scripts/Frontend/index.html"

# Local tool cache (no sudo needed)
LOCAL_BIN="${REPO_ROOT}/.bin"
LOCAL_CONDA="${REPO_ROOT}/.miniconda"
LOCAL_CONDA_SH="${LOCAL_CONDA}/etc/profile.d/conda.sh"

# Conda env
ENV_NAME="pa-predict"
ENV_FILE="${REPO_ROOT}/Setup/environment.yml"

# Controls
SELECTED_CSV=""
AUTO_OPEN=true
USE_FZF=true

# ===============  Help  ===============
show_help() {
  cat <<'EOF'

🚀 PA-ML Pipeline Runner (ALL MODELS) — Zero-dep bootstrap

Usage:
  ./pa_predict.sh [options]

Options:
  --data PATH.csv         Use this CSV directly (skips picker)
  --output-dir DIR        Output directory (default: ./output)
  --figs-dir DIR          Figures directory (default: ./output/figs)
  --template PATH.html    Report template (default: Scripts/Frontend/index.html)
  --no-fzf                Disable fzf; use numbered fallback
  --open-report           Try to open the HTML report when finished
  -h, --help              Show this help

Flow:
  0) Prefer system Conda; otherwise bootstrap local Miniconda in ./.miniconda
  1) Create conda env from Setup/environment.yml if missing
  2) Pick dataset (portable fzf in ./.bin or numbered fallback)
  3) EDA (stats, balance, missingness, histograms, correlations)
  4) Benchmark ALL models (logistic, RF, XGB, SVM) + save plots
  5) Generate HTML dashboard (embedded images)

EOF
}

# ===============  CLI Parsing  ===============
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data)         SELECTED_CSV="${2:?}"; shift 2 ;;
    --output-dir)   OUTPUT_DIR="${2:?}"; shift 2 ;;
    --figs-dir)     FIGS_DIR="${2:?}"; shift 2 ;;
    --template)     TEMPLATE_PATH="${2:?}"; shift 2 ;;
    --no-fzf)       USE_FZF=false; shift ;;
    --open-report)  AUTO_OPEN=true; shift ;;
    -h|--help)      show_help; exit 0 ;;
    *) echo -e "${RED}❌ Unknown option: $1${NC}"; show_help; exit 1 ;;
  esac
done
CSV_REPORT="${OUTPUT_DIR}/model_comparison.csv"
HTML_REPORT="${OUTPUT_DIR}/report.html"

# ===============  OS / Arch detection  ===============
OS="$(uname -s)"
ARCH="$(uname -m)"
IS_LINUX=false; IS_MAC=false; IS_WSL=false
case "$OS" in
  Linux*)  IS_LINUX=true ;;
  Darwin*) IS_MAC=true ;;
esac
if [[ -f /proc/version ]] && grep -qi microsoft /proc/version 2>/dev/null; then IS_WSL=true; fi
echo -e "${CYAN}🔎 System: ${OS} (${ARCH})  (WSL: ${IS_WSL})${NC}"

# ===============  Utils  ===============
mkdir -p "$LOCAL_BIN"
export PATH="${LOCAL_BIN}:$PATH"

# --- NEW: minimal core tools sanity check (no sudo; just helpful errors) ---
ensure_core_tools() {
  local missing=()
  for cmd in uname grep awk sed find tar gzip; do
    command -v "$cmd" >/dev/null 2>&1 || missing+=("$cmd")
  done
  if ((${#missing[@]})); then
    echo -e "${RED}❌ Missing required system utilities:${NC} ${missing[*]}"
    echo -e "   Please install them via your package manager and re-run."
    exit 1
  fi
}
ensure_core_tools

# --- UPDATED: download with curl → wget → Python (urllib) fallback ---
download() {
  # usage: download URL DEST
  local url="$1" dest="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -L --retry 3 --fail -o "$dest" "$url" && return 0
  fi
  if command -v wget >/dev/null 2>&1; then
    wget -q -O "$dest" "$url" && return 0
  fi
  # Python fallback (no internet libs beyond stdlib)
  if command -v python3 >/dev/null 2>&1; then
    python3 - "$url" "$dest" <<'PY'
import sys, urllib.request
url, dest = sys.argv[1], sys.argv[2]
urllib.request.urlretrieve(url, dest)
PY
    return 0
  elif command -v python >/dev/null 2>&1; then
    python - "$url" "$dest" <<'PY'
import sys, urllib.request
url, dest = sys.argv[1], sys.argv[2]
urllib.request.urlretrieve(url, dest)
PY
    return 0
  fi
  echo -e "${RED}❌ Need curl, wget, or python to download: $url${NC}"
  exit 1
}

open_html() {
  local f="$1"
  [[ -f "$f" ]] || { echo -e "${YELLOW}⚠️  Report not found: $f${NC}"; return; }
  if $IS_WSL && command -v wslview >/dev/null 2>&1; then wslview "$f" >/dev/null 2>&1 &
  elif $IS_MAC && command -v open >/dev/null 2>&1; then open "$f" >/dev/null 2>&1 &
  elif $IS_LINUX && command -v xdg-open >/dev/null 2>&1; then xdg-open "$f" >/dev/null 2>&1 &
  else echo -e "${YELLOW}ℹ️  Open manually: file://$f${NC}"; fi
}

# --- helpers: Homebrew + Linux package managers -----------------
ensure_homebrew() {
  # Only needed on macOS; installs Homebrew non-interactively if missing.
  if command -v brew >/dev/null 2>&1; then
    return 0
  fi
  echo -e "${YELLOW}🍺 Homebrew not found. Installing Homebrew (non-interactive)...${NC}"
  if ! command -v curl >/dev/null 2>&1; then
    echo -e "${RED}❌ curl is required to install Homebrew.${NC}"
    return 1
  fi
  NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || {
    echo -e "${YELLOW}⚠️  Homebrew install script failed.${NC}"
    return 1
  }
  # Add brew to PATH for current shell
  if [[ -x "/opt/homebrew/bin/brew" ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  elif [[ -x "/usr/local/bin/brew" ]]; then
    eval "$(/usr/local/bin/brew shellenv)"
  fi
  if ! command -v brew >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Homebrew installation completed but 'brew' not on PATH yet. You may need to restart your shell.${NC}"
    return 1
  fi
  return 0
}

linux_install_fzf_pkg() {
  # Attempts to install fzf via the distro's package manager, using sudo when needed.
  local cmd=""
  if command -v apt-get >/dev/null 2>&1; then
    cmd="apt-get update -y && apt-get install -y fzf"
  elif command -v dnf >/dev/null 2>&1; then
    cmd="dnf install -y fzf"
  elif command -v yum >/dev/null 2>&1; then
    cmd="yum install -y fzf"
  elif command -v pacman >/dev/null 2>&1; then
    cmd="pacman -Sy --noconfirm fzf"
  elif command -v zypper >/dev/null 2>&1; then
    cmd="zypper -n install fzf"
  elif command -v apk >/dev/null 2>&1; then
    cmd="apk add --no-cache fzf"
  fi

  if [[ -z "$cmd" ]]; then
    return 1
  fi

  echo -e "${YELLOW}🧰 Installing fzf via package manager...${NC}"
  if [[ $EUID -ne 0 ]] && command -v sudo >/dev/null 2>&1; then
    sudo bash -lc "$cmd" || return 1
  else
    bash -lc "$cmd" || return 1
  fi

  command -v fzf >/dev/null 2>&1
}

# ===============  Portable fzf (no sudo)  ===============
ensure_portable_fzf() {
  if ! $USE_FZF; then return 0; fi
  if command -v fzf >/dev/null 2>&1; then return 0; fi

  # Try system installers first
  if $IS_MAC; then
    echo -e "${YELLOW}🔎 fzf not found. Trying Homebrew on macOS...${NC}"
    if ensure_homebrew && brew install fzf; then
      echo -e "${GREEN}✅ fzf installed via Homebrew.${NC}"
      return 0
    else
      echo -e "${YELLOW}⚠️  Homebrew path failed; falling back to portable fzf.${NC}"
    fi
  elif $IS_LINUX; then
    if linux_install_fzf_pkg; then
      echo -e "${GREEN}✅ fzf installed via system package manager.${NC}"
      return 0
    else
      echo -e "${YELLOW}⚠️  Package manager path failed; falling back to portable fzf.${NC}"
    fi
  fi

  echo -e "${YELLOW}⚙️  Installing portable fzf to ${LOCAL_BIN} ...${NC}"
  local fzf_ver="0.53.0"  # known good
  local pkg=""
  if $IS_MAC; then
    if [[ "$ARCH" == "arm64" ]]; then
      pkg="fzf-${fzf_ver}-darwin_arm64.tar.gz"
    else
      pkg="fzf-${fzf_ver}-darwin_amd64.tar.gz"
    fi
  elif $IS_LINUX; then
    case "$ARCH" in
      x86_64|amd64) pkg="fzf-${fzf_ver}-linux_amd64.tar.gz" ;;
      aarch64|arm64) pkg="fzf-${fzf_ver}-linux_arm64.tar.gz" ;;
      armv7l) pkg="fzf-${fzf_ver}-linux_armv7.tar.gz" ;;
      *) echo -e "${YELLOW}⚠️  Unsupported arch for fzf binary (${ARCH}). Fallback UI will be used.${NC}"; return 0 ;;
    esac
  else
    echo -e "${YELLOW}⚠️  Unsupported OS for fzf binary. Fallback UI will be used.${NC}"; return 0
  fi

  local url="https://github.com/junegunn/fzf/releases/download/v${fzf_ver}/${pkg}"
  local tmp; tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
  download "$url" "${tmp}/${pkg}" || { echo -e "${YELLOW}⚠️  fzf download failed; using fallback UI.${NC}"; return 0; }
  tar -xzf "${tmp}/${pkg}" -C "$tmp" >/dev/null 2>&1 || true
  if [[ -f "${tmp}/fzf" ]]; then
    mv "${tmp}/fzf" "${LOCAL_BIN}/fzf"
    chmod +x "${LOCAL_BIN}/fzf"
    echo -e "${GREEN}✅ fzf installed locally.${NC}"
  else
    echo -e "${YELLOW}⚠️  fzf install failed; using fallback UI.${NC}"
  fi
}

# ===============  Conda: prefer system, fallback to local  ===============
source_conda_shell() {
  # 1) If CONDA_EXE is set & shell hook is available, use it.
  if [[ -n "${CONDA_EXE:-}" ]]; then
    local base; base="$("$CONDA_EXE" info --base 2>/dev/null || true)"
    if [[ -n "$base" && -f "$base/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "$base/etc/profile.d/conda.sh"
      return 0
    fi
  fi
  # 2) If 'conda' is in PATH, try its base.
  if command -v conda >/dev/null 2>&1; then
    local base; base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "$base" && -f "$base/etc/profile.d/conda.sh" ]]; then
      # shellcheck disable=SC1090
      source "$base/etc/profile.d/conda.sh"
      return 0
    fi
  fi
  return 1
}

install_local_miniconda_if_needed() {
  # If local conda hook exists, just source it.
  if [[ -f "$LOCAL_CONDA_SH" ]]; then
    # shellcheck disable=SC1090
    source "$LOCAL_CONDA_SH"
    return 0
  fi

  # If directory exists but conda.sh missing, try updating in place (-u).
  if [[ -d "$LOCAL_CONDA" && ! -f "$LOCAL_CONDA_SH" ]]; then
    echo -e "${YELLOW}⚙️  Repairing local Miniconda in ${LOCAL_CONDA} ...${NC}"
    _install_miniconda "-u"
    # shellcheck disable=SC1090
    source "$LOCAL_CONDA_SH"
    return 0
  fi

  echo -e "${YELLOW}🔧 Installing local Miniconda into ${LOCAL_CONDA} (no sudo)...${NC}"
  _install_miniconda ""
  # shellcheck disable=SC1090
  source "$LOCAL_CONDA_SH"
  echo -e "${GREEN}✅ Local Miniconda ready.${NC}"
}

_install_miniconda() {
  local update_flag="$1"  # "" or "-u"
  mkdir -p "$LOCAL_CONDA"
  local installer="" url=""

  if $IS_MAC; then
    if [[ "$ARCH" == "arm64" ]]; then
      installer="Miniconda3-latest-MacOSX-arm64.sh"
    else
      installer="Miniconda3-latest-MacOSX-x86_64.sh"
    fi
    url="https://repo.anaconda.com/miniconda/${installer}"
  elif $IS_LINUX; then
    case "$ARCH" in
      x86_64|amd64) installer="Miniconda3-latest-Linux-x86_64.sh" ;;
      aarch64|arm64) installer="Miniconda3-latest-Linux-aarch64.sh" ;;
      *) echo -e "${RED}❌ Unsupported Linux arch for Miniconda: ${ARCH}${NC}"; exit 1 ;;
    esac
    url="https://repo.anaconda.com/miniconda/${installer}"
  else
    echo -e "${RED}❌ Unsupported OS for Miniconda bootstrap.${NC}"
    exit 1
  fi

  local tmp; tmp="$(mktemp -d)"
  download "$url" "${tmp}/${installer}"
  bash "${tmp}/${installer}" -b -p "$LOCAL_CONDA" $update_flag
  rm -rf "$tmp"

  if [[ ! -f "$LOCAL_CONDA_SH" ]]; then
    echo -e "${RED}❌ Miniconda installation failed.${NC}"
    exit 1
  fi
}

resolve_conda() {
  if source_conda_shell; then
    echo -e "${GREEN}✅ Using system Conda at: $(conda info --base)${NC}"
    return 0
  fi
  install_local_miniconda_if_needed
  echo -e "${GREEN}✅ Using local Conda at: ${LOCAL_CONDA}${NC}"
}

# ===============  Conda env creation/use  ===============
ensure_conda_env() {
  if [[ ! -f "$ENV_FILE" ]]; then
    echo -e "${RED}❌ Missing environment file: $ENV_FILE${NC}"
    exit 1
  fi

  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo -e "${YELLOW}🔧 Creating conda env '${ENV_NAME}' from $ENV_FILE ...${NC}"
    conda env create -n "$ENV_NAME" -f "$ENV_FILE"
    echo -e "${GREEN}✅ Env created.${NC}"
  fi

  # Verify conda run works
  if ! conda run -n "$ENV_NAME" python -c "import sys; print(sys.version)" >/dev/null 2>&1; then
    echo -e "${YELLOW}ℹ️  Updating conda base to enable 'conda run'...${NC}"
    conda update -n base -y conda || true
  fi
  if ! conda run -n "$ENV_NAME" python -c "import sys; print(sys.version)" >/dev/null 2>&1; then
    echo -e "${RED}❌ 'conda run' failed for env '${ENV_NAME}'.${NC}"
    exit 1
  fi
}

# ===============  Preflight (inside env)  ===============
preflight_env() {
  echo -e "${CYAN}🔎 Preflight checks (inside env)...${NC}"

  conda run -n "$ENV_NAME" python - <<'PY'
import sys, importlib
maj, min = sys.version_info[:2]
assert (maj, min) >= (3,9), f"Python >=3.9 required, found {maj}.{min}"
pkgs = ["numpy","pandas","sklearn","xgboost","shap","matplotlib","seaborn","jinja2","joblib","scipy"]
missing = [p for p in pkgs if importlib.util.find_spec(p) is None]
if missing:
    raise SystemExit("Missing packages: " + ", ".join(missing))
PY
  echo -e "${GREEN}✅ Python deps OK${NC}"

  # Project files exist?
  for f in \
    "${REPO_ROOT}/Scripts/benchmark_models.py" \
    "${REPO_ROOT}/Scripts/generate_report.py" \
    "${REPO_ROOT}/Scripts/eda_profile.py" \
    "$TEMPLATE_PATH"
  do
    [[ -f "$f" ]] || { echo -e "${RED}❌ Missing file: $f${NC}"; exit 1; }
  done
  echo -e "${GREEN}✅ Script/template files OK${NC}"

  # Output area
  mkdir -p "$OUTPUT_DIR" "$FIGS_DIR"
  touch "${OUTPUT_DIR}/.write_test" && rm -f "${OUTPUT_DIR}/.write_test" || {
    echo -e "${RED}❌ No write access to ${OUTPUT_DIR}${NC}"; exit 1; }

  echo -e "${GREEN}✅ Output directory writable${NC}"

  # Matplotlib headless + keep BLAS tame
  export MPLBACKEND=Agg
  export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
  export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
  export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
  export KMP_DUPLICATE_LIB_OK=TRUE
}

# ===============  CSV Selector  ===============
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
    [[ -z "$file" ]] && { echo -e "${RED}❌ No file selected! Exiting.${NC}"; return 1; }
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
      echo -e "${RED}❌ Invalid choice: ${sel}${NC}"; return 1
    fi
  fi
  [[ "$file" == "$PWD"* ]] && file="."${file#$PWD}
  printf '%s\n' "$file"
}

# ===============  Bootstrap & Run  ===============
ensure_portable_fzf
resolve_conda
ensure_conda_env
preflight_env

# Dataset selection
if [[ -z "$SELECTED_CSV" ]]; then
  SELECTED_CSV="$(select_csv '📂 CSV >' "${REPO_ROOT}/Data" "${REPO_ROOT}/data")" || exit 1
fi
echo -e "📂 Using dataset: ${GREEN}${SELECTED_CSV}${NC}"

# -------- EDA --------
echo -e "${CYAN}🧪 Running dataset EDA...${NC}"
conda run -n "$ENV_NAME" python "${REPO_ROOT}/Scripts/eda_profile.py" \
  --data "$SELECTED_CSV" \
  --target "Diagnosed_PA" \
  --figs_dir "$FIGS_DIR" \
  --output_json "${OUTPUT_DIR}/eda_summary.json" \
  --table1_csv "${OUTPUT_DIR}/eda_table1.csv" \
  --tests_csv "${OUTPUT_DIR}/eda_group_tests.csv"

# -------- Benchmark ALL models --------
echo -e "${CYAN}📈 Benchmarking ALL models (logistic, RF, XGB, SVM)...${NC}"
conda run -n "$ENV_NAME" python "${REPO_ROOT}/Scripts/benchmark_models.py" \
  --data "$SELECTED_CSV" \
  --output_csv "$CSV_REPORT" \
  --figs_dir "$FIGS_DIR" \
  --models_dir "${OUTPUT_DIR}/models"

# -------- Generate report --------
echo -e "${CYAN}📝 Building HTML report...${NC}"
conda run -n "$ENV_NAME" python "${REPO_ROOT}/Scripts/generate_report.py" \
  --csv "$CSV_REPORT" \
  --output "$HTML_REPORT" \
  --template "$TEMPLATE_PATH" \
  --figs_dir "$FIGS_DIR"

# -------- Done --------
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
