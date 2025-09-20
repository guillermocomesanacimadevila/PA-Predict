#!/usr/bin/env python
"""
Quick EDA for PA-Predict (publishable aesthetics)
- Figures: class balance, missingness, correlations, numeric dists (+ by class)
- Table 1: descriptive stats (overall + by class)
- Group tests: Welch's t-test (numeric), Chi-square (categorical) vs target
- Summary JSON for dashboard

Outputs (defaults):
  output/figs/eda_*.png
  output/eda_summary.json
  output/eda_table1.csv
  output/eda_group_tests.csv
"""

import argparse, json, os, math, warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────────────────────
# Optional stats (Welch t / Chi2) — add "scipy" to environment.yml
# ────────────────────────────────────────────────────────────────
try:
    from scipy import stats
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False
    warnings.warn("scipy not available; group tests will be skipped.")

# Always render off-screen
plt.switch_backend("Agg")

# ────────────────────────────────────────────────────────────────
# Aesthetics — tuned for “publishable” quality
# ────────────────────────────────────────────────────────────────
mpl.rcParams.update({
    # figure / save
    "figure.dpi": 140,
    "savefig.dpi": 240,
    "savefig.bbox": "tight",
    "figure.constrained_layout.use": True,

    # fonts & text
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,

    # axes & grid
    "axes.grid": True,
    "grid.color": "#9aa4b1",
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,

    # legend
    "legend.frameon": False,

    # color cycle (subtle, colorblind-friendly-ish)
    "axes.prop_cycle": mpl.cycler(color=["#4464ad", "#e07a5f", "#3d9970", "#d38c2c", "#9d4edd", "#2a9d8f"]),
})

# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────
def ensure_dir(p: str):
    d = os.path.dirname(p) if os.path.splitext(p)[1] else p
    if d:
        os.makedirs(d, exist_ok=True)

def save_and_close(fig, path):
    ensure_dir(path)
    fig.savefig(path, facecolor="white")
    plt.close(fig)

def shannon_entropy(probs):
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def pct(x): return float(np.round(100.0 * x, 2))

def _fmt_float(x, nd=3):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return ""

def _thousands(ax, axis="y"):
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: f"{int(v):,}")
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)

def pretty_bar_labels(ax, fmt="{:,.0f}", min_height=0):
    for p in ax.patches:
        h = p.get_height()
        if h >= min_height:
            ax.annotate(fmt.format(h),
                        (p.get_x() + p.get_width()/2, h),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9)

def is_binary_like(s: pd.Series) -> bool:
    vals = pd.unique(s.dropna())
    try:
        vals = vals.astype(float)  # allow 0/1 stored as strings
    except Exception:
        return False
    return set(vals).issubset({0.0, 1.0}) and len(vals) <= 2

# ────────────────────────────────────────────────────────────────
# Figures (polished)
# ────────────────────────────────────────────────────────────────
def make_fig_target_dist(y, figs_dir):
    fig_w = 5.8 if y.nunique() <= 5 else 7.6
    fig, ax = plt.subplots(figsize=(fig_w, 4.4))
    vc = y.value_counts().sort_index()
    ax.bar(vc.index.astype(str), vc.values, linewidth=0.8, edgecolor="white")
    ax.set_title("Target distribution")
    ax.set_xlabel(y.name)
    ax.set_ylabel("Count")
    _thousands(ax, "y")
    ax.set_ylim(0, vc.values.max() * 1.15)
    pretty_bar_labels(ax, "{:,.0f}")
    path = os.path.join(figs_dir, "eda_target_dist.png")
    save_and_close(fig, path)
    return path

def make_fig_missing(df, figs_dir):
    miss_pct = (df.isna().sum() / len(df) * 100.0).sort_values(ascending=False)
    fig_w = max(6, 0.36*len(df.columns) + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 4.4))
    ax.bar(miss_pct.index, miss_pct.values, linewidth=0.8, edgecolor="white")
    ax.set_title("Missingness by column (%)")
    ax.set_ylabel("% missing")
    ax.set_ylim(0, max(5, miss_pct.values.max()*1.15))
    ax.tick_params(axis="x", rotation=60)
    for i, v in enumerate(miss_pct.values):  # annotate >=1%
        if v >= 1.0:
            ax.annotate(f"{v:.1f}%", (i, v), xytext=(0, 5),
                        textcoords="offset points", ha="center", va="bottom", fontsize=9)
    path = os.path.join(figs_dir, "eda_missing.png")
    save_and_close(fig, path)
    return path

def make_fig_corr(X_num, figs_dir):
    if X_num.shape[1] < 2:
        return None
    corr = X_num.corr()
    k = X_num.shape[1]
    sz = min(0.45*k + 3, 13)
    fig, ax = plt.subplots(figsize=(sz, sz))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    cbar = fig.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    ax.set_title("Feature correlations (numeric)")
    ax.set_xticks(range(k), labels=X_num.columns, rotation=90)
    ax.set_yticks(range(k), labels=X_num.columns)
    if k <= 12:  # annotate values if not too many features
        for (i, j), val in np.ndenumerate(corr.values):
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black")
    path = os.path.join(figs_dir, "eda_corr.png")
    save_and_close(fig, path)
    return path

def make_fig_num_hists(X_num, figs_dir):
    if X_num.shape[1] == 0:
        return None
    n = X_num.shape[1]
    cols = 3
    rows = int(math.ceil(n / cols))
    fig = plt.figure(figsize=(cols*4.6, max(3.8*rows, 3.8)))
    for i, c in enumerate(X_num.columns, 1):
        ax = plt.subplot(rows, cols, i)
        vals = X_num[c].dropna().values
        ax.hist(vals, bins="auto", edgecolor="white", linewidth=0.6, alpha=0.9)
        m = np.nanmedian(vals)
        if np.isfinite(m):
            ax.axvline(m, linestyle="--", linewidth=1.2, label=f"median = {m:.2f}")
        ax.set_title(c, fontsize=10, pad=6)
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        if i % cols == 1:
            ax.set_ylabel("Count")
        _thousands(ax, "y")
        if i == 1:
            ax.legend(fontsize=8, loc="upper right")
    plt.suptitle("Numeric feature distributions", y=1.02, fontsize=12, fontweight="semibold")
    path = os.path.join(figs_dir, "eda_feature_hists.png")
    save_and_close(plt.gcf(), path)
    return path

def make_fig_num_hists_by_class(X_num, y, figs_dir, max_feats=6):
    if X_num.shape[1] == 0:
        return None
    cols_sel = X_num.columns[:max_feats]
    n = len(cols_sel)
    cols = 3
    rows = int(math.ceil(n / cols))
    fig = plt.figure(figsize=(cols*4.8, max(3.9*rows, 3.9)))
    for i, c in enumerate(cols_sel, 1):
        ax = plt.subplot(rows, cols, i)
        a = X_num.loc[y==0, c].dropna().values
        b = X_num.loc[y==1, c].dropna().values
        ax.hist(a, bins="auto", alpha=0.70, label="Class 0", edgecolor="white", linewidth=0.6)
        ax.hist(b, bins="auto", alpha=0.55, label="Class 1", edgecolor="white", linewidth=0.6)
        if a.size: ax.axvline(np.nanmedian(a), linestyle="--", linewidth=1.1, alpha=0.9)
        if b.size: ax.axvline(np.nanmedian(b), linestyle=":",  linewidth=1.1, alpha=0.9)
        ax.set_title(c, fontsize=10, pad=6)
        ax.tick_params(axis='x', labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        if i % cols == 1:
            ax.set_ylabel("Count")
        _thousands(ax, "y")
        if i == 1:
            ax.legend(fontsize=8, frameon=False)
    plt.suptitle("Numeric distributions by class (subset)", y=1.02, fontsize=12, fontweight="semibold")
    path = os.path.join(figs_dir, "eda_feature_hists_by_class.png")
    save_and_close(plt.gcf(), path)
    return path

# ────────────────────────────────────────────────────────────────
# Tables
# ────────────────────────────────────────────────────────────────
def table1_descriptives(df, target):
    """Table 1: overall + by-class for numeric and categorical/binary."""
    y = df[target]
    X = df.drop(columns=[target])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    rows = []

    # numeric
    for c in num_cols:
        s = X[c]; s0 = X.loc[y==0, c]; s1 = X.loc[y==1, c]
        rows.append({
            "Feature": c, "Type": "numeric", "Missing_%": pct(s.isna().mean()),
            "Overall_mean": _fmt_float(s.mean()),
            "Overall_sd": _fmt_float(s.std()),
            "Class0_mean": _fmt_float(s0.mean()),
            "Class0_sd": _fmt_float(s0.std()),
            "Class1_mean": _fmt_float(s1.mean()),
            "Class1_sd": _fmt_float(s1.std()),
        })

    # categorical / binary
    for c in cat_cols:
        s = X[c]; miss = pct(s.isna().mean())
        if is_binary_like(s):
            s2 = pd.to_numeric(s, errors="coerce")
            s0, s1 = s2[y==0], s2[y==1]
            rows.append({
                "Feature": c, "Type": "binary", "Missing_%": miss,
                "Overall_%1": _fmt_float(100*s2.mean(), 2),
                "Class0_%1": _fmt_float(100*s0.mean(), 2),
                "Class1_%1": _fmt_float(100*s1.mean(), 2),
            })
        else:
            vc = s.value_counts(dropna=True)
            top_cat = vc.index[0] if len(vc) else ""
            top_pct = pct(vc.iloc[0] / len(s)) if len(vc) else 0.0
            rows.append({
                "Feature": c, "Type": "categorical", "Missing_%": miss,
                "Top_category": str(top_cat), "Top_category_%": top_pct
            })

    return pd.DataFrame(rows)

def group_tests(df, target):
    if not SCIPY_OK:
        return pd.DataFrame(columns=["Feature","Type","Test","Statistic","PValue"])
    y = df[target]; X = df.drop(columns=[target])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    rows = []
    # Welch t-tests
    for c in num_cols:
        a = X.loc[y==0, c].dropna().values
        b = X.loc[y==1, c].dropna().values
        if len(a) >= 2 and len(b) >= 2:
            t, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
            rows.append({"Feature": c, "Type": "numeric", "Test": "Welch t", "Statistic": float(t), "PValue": float(p)})
    # Chi-square
    for c in cat_cols:
        cont = pd.crosstab(X[c], y)
        if cont.shape[0] >= 2 and cont.shape[1] == 2:
            chi2, p, dof, _ = stats.chi2_contingency(cont.fillna(0))
            rows.append({"Feature": c, "Type": "categorical", "Test": "Chi-square", "Statistic": float(chi2), "PValue": float(p)})
    return pd.DataFrame(rows).sort_values("PValue").reset_index(drop=True)

# ────────────────────────────────────────────────────────────────
# Main profiling
# ────────────────────────────────────────────────────────────────
def profile_dataset(csv_path: str, target_col: str, figs_dir: str, out_json: str,
                    out_table1_csv: str, out_tests_csv: str):
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {csv_path}")

    y = df[target_col]
    X = df.drop(columns=[target_col])

    n_samples = int(df.shape[0])
    n_features = int(X.shape[1])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # class balance & entropy
    class_counts = y.value_counts().sort_index()
    class_pct = (class_counts / n_samples * 100.0).round(2)
    maj = int(class_counts.max()); mino = int(class_counts.min())
    imb_ratio = float(np.round(mino / maj, 4)) if maj > 0 else 1.0
    probs = (class_counts / class_counts.sum()).values
    entropy_bits = float(np.round(shannon_entropy(probs), 4))

    # missingness & sparsity
    missing_pct = (df.isna().sum() / len(df) * 100.0).round(2).to_dict()
    sparsity = {c: pct((X[c] == 0).mean()) for c in num_cols}

    # figures
    p_target = make_fig_target_dist(y, figs_dir)
    p_missing = make_fig_missing(df, figs_dir)
    p_corr = make_fig_corr(X[num_cols], figs_dir) if len(num_cols) >= 2 else None
    p_hists = make_fig_num_hists(X[num_cols], figs_dir) if len(num_cols) else None
    p_hists_by_cls = make_fig_num_hists_by_class(X[num_cols], y, figs_dir) if len(num_cols) else None

    # tables
    t1 = table1_descriptives(df, target_col)
    t1.to_csv(out_table1_csv, index=False)

    gt = group_tests(df, target_col)
    gt.to_csv(out_tests_csv, index=False)

    # summary json
    summary = {
        "n_samples": n_samples,
        "n_features": n_features,
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "class_counts": class_counts.to_dict(),
        "class_pct": class_pct.to_dict(),
        "imbalance_ratio": imb_ratio,
        "target_entropy_bits": entropy_bits,
        "missing_pct": missing_pct,
        "sparsity_pct_numeric": sparsity,
        "figs": {
            "eda_target_dist": os.path.basename(p_target) if p_target else None,
            "eda_corr": os.path.basename(p_corr) if p_corr else None,
            "eda_missing": os.path.basename(p_missing) if p_missing else None,
            "eda_feature_hists": os.path.basename(p_hists) if p_hists else None,
            "eda_feature_hists_by_class": os.path.basename(p_hists_by_cls) if p_hists_by_cls else None
        },
        "artifacts": {
            "table1_csv": os.path.basename(out_table1_csv),
            "group_tests_csv": os.path.basename(out_tests_csv)
        }
    }
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"📊 EDA summary: {out_json}")
    print(f"📄 Table 1:     {out_table1_csv}")
    print(f"📄 Group tests: {out_tests_csv}")
    for k, v in summary["figs"].items():
        if v: print(f"📁 {k}: {os.path.join(figs_dir, v)}")

# ────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Quick EDA profile + polished figures")
    ap.add_argument("--data", required=True, help="Path to CSV")
    ap.add_argument("--target", default="Diagnosed_PA", help="Target column name")
    ap.add_argument("--figs_dir", default="output/figs", help="Directory for EDA figures")
    ap.add_argument("--output_json", default="output/eda_summary.json", help="Summary JSON path")
    ap.add_argument("--table1_csv", default="output/eda_table1.csv", help="Descriptive stats (Table 1)")
    ap.add_argument("--tests_csv", default="output/eda_group_tests.csv", help="Group tests CSV")
    args = ap.parse_args()

    profile_dataset(
        csv_path=args.data,
        target_col=args.target,
        figs_dir=args.figs_dir,
        out_json=args.output_json,
        out_table1_csv=args.table1_csv,
        out_tests_csv=args.tests_csv
    )

if __name__ == "__main__":
    main()
