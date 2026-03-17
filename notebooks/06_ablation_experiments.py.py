"""
07_ablation_experiments.py — Model ablation and commodity-specific analysis
============================================================================
Runs three experiments to complete the analytical story:

  Experiment 1: Full model (baseline — already done, re-run for consistency)
  Experiment 2: Remove price_pct_change (pure early-warning model)
  Experiment 3: Commodity-specific models (cabai group vs bawang group)

All results saved to outputs/ with clear filenames.

REQUIRES: data/processed/real_panel_dataset.csv (from script 06)

USAGE:
    python scripts/07_ablation_experiments.py
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
PROC_DIR = PROJECT / "data" / "processed"
OUTPUT_DIR = PROJECT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_fscore_support, roc_curve)
from sklearn.preprocessing import StandardScaler


# ══════════════════════════════════════════════════════════
# FEATURE SETS
# ══════════════════════════════════════════════════════════

FULL_FEATURES = [
    "supply_ndvi_lag2", "supply_ndvi_lag3", "supply_ndvi_lag4",
    "local_ndvi_lag2", "local_ndvi_lag3",
    "rain_lag1", "rain_lag2", "rain_4wk_sum",
    "supply_ndvi_4wk_trend",
    "min_distance_km", "min_travel_hrs",
    "month", "is_wet_season",
    "flood_risk_lag1", "stress_x_distance",
    "price_pct_change",
]

# Experiment 2: pure early-warning (no market momentum)
EARLYWARNING_FEATURES = [
    "supply_ndvi_lag2", "supply_ndvi_lag3", "supply_ndvi_lag4",
    "local_ndvi_lag2", "local_ndvi_lag3",
    "rain_lag1", "rain_lag2", "rain_4wk_sum",
    "supply_ndvi_4wk_trend",
    "min_distance_km", "min_travel_hrs",
    "month", "is_wet_season",
    "flood_risk_lag1", "stress_x_distance",
]

# Commodity groups
CABAI_COMMODITIES = ["Cabai Merah Besar + Keriting", "Cabai Rawit Merah + Hijau"]
BAWANG_COMMODITIES = ["Bawang Merah Ukuran Sedang", "Bawang Putih Ukuran Sedang"]


# ══════════════════════════════════════════════════════════
# MODEL TRAINING FUNCTION
# ══════════════════════════════════════════════════════════

def run_experiment(panel, features, experiment_name, split_quantile=0.8):
    """Train LR + RF on given features, return results dict."""
    log.info(f"\n{'='*60}")
    log.info(f"EXPERIMENT: {experiment_name}")
    log.info(f"{'='*60}")
    log.info(f"Features ({len(features)}): {features}")
    log.info(f"Data: {len(panel)} rows")

    panel = panel.sort_values("date")
    split_date = panel["date"].quantile(split_quantile)
    train = panel[panel["date"] <= split_date]
    test = panel[panel["date"] > split_date]

    avail = [c for c in features if c in panel.columns]
    X_train = train[avail].fillna(0)
    y_train = train["spike"]
    X_test = test[avail].fillna(0)
    y_test = test["spike"]

    if len(X_test) == 0 or y_test.nunique() < 2:
        log.warning(f"  Insufficient test data ({len(X_test)} rows, {y_test.nunique()} classes). Skipping.")
        return None

    log.info(f"  Train: {len(train)} rows | spike rate: {y_train.mean()*100:.1f}%")
    log.info(f"  Test:  {len(test)} rows | spike rate: {y_test.mean()*100:.1f}%")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    results = {"experiment": experiment_name, "n_train": len(train), "n_test": len(test),
               "n_features": len(avail), "features": avail,
               "spike_rate_train": y_train.mean(), "spike_rate_test": y_test.mean()}

    # ── Logistic Regression ──
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=42)
    lr.fit(X_train_s, y_train)
    lr_pred = lr.predict(X_test_s)
    lr_prob = lr.predict_proba(X_test_s)[:, 1]
    lr_p, lr_r, lr_f1, _ = precision_recall_fscore_support(y_test, lr_pred, average="binary", zero_division=0)
    lr_auc = roc_auc_score(y_test, lr_prob)

    results.update({
        "lr_precision": lr_p, "lr_recall": lr_r, "lr_f1": lr_f1, "lr_auc": lr_auc,
        "lr_cm": confusion_matrix(y_test, lr_pred).tolist(),
    })

    coef_df = pd.DataFrame({"feature": avail, "coefficient": lr.coef_[0],
                             "abs_coef": np.abs(lr.coef_[0])}).sort_values("abs_coef", ascending=False)
    results["lr_coefficients"] = coef_df

    log.info(f"  LR: P={lr_p:.3f} R={lr_r:.3f} F1={lr_f1:.3f} AUC={lr_auc:.3f}")
    log.info(f"  Top LR coefficients:")
    for _, row in coef_df.head(5).iterrows():
        log.info(f"    {row['feature']:30s} {row['coefficient']:+.4f}")

    # ── Random Forest ──
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10,
                                class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_p, rf_r, rf_f1, _ = precision_recall_fscore_support(y_test, rf_pred, average="binary", zero_division=0)
    rf_auc = roc_auc_score(y_test, rf_prob)

    results.update({
        "rf_precision": rf_p, "rf_recall": rf_r, "rf_f1": rf_f1, "rf_auc": rf_auc,
        "rf_cm": confusion_matrix(y_test, rf_pred).tolist(),
    })

    fi_df = pd.DataFrame({"feature": avail, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
    results["rf_feature_importance"] = fi_df

    log.info(f"  RF: P={rf_p:.3f} R={rf_r:.3f} F1={rf_f1:.3f} AUC={rf_auc:.3f}")
    log.info(f"  Top RF features:")
    for _, row in fi_df.head(5).iterrows():
        log.info(f"    {row['feature']:30s} {row['importance']:.4f}")

    # Store models and test data for plotting
    results["_lr_model"] = lr
    results["_rf_model"] = rf
    results["_y_test"] = y_test
    results["_lr_prob"] = lr_prob
    results["_rf_prob"] = rf_prob
    results["_lr_pred"] = lr_pred
    results["_rf_pred"] = rf_pred

    return results


# ══════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════

def plot_comparison(all_results):
    """Generate comparison charts across all experiments."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not available, skipping plots")
        return

    valid_results = {k: v for k, v in all_results.items() if v is not None}

    # ── 1. AUC comparison bar chart ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    names = list(valid_results.keys())
    lr_aucs = [v["lr_auc"] for v in valid_results.values()]
    rf_aucs = [v["rf_auc"] for v in valid_results.values()]

    x = np.arange(len(names))
    w = 0.35
    axes[0].bar(x - w/2, lr_aucs, w, label="Logistic Reg.", color="#2E75B6")
    axes[0].bar(x + w/2, rf_aucs, w, label="Random Forest", color="#E8593C")
    axes[0].set_ylabel("AUC-ROC")
    axes[0].set_title("AUC comparison across experiments", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
    axes[0].legend()
    axes[0].set_ylim(0.5, 1.0)
    for i, (la, ra) in enumerate(zip(lr_aucs, rf_aucs)):
        axes[0].text(i - w/2, la + 0.01, f"{la:.3f}", ha="center", fontsize=8)
        axes[0].text(i + w/2, ra + 0.01, f"{ra:.3f}", ha="center", fontsize=8)

    # F1 comparison
    lr_f1s = [v["lr_f1"] for v in valid_results.values()]
    rf_f1s = [v["rf_f1"] for v in valid_results.values()]

    axes[1].bar(x - w/2, lr_f1s, w, label="Logistic Reg.", color="#2E75B6")
    axes[1].bar(x + w/2, rf_f1s, w, label="Random Forest", color="#E8593C")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("F1 comparison across experiments", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([n.replace(" ", "\n") for n in names], fontsize=9)
    axes[1].legend()
    axes[1].set_ylim(0, 1.0)
    for i, (lf, rf) in enumerate(zip(lr_f1s, rf_f1s)):
        axes[1].text(i - w/2, lf + 0.01, f"{lf:.3f}", ha="center", fontsize=8)
        axes[1].text(i + w/2, rf + 0.01, f"{rf:.3f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ablation_auc_f1_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: ablation_auc_f1_comparison.png")

    # ── 2. ROC curves overlay ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    colors = {"Full model": "#2E75B6", "Early-warning only": "#E8593C",
              "Cabai only": "#1D9E75", "Bawang only": "#854F0B"}

    for name, res in valid_results.items():
        c = colors.get(name, "#888888")
        fpr_lr, tpr_lr, _ = roc_curve(res["_y_test"], res["_lr_prob"])
        axes[0].plot(fpr_lr, tpr_lr, label=f"{name} (AUC={res['lr_auc']:.3f})", color=c, linewidth=1.5)
        fpr_rf, tpr_rf, _ = roc_curve(res["_y_test"], res["_rf_prob"])
        axes[1].plot(fpr_rf, tpr_rf, label=f"{name} (AUC={res['rf_auc']:.3f})", color=c, linewidth=1.5)

    for ax, title in zip(axes, ["Logistic Regression", "Random Forest"]):
        ax.plot([0,1],[0,1],"k--",alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)

    plt.suptitle("ROC Curves — All Experiments", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ablation_roc_overlay.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: ablation_roc_overlay.png")

    # ── 3. Feature importance comparison (full vs early-warning) ──
    if "Full model" in valid_results and "Early-warning only" in valid_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, (name, res) in zip(axes, [("Full model", valid_results["Full model"]),
                                           ("Early-warning only", valid_results["Early-warning only"])]):
            fi = res["rf_feature_importance"].head(12)
            colors_fi = ["#2E75B6" if "ndvi" in f or "supply" in f else
                         "#1D9E75" if "rain" in f or "flood" in f else
                         "#E8593C" if "price" in f else "#888888"
                         for f in fi["feature"]]
            ax.barh(fi["feature"][::-1], fi["importance"][::-1], color=colors_fi[::-1])
            ax.set_xlabel("Importance (Gini)")
            ax.set_title(f"{name}\nRF AUC={res['rf_auc']:.3f}", fontsize=11, fontweight="bold")

        plt.suptitle("Feature importance: Full model vs Early-warning only", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "ablation_feature_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"  Saved: ablation_feature_comparison.png")

    # ── 4. Confusion matrices for all experiments ──
    n_exp = len(valid_results)
    fig, axes = plt.subplots(2, n_exp, figsize=(5*n_exp, 9))
    if n_exp == 1:
        axes = axes.reshape(2, 1)

    for col, (name, res) in enumerate(valid_results.items()):
        for row, (model_name, preds) in enumerate([("LR", res["_lr_pred"]), ("RF", res["_rf_pred"])]):
            ax = axes[row, col]
            cm = confusion_matrix(res["_y_test"], preds)
            ax.imshow(cm, cmap="Blues")
            ax.set_title(f"{name}\n{model_name}", fontsize=10, fontweight="bold")
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["No","Spike"], fontsize=9); ax.set_yticklabels(["No","Spike"], fontsize=9)
            for i in range(2):
                for j in range(2):
                    ax.text(j,i,str(cm[i,j]),ha="center",va="center",
                            color="white" if cm[i,j]>cm.max()/2 else "black", fontsize=14)

    plt.suptitle("Confusion Matrices — All Experiments", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ablation_confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: ablation_confusion_matrices.png")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    log.info("=" * 60)
    log.info("ABLATION EXPERIMENTS — Price Spike Early-Warning")
    log.info("=" * 60)

    # Load the panel dataset
    panel = pd.read_csv(PROC_DIR / "real_panel_dataset.csv", parse_dates=["date"])
    log.info(f"Panel loaded: {len(panel)} rows, {len(panel.columns)} cols")
    log.info(f"Commodities: {panel['commodity'].unique().tolist()}")
    log.info(f"Locations: {panel['district'].unique().tolist()}")

    all_results = {}

    # ── Experiment 1: Full model (baseline) ──
    all_results["Full model"] = run_experiment(
        panel, FULL_FEATURES, "Full model (with price momentum)")

    # ── Experiment 2: Early-warning only (no price_pct_change) ──
    all_results["Early-warning only"] = run_experiment(
        panel, EARLYWARNING_FEATURES, "Early-warning only (no price momentum)")

    # ── Experiment 3a: Cabai only ──
    cabai_panel = panel[panel["commodity"].isin(CABAI_COMMODITIES)].copy()
    if len(cabai_panel) > 100:
        all_results["Cabai only"] = run_experiment(
            cabai_panel, EARLYWARNING_FEATURES, "Cabai only (early-warning features)")
    else:
        log.warning(f"  Cabai panel too small ({len(cabai_panel)} rows), skipping")

    # ── Experiment 3b: Bawang only ──
    bawang_panel = panel[panel["commodity"].isin(BAWANG_COMMODITIES)].copy()
    if len(bawang_panel) > 100:
        all_results["Bawang only"] = run_experiment(
            bawang_panel, EARLYWARNING_FEATURES, "Bawang only (early-warning features)")
    else:
        log.warning(f"  Bawang panel too small ({len(bawang_panel)} rows), skipping")

    # ── Save summary CSV ──
    summary_rows = []
    for name, res in all_results.items():
        if res is None:
            continue
        summary_rows.append({
            "experiment": name,
            "n_train": res["n_train"],
            "n_test": res["n_test"],
            "n_features": res["n_features"],
            "spike_rate_test": round(res["spike_rate_test"], 3),
            "lr_precision": round(res["lr_precision"], 3),
            "lr_recall": round(res["lr_recall"], 3),
            "lr_f1": round(res["lr_f1"], 3),
            "lr_auc": round(res["lr_auc"], 3),
            "rf_precision": round(res["rf_precision"], 3),
            "rf_recall": round(res["rf_recall"], 3),
            "rf_f1": round(res["rf_f1"], 3),
            "rf_auc": round(res["rf_auc"], 3),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "real_model_ablation.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info(f"\n{'='*60}")
    log.info(f"ABLATION SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"\n{summary_df.to_string(index=False)}")
    log.info(f"\nSaved to: {summary_path}")

    # ── Save detailed reports ──
    report = []
    report.append("=" * 70)
    report.append("ABLATION EXPERIMENT RESULTS")
    report.append("Horticultural Price Spike Early-Warning, South Sumatra")
    report.append("=" * 70)

    for name, res in all_results.items():
        if res is None:
            continue
        report.append(f"\n{'_'*70}")
        report.append(f"EXPERIMENT: {name}")
        report.append(f"{'_'*70}")
        report.append(f"Train: {res['n_train']} rows | Test: {res['n_test']} rows")
        report.append(f"Features ({res['n_features']}): {res['features']}")
        report.append(f"Spike rate (test): {res['spike_rate_test']*100:.1f}%")
        report.append(f"\nLogistic Regression:")
        report.append(f"  P={res['lr_precision']:.3f} R={res['lr_recall']:.3f} F1={res['lr_f1']:.3f} AUC={res['lr_auc']:.3f}")
        report.append(f"  Confusion matrix: {res['lr_cm']}")
        report.append(f"  Top coefficients:")
        for _, row in res["lr_coefficients"].head(8).iterrows():
            report.append(f"    {row['feature']:30s} {row['coefficient']:+.4f}")
        report.append(f"\nRandom Forest:")
        report.append(f"  P={res['rf_precision']:.3f} R={res['rf_recall']:.3f} F1={res['rf_f1']:.3f} AUC={res['rf_auc']:.3f}")
        report.append(f"  Confusion matrix: {res['rf_cm']}")
        report.append(f"  Top features:")
        for _, row in res["rf_feature_importance"].head(8).iterrows():
            report.append(f"    {row['feature']:30s} {row['importance']:.4f}")

    # Key comparison
    if "Full model" in all_results and all_results["Full model"] and \
       "Early-warning only" in all_results and all_results["Early-warning only"]:
        fm = all_results["Full model"]
        ew = all_results["Early-warning only"]
        report.append(f"\n{'='*70}")
        report.append("KEY COMPARISON: Full model vs Early-warning only")
        report.append(f"{'='*70}")
        report.append(f"AUC drop (LR):  {fm['lr_auc']:.3f} -> {ew['lr_auc']:.3f} (delta: {ew['lr_auc']-fm['lr_auc']:+.3f})")
        report.append(f"AUC drop (RF):  {fm['rf_auc']:.3f} -> {ew['rf_auc']:.3f} (delta: {ew['rf_auc']-fm['rf_auc']:+.3f})")
        report.append(f"F1 drop (LR):   {fm['lr_f1']:.3f} -> {ew['lr_f1']:.3f} (delta: {ew['lr_f1']-fm['lr_f1']:+.3f})")
        report.append(f"F1 drop (RF):   {fm['rf_f1']:.3f} -> {ew['rf_f1']:.3f} (delta: {ew['rf_f1']-fm['rf_f1']:+.3f})")
        report.append(f"\nInterpretation:")
        auc_drop_lr = fm['lr_auc'] - ew['lr_auc']
        if auc_drop_lr < 0.05:
            report.append(f"  AUC drop < 0.05: Environmental variables ALONE are nearly as powerful as the full model.")
            report.append(f"  This STRONGLY supports the satellite early-warning claim.")
        elif auc_drop_lr < 0.10:
            report.append(f"  AUC drop 0.05-0.10: Environmental variables provide substantial standalone power.")
            report.append(f"  Price momentum adds value but is not essential for useful predictions.")
        elif auc_drop_lr < 0.15:
            report.append(f"  AUC drop 0.10-0.15: Environmental variables provide moderate standalone power.")
            report.append(f"  Both signals are needed for best performance.")
        else:
            report.append(f"  AUC drop > 0.15: The model relies heavily on price momentum.")
            report.append(f"  Environmental variables help but cannot replace market signals alone.")

    report_path = OUTPUT_DIR / "ablation_full_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    log.info(f"Full report saved: {report_path}")

    # ── Generate plots ──
    log.info("\nGenerating comparison plots...")
    plot_comparison(all_results)

    log.info(f"\n{'='*60}")
    log.info("ALL EXPERIMENTS COMPLETE")
    log.info(f"{'='*60}")
    log.info(f"Files generated:")
    for f in sorted(OUTPUT_DIR.glob("ablation_*")):
        log.info(f"  {f.name}")
    log.info(f"  real_model_ablation.csv")


if __name__ == "__main__":
    main()
