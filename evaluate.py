"""
evaluate.py  (v2 — improvements applied)
===========
Phase 4: Clinical Validation and Narrative Generation.

IMPROVEMENT CHANGELOG vs v1:
  IMP 1 — full_evaluation now accepts an optional TemperatureScaling module.
           All probability outputs (ROC, PR, calibration, metrics) are computed
           from temperature-calibrated logits when ts is provided.
           This directly fixes the calibration curve which showed systematic
           over-confidence (fraction positives << predicted probability across
           all bins in the original run).
  IMP 2 — Calibration plot shows BOTH raw and calibrated curves side by side
           to demonstrate the improvement (useful for the paper's Pillar 1 claim).
  IMP 3 — print_clinical_report now shows rule polarity summary
           (# positive / # negative rules) to detect imbalanced rule learning.

Original features retained:
  - ROC-AUC, PR curve, calibration, rule weight plots
  - Rule extraction and clinical narrative generation
  - Cross-population validation (Pima Bias test)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    f1_score, recall_score,
)
from sklearn.calibration import calibration_curve
from kanfis_model import KANFIS, TemperatureScaling


# ─────────────────────────────────────────────
# LINGUISTIC VARIABLE MAPPING
# ─────────────────────────────────────────────
LINGUISTIC_MAP = {
    "FastingGlucose":   {(-4,-1.5): "Normal",      (-1.5, 0.5): "Elevated",         (0.5, 4): "Critically Elevated"},
    "RandomGlucose":    {(-4,-1.5): "Normal",      (-1.5, 0.5): "Moderately High",  (0.5, 4): "Very High"},
    "BMI":              {(-4,-1.0): "Underweight",  (-1.0, 1.0): "Normal",           (1.0, 4): "Obese"},
    "Insulin":          {(-4,-0.5): "Low",          (-0.5, 0.5): "Normal",           (0.5, 4): "High"},
    "SystolicBP":       {(-4,-1.0): "Normal",       (-1.0, 1.0): "Elevated",         (1.0, 4): "Hypertensive"},
    "DiastolicBP":      {(-4,-1.0): "Normal",       (-1.0, 1.0): "Moderately High",  (1.0, 4): "Severely High"},
    "Hypertension":     {(-4,  0.0): "Absent",      (0.0, 4): "Present"},
    "IschemicStroke":   {(-4,  0.0): "No History",  (0.0, 4): "History Present"},
    "HeartDisease":     {(-4,  0.0): "No History",  (0.0, 4): "History Present"},
    "Age":              {(-4, -1.0): "Young",        (-1.0, 1.0): "Middle-Aged",     (1.0, 4): "Elderly"},
    "VascularRiskScore":{(-4,  0.0): "Low Risk",    (0.0, 1.5): "Moderate Risk",     (1.5, 4): "High Risk"},
    "MetabolicBurden":  {(-4, -0.5): "Low",          (-0.5, 0.5): "Moderate",        (0.5, 4): "High"},
    "CardioFlag":       {(-4,  0.0): "No Comorbidities", (0.0, 4): "Comorbidities Present"},
}

DEFAULT_TERM = "Elevated"


def _z_to_linguistic(feature_name: str, z_value: float) -> str:
    mapping = LINGUISTIC_MAP.get(feature_name, {})
    for (low, high), term in mapping.items():
        if low <= z_value < high:
            return term
    return DEFAULT_TERM


# ─────────────────────────────────────────────
# 1.  RULE EXTRACTION
# ─────────────────────────────────────────────
def extract_rules(
    model: KANFIS,
    feature_names: list[str],
    active_threshold: float = 0.01,
) -> list[dict]:
    """
    Extract surviving (non-pruned) fuzzy rules and translate them into
    human-readable antecedents + confidence weights.
    """
    centres = model.fuzzy_layer.centres.detach().cpu().numpy()
    weights = model.rule_head.rule_weights.weight.detach().cpu().numpy().squeeze(0)

    active_rule_indices = [
        i for i, w in enumerate(weights) if abs(w) >= active_threshold
    ]

    rules = []
    for r_idx in active_rule_indices:
        rule_centres = centres[r_idx]
        weight_val   = float(weights[r_idx])

        antecedents = []
        offset = 0
        for group_name, feat_indices in model.group_map.items():
            g_out = 16
            group_centre_mean = float(rule_centres[offset: offset + g_out].mean())
            offset += g_out
            for f_idx in feat_indices:
                if f_idx < len(feature_names):
                    fname = feature_names[f_idx]
                    term  = _z_to_linguistic(fname, group_centre_mean)
                    antecedents.append((fname, term))

        rules.append({
            "rule_idx":    r_idx,
            "weight":      weight_val,
            "antecedents": antecedents,
            "raw_centres": rule_centres,
        })

    rules.sort(key=lambda r: abs(r["weight"]), reverse=True)
    return rules


# ─────────────────────────────────────────────
# 2.  CLINICAL NARRATIVE GENERATION
# ─────────────────────────────────────────────
def generate_narratives(
    rules: list[dict],
    top_k: int = 7,
) -> list[str]:
    narratives = []
    for i, rule in enumerate(rules[:top_k]):
        conditions = " AND ".join(
            f"{feat} is {term}" for feat, term in rule["antecedents"]
        )
        direction  = "HIGH" if rule["weight"] > 0 else "PROTECTIVE / LOW"
        confidence = abs(rule["weight"])
        narrative  = (
            f"Rule {i+1}:\n"
            f"  IF   {conditions}\n"
            f"  THEN Risk of Type-2 Diabetes is {direction}\n"
            f"       (Consequent Weight = {rule['weight']:+.4f}, "
            f"Confidence ∝ {confidence:.4f})\n"
        )
        narratives.append(narrative)
    return narratives


def print_clinical_report(
    model: KANFIS,
    feature_names: list[str],
    top_k: int = 7,
) -> None:
    """
    IMP 3 — now also prints rule polarity summary to detect imbalanced
    rule learning (e.g. 7 protective vs 3 risk rules signals class
    imbalance surviving SMOTE, or diversity collapse).
    """
    print("\n" + "═"*65)
    print("  KANFIS — Extracted Clinical Decision Rules")
    print("═"*65)

    rules     = extract_rules(model, feature_names)
    narratives = generate_narratives(rules, top_k)

    active    = model.get_active_rules()
    n_pos     = sum(1 for r in rules if r["weight"] > 0)
    n_neg     = len(rules) - n_pos

    print(f"  Total active rules  : {len(active)} (after L1 pruning)")
    # IMP 3: polarity summary
    print(f"  Rule polarity       : {n_pos} risk-positive / {n_neg} protective")
    if n_neg > 2 * n_pos:
        print("  ⚠ WARNING: Protective rules dominate. "
              "Consider increasing focal_gamma or checking class balance.")
    print()

    for n in narratives:
        print(n)


# ─────────────────────────────────────────────
# 3.  FULL EVALUATION SUITE
# ─────────────────────────────────────────────
@torch.no_grad()
def full_evaluation(
    model: KANFIS,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    ts: "TemperatureScaling | None" = None,   # IMP 1: optional calibration
    save_plots: bool = True,
    output_dir: str = ".",
) -> dict:
    """
    IMP 1 — full_evaluation now accepts a TemperatureScaling module (ts).
    When provided, all probability outputs are derived from calibrated
    logits (logit / T). This affects:
      - All reported metrics (AUC, F1, sensitivity, specificity, precision)
      - ROC curve
      - PR curve
      - Calibration plot (now shows raw vs calibrated comparison)

    If ts=None, behaviour is identical to v1 (raw logits used).
    """
    device = next(model.parameters()).device
    model.eval()
    if ts is not None:
        ts.eval()

    ds  = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    ldr = DataLoader(ds, batch_size=128)

    all_logits, all_labels = [], []
    for X_b, y_b in ldr:
        logit = model(X_b.to(device)).squeeze(-1).cpu()
        all_logits.append(logit)
        all_labels.append(y_b)

    raw_logits = torch.cat(all_logits).numpy()
    labels     = torch.cat(all_labels).numpy()

    # IMP 1 — apply temperature scaling if provided
    if ts is not None:
        T = ts.temperature.item()
        cal_logits = raw_logits / max(T, 0.05)
    else:
        T          = 1.0
        cal_logits = raw_logits

    raw_probs = 1 / (1 + np.exp(-raw_logits))
    probs     = 1 / (1 + np.exp(-cal_logits))
    preds     = (probs >= 0.5).astype(int)

    # ── Pillar 1: Predictive Fidelity ──────────────
    auc  = roc_auc_score(labels, probs)
    ap   = average_precision_score(labels, probs)
    f1   = f1_score(labels, preds, zero_division=0)
    sens = recall_score(labels, preds, pos_label=1, zero_division=0)
    spec = recall_score(labels, preds, pos_label=0, zero_division=0)
    cm   = confusion_matrix(labels, preds)

    print("\n" + "═"*65)
    print("  KANFIS — Evaluation Report")
    print("═"*65)
    cal_note = f" (T={T:.3f} applied)" if ts is not None else " (uncalibrated)"
    print(f"  ROC-AUC    : {auc:.4f}{cal_note}")
    print(f"  Avg Prec.  : {ap:.4f}")
    print(f"  F1-Score   : {f1:.4f}")
    print(f"  Sensitivity: {sens:.4f}  (Recall for diabetic class)")
    print(f"  Specificity: {spec:.4f}  (Recall for non-diabetic class)")
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n{classification_report(labels, preds, target_names=['Non-Diabetic', 'Diabetic'])}")

    # ── Pillar 2: Interpretability ──────────────────
    import time
    t0 = time.perf_counter()
    active_rules   = model.get_active_rules()
    rule_time_ms   = (time.perf_counter() - t0) * 1000

    print(f"  Active rules after pruning : {len(active_rules)}")
    print(f"  Rule inference time        : {rule_time_ms:.3f} ms  (vs SHAP O(n²))")

    # ── Pillar 3: Narrative output ──────────────────
    print_clinical_report(model, feature_names)

    # ── Plots ───────────────────────────────────────
    if save_plots:
        _plot_roc(labels, probs, auc, output_dir)
        _plot_pr(labels, probs, ap, output_dir)
        _plot_calibration(labels, raw_probs, probs, output_dir, T)   # IMP 2
        _plot_rule_weights(model, output_dir)

    return {
        "auc": auc, "ap": ap, "f1": f1,
        "sensitivity": sens, "specificity": spec,
        "n_active_rules": len(active_rules),
        "rule_inference_ms": rule_time_ms,
        "confusion_matrix": cm,
        "temperature": T,
    }


# ─────────────────────────────────────────────
# 4.  CROSS-POPULATION VALIDATION (Pima Bias test)
# ─────────────────────────────────────────────
def cross_population_test(
    model_pidd: KANFIS,
    X_diabd: np.ndarray,
    y_diabd: np.ndarray,
    model_kanfis: KANFIS,
) -> None:
    device = next(model_pidd.parameters()).device

    def _get_auc(m, X, y):
        m.eval()
        with torch.no_grad():
            logit = m(torch.FloatTensor(X).to(device)).squeeze(-1).cpu().numpy()
        probs = 1 / (1 + np.exp(-logit))
        return roc_auc_score(y, probs)

    auc_pidd_on_diabd   = _get_auc(model_pidd, X_diabd, y_diabd)
    auc_kanfis_on_diabd = _get_auc(model_kanfis, X_diabd, y_diabd)

    delta = auc_kanfis_on_diabd - auc_pidd_on_diabd
    print("\n" + "═"*65)
    print("  Cross-Population Validation (Pima Bias Quantification)")
    print("═"*65)
    print(f"  PIDD-trained model on DiaBD  : AUC = {auc_pidd_on_diabd:.4f}")
    print(f"  KANFIS (DiaBD-trained)        : AUC = {auc_kanfis_on_diabd:.4f}")
    print(f"  Improvement (ΔAUC)            : {delta:+.4f}")
    if delta > 0:
        print("  ✓ KANFIS demonstrates improved cross-population generalizability")
    else:
        print("  ✗ Further training or feature adjustment required")


# ─────────────────────────────────────────────
# 5.  PLOT HELPERS
# ─────────────────────────────────────────────
def _plot_roc(labels, probs, auc, output_dir):
    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2563EB", lw=2, label=f"KANFIS (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — KANFIS Diabetes Diagnostics")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curve.png", dpi=150); plt.close()
    print("  [Plot] ROC curve saved.")


def _plot_pr(labels, probs, ap, output_dir):
    prec, rec, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.step(rec, prec, color="#16A34A", lw=2, where="post",
             label=f"KANFIS (AP = {ap:.4f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right"); plt.tight_layout()
    plt.savefig(f"{output_dir}/pr_curve.png", dpi=150); plt.close()
    print("  [Plot] PR curve saved.")


def _plot_calibration(labels, raw_probs, cal_probs, output_dir, T: float):
    """
    IMP 2 — Shows both raw and calibrated curves in the same figure.
    Makes the calibration improvement visible and citable in the paper.
    """
    frac_raw, mean_raw = calibration_curve(labels, raw_probs, n_bins=10)
    frac_cal, mean_cal = calibration_curve(labels, cal_probs,  n_bins=10)

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
    plt.plot(mean_raw, frac_raw, "s--", color="#94A3B8",
             label=f"KANFIS (raw, T=1.00)", alpha=0.7)
    plt.plot(mean_cal, frac_cal, "s-",  color="#DC2626",
             label=f"KANFIS (calibrated, T={T:.2f})")
    plt.xlabel("Mean Predicted Probability"); plt.ylabel("Fraction Positives")
    plt.title("Calibration Curve — Raw vs Temperature-Scaled")
    plt.legend(); plt.tight_layout()
    plt.savefig(f"{output_dir}/calibration.png", dpi=150); plt.close()
    print("  [Plot] Calibration curve saved (raw + calibrated).")


def _plot_rule_weights(model: KANFIS, output_dir: str):
    weights = model.rule_head.rule_weights.weight.detach().cpu().numpy().squeeze(0)
    indices = np.arange(len(weights))
    colors  = ["#2563EB" if w > 0 else "#DC2626" for w in weights]
    plt.figure(figsize=(10, 4))
    plt.bar(indices, weights, color=colors, edgecolor="white", linewidth=0.5)
    plt.axhline(0, color="black", lw=0.8)
    plt.xlabel("Rule Index"); plt.ylabel("Consequent Weight")
    plt.title("KANFIS Rule Weights (after L1 Pruning)\nBlue=Diabetic Risk, Red=Protective")
    blue_p = mpatches.Patch(color="#2563EB", label="Positive (diabetic risk)")
    red_p  = mpatches.Patch(color="#DC2626", label="Negative (protective)")
    plt.legend(handles=[blue_p, red_p])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rule_weights.png", dpi=150); plt.close()
    print("  [Plot] Rule weight chart saved.")