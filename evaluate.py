"""
evaluate.py  (v3 — sensitivity-focused + interpretability fix)
===========
Phase 4: Clinical Validation and Narrative Generation.

CHANGELOG vs v2:
  FIX 1 (CRITICAL) — Rule extraction completely rewritten.
           v2 indexed into fuzzy-layer centres using a hardcoded g_out=16
           offset, mapping to a latent bottleneck space. This produced
           linguistically meaningless narratives.

           v3: IT2FuzzyLayer now operates in feature space (kanfis_model.py
           FIX 1), so centres[r, f] IS the Z-score value of feature f
           where rule r fires most strongly. Rule extraction now iterates
           directly over (feature_index, feature_name) pairs.

  IMP 1 — full_evaluation accepts opt_threshold (from train.py IMP 2).
           All classification metrics (F1, F2, sensitivity, specificity,
           precision, confusion matrix) are reported at BOTH the default
           0.5 threshold and the sensitivity-optimised threshold so the
           paper can justify the clinical threshold choice.

  IMP 2 — Sensitivity Operating Point table added to the clinical report:
           Prints (threshold, sensitivity, specificity, precision, F2)
           for thresholds 0.3, 0.35, 0.4, 0.45, 0.5 so reviewers can see
           the full sensitivity/specificity trade-off curve in numeric form.

  IMP 3 — Calibration plot updated: labels the optimal threshold line.
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
    f1_score, fbeta_score, recall_score, precision_score,
)
from sklearn.calibration import calibration_curve
from kanfis_model import KANFIS, TemperatureScaling


# ─────────────────────────────────────────────
# LINGUISTIC VARIABLE MAPPING  (Z-score → clinical term)
# ─────────────────────────────────────────────
LINGUISTIC_MAP = {
    # DiaBD features
    "glucose":              {(-4, -1.5): "Normal",       (-1.5, 0.5): "Elevated",       (0.5, 4): "Critically Elevated"},
    "FastingGlucose":       {(-4, -1.5): "Normal",       (-1.5, 0.5): "Elevated",       (0.5, 4): "Critically Elevated"},
    "RandomGlucose":        {(-4, -1.5): "Normal",       (-1.5, 0.5): "Moderately High", (0.5, 4): "Very High"},
    "bmi":                  {(-4, -1.0): "Underweight",  (-1.0, 1.0): "Normal",          (1.0, 4): "Obese"},
    "BMI":                  {(-4, -1.0): "Underweight",  (-1.0, 1.0): "Normal",          (1.0, 4): "Obese"},
    "Insulin":              {(-4, -0.5): "Low",          (-0.5, 0.5): "Normal",          (0.5, 4): "High"},
    "systolic_bp":          {(-4, -1.0): "Normal",       (-1.0, 1.0): "Elevated",       (1.0, 4): "Hypertensive"},
    "SystolicBP":           {(-4, -1.0): "Normal",       (-1.0, 1.0): "Elevated",       (1.0, 4): "Hypertensive"},
    "diastolic_bp":         {(-4, -1.0): "Normal",       (-1.0, 1.0): "Moderately High", (1.0, 4): "Severely High"},
    "DiastolicBP":          {(-4, -1.0): "Normal",       (-1.0, 1.0): "Moderately High", (1.0, 4): "Severely High"},
    "hypertensive":         {(-4,  0.5): "Absent",       (0.5, 4): "Present"},
    "Hypertension":         {(-4,  0.5): "Absent",       (0.5, 4): "Present"},
    "stroke":               {(-4,  0.5): "No History",   (0.5, 4): "History Present"},
    "IschemicStroke":       {(-4,  0.5): "No History",   (0.5, 4): "History Present"},
    "cardiovascular_disease":{(-4, 0.5): "No History",   (0.5, 4): "History Present"},
    "HeartDisease":         {(-4,  0.5): "No History",   (0.5, 4): "History Present"},
    "age":                  {(-4, -1.0): "Young",        (-1.0, 1.0): "Middle-Aged",    (1.0, 4): "Elderly"},
    "Age":                  {(-4, -1.0): "Young",        (-1.0, 1.0): "Middle-Aged",    (1.0, 4): "Elderly"},
    "gender":               {(-4,  0.0): "Female",       (0.0, 4): "Male"},
    "VascularRiskScore":    {(-4,  0.0): "Low Risk",     (0.0, 1.5): "Moderate Risk",   (1.5, 4): "High Risk"},
    "MetabolicBurden":      {(-4, -0.5): "Low",          (-0.5, 0.5): "Moderate",       (0.5, 4): "High"},
    "CardioFlag":           {(-4,  0.5): "No Comorbidities", (0.5, 4): "Comorbidities Present"},
    # PIDD features
    "Glucose":              {(-4, -1.5): "Normal",       (-1.5, 0.5): "Elevated",       (0.5, 4): "Critically Elevated"},
    "BloodPressure":        {(-4, -1.0): "Normal",       (-1.0, 1.0): "Elevated",       (1.0, 4): "Hypertensive"},
    "SkinThickness":        {(-4, -0.5): "Thin",         (-0.5, 0.5): "Normal",         (0.5, 4): "Thick"},
    "DiabetesPedigreeFunction": {(-4, -0.5): "Low",      (-0.5, 0.5): "Moderate",       (0.5, 4): "High"},
    "Pregnancies":          {(-4, -0.5): "None/Few",     (-0.5, 0.5): "Moderate",       (0.5, 4): "Many"},
}

DEFAULT_TERM = "Elevated"


def _z_to_linguistic(feature_name: str, z_value: float) -> str:
    mapping = LINGUISTIC_MAP.get(feature_name, {})
    for (low, high), term in mapping.items():
        if low <= z_value < high:
            return term
    return DEFAULT_TERM


# ─────────────────────────────────────────────
# 1.  RULE EXTRACTION  (FIX 1: correct feature-space extraction)
# ─────────────────────────────────────────────
def extract_rules(
    model: KANFIS,
    feature_names: list,
    active_threshold: float = 0.01,
) -> list:
    """
    FIX 1 — Rule extraction now operates correctly in feature space.
    """
    centres = model.fuzzy_layer.centres.detach().cpu().numpy()
    
    # FIX: Calculate effective rule weights for the 2-layer MLP rule head
    W_hid = model.rule_head.hidden.weight.detach().cpu().numpy()
    W_out = model.rule_head.output.weight.detach().cpu().numpy()
    weights = np.dot(W_out, W_hid).squeeze(0)

    active_rule_indices = [i for i, w in enumerate(weights) if abs(w) >= active_threshold]

    rules = []
    for r_idx in active_rule_indices:
        rule_centres = centres[r_idx]   
        weight_val   = float(weights[r_idx])

        antecedents = []
        for f_idx, fname in enumerate(feature_names):
            if f_idx >= len(rule_centres):
                break
            z_val = float(rule_centres[f_idx])
            
            if abs(z_val) > 0.5:
                term = _z_to_linguistic(fname, z_val)
                antecedents.append((fname, term, z_val))

        if not antecedents and len(feature_names) > 0:
            strongest_f = int(np.argmax(np.abs(rule_centres[:len(feature_names)])))
            z_val = float(rule_centres[strongest_f])
            term  = _z_to_linguistic(feature_names[strongest_f], z_val)
            antecedents.append((feature_names[strongest_f], term, z_val))

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
def generate_narratives(rules: list, top_k: int = 7) -> list:
    narratives = []
    for i, rule in enumerate(rules[:top_k]):
        conditions = " AND\n       ".join(
            f"{feat} is {term}" for feat, term, _ in rule["antecedents"]
        )
        direction  = "HIGH" if rule["weight"] > 0 else "PROTECTIVE / LOW"
        confidence = abs(rule["weight"])
        narrative  = (
            f"Rule {i+1}:\n"
            f"  IF   {conditions}\n"
            f"  THEN Risk of Type-2 Diabetes is {direction}\n"
            f"       (Weight = {rule['weight']:+.4f}, |Confidence| ∝ {confidence:.4f})\n"
        )
        narratives.append(narrative)
    return narratives


def print_clinical_report(
    model: KANFIS,
    feature_names: list,
    top_k: int = 7,
) -> None:
    print("\n" + "═"*65)
    print("  KANFIS — Extracted Clinical Decision Rules (v3 — FIX 1)")
    print("═"*65)

    rules      = extract_rules(model, feature_names)
    narratives = generate_narratives(rules, top_k)

    active = model.get_active_rules()
    n_pos  = sum(1 for r in rules if r["weight"] > 0)
    n_neg  = len(rules) - n_pos

    print(f"  Total active rules  : {len(active)} (after L1 pruning)")
    print(f"  Rule polarity       : {n_pos} risk-positive / {n_neg} protective")
    if n_neg > 2 * n_pos:
        print("  ⚠ WARNING: Protective rules dominate. "
              "Consider increasing alpha_pos or focal_gamma.")
    print()

    for n in narratives:
        print(n)


# ─────────────────────────────────────────────
# 3.  SENSITIVITY OPERATING POINT TABLE  (IMP 2)
# ─────────────────────────────────────────────
def print_sensitivity_operating_points(
    labels: np.ndarray,
    probs: np.ndarray,
    thresholds: list = None,
    opt_threshold: float = None,
) -> None:
    """
    IMP 2 — Print a table of sensitivity, specificity, precision, F2
    at a range of decision thresholds.

    This is the core sensitivity analysis table for the paper:
    reviewers can see exactly what clinical trade-off each threshold
    implies, and the optimal threshold is highlighted.
    """
    if thresholds is None:
        thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    print("\n  ── Sensitivity Operating Point Analysis ──────────────")
    print(f"  {'Threshold':>10} {'Sensitivity':>12} {'Specificity':>12} "
          f"{'Precision':>10} {'F2-Score':>9}  {'Note'}")
    print(f"  {'─'*70}")

    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        sens  = recall_score(labels, preds, pos_label=1, zero_division=0)
        spec  = recall_score(labels, preds, pos_label=0, zero_division=0)
        prec  = precision_score(labels, preds, zero_division=0)
        f2    = fbeta_score(labels, preds, beta=2, zero_division=0)
        note  = " ← optimal" if (opt_threshold is not None
                                  and abs(thr - opt_threshold) < 0.025) else ""
        print(f"  {thr:>10.2f} {sens:>12.4f} {spec:>12.4f} "
              f"{prec:>10.4f} {f2:>9.4f} {note}")

    # Also print the exact optimal threshold row if not already shown
    if opt_threshold is not None and opt_threshold not in thresholds:
        preds = (probs >= opt_threshold).astype(int)
        sens  = recall_score(labels, preds, pos_label=1, zero_division=0)
        spec  = recall_score(labels, preds, pos_label=0, zero_division=0)
        prec  = precision_score(labels, preds, zero_division=0)
        f2    = fbeta_score(labels, preds, beta=2, zero_division=0)
        print(f"  {opt_threshold:>10.3f} {sens:>12.4f} {spec:>12.4f} "
              f"{prec:>10.4f} {f2:>9.4f}  ← optimal")

    print(f"  {'─'*70}")


# ─────────────────────────────────────────────
# 4.  FULL EVALUATION SUITE  (IMP 1: dual threshold reporting)
# ─────────────────────────────────────────────
@torch.no_grad()
def full_evaluation(
    model: KANFIS,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    ts: "TemperatureScaling | None" = None,
    opt_threshold: float = None,      # IMP 1: sensitivity-optimised threshold
    save_plots: bool = True,
    output_dir: str = ".",
) -> dict:
    """
    IMP 1 — Reports metrics at both threshold=0.5 and the sensitivity-
    constrained optimal threshold from train.py.

    When opt_threshold is provided:
      - Main metrics table shows both thresholds side by side
      - Calibration plot marks the optimal operating point
      - Sensitivity Operating Point table (IMP 2) covers the full range
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

    # Apply temperature scaling if provided
    if ts is not None:
        T = ts.temperature.item()
        cal_logits = raw_logits / max(T, 0.05)
    else:
        T          = 1.0
        cal_logits = raw_logits

    raw_probs = 1 / (1 + np.exp(-raw_logits))
    probs     = 1 / (1 + np.exp(-cal_logits))

    # If no opt_threshold passed, default to 0.5
    thr = opt_threshold if opt_threshold is not None else 0.5

    # ── Compute metrics at threshold=0.5 (baseline) ──────────────────────
    preds_05  = (probs >= 0.5).astype(int)
    auc       = roc_auc_score(labels, probs)
    ap        = average_precision_score(labels, probs)
    f1_05     = f1_score(labels, preds_05, zero_division=0)
    f2_05     = fbeta_score(labels, preds_05, beta=2, zero_division=0)
    sens_05   = recall_score(labels, preds_05, pos_label=1, zero_division=0)
    spec_05   = recall_score(labels, preds_05, pos_label=0, zero_division=0)
    prec_05   = precision_score(labels, preds_05, zero_division=0)
    cm_05     = confusion_matrix(labels, preds_05)

    # ── Compute metrics at sensitivity-optimised threshold ────────────────
    preds_opt = (probs >= thr).astype(int)
    f1_opt    = f1_score(labels, preds_opt, zero_division=0)
    f2_opt    = fbeta_score(labels, preds_opt, beta=2, zero_division=0)
    sens_opt  = recall_score(labels, preds_opt, pos_label=1, zero_division=0)
    spec_opt  = recall_score(labels, preds_opt, pos_label=0, zero_division=0)
    prec_opt  = precision_score(labels, preds_opt, zero_division=0)
    cm_opt    = confusion_matrix(labels, preds_opt)

    # ── Print report ──────────────────────────────────────────────────────
    print("\n" + "═"*65)
    print("  KANFIS — Evaluation Report  (v3)")
    print("═"*65)
    cal_note = f" (T={T:.3f} applied)" if ts is not None else " (uncalibrated)"
    print(f"  ROC-AUC    : {auc:.4f}{cal_note}")
    print(f"  Avg Prec.  : {ap:.4f}")

    print(f"\n  ── At threshold = 0.50 (default) ──────────────────────")
    print(f"  F1-Score   : {f1_05:.4f}")
    print(f"  F2-Score   : {f2_05:.4f}")
    print(f"  Sensitivity: {sens_05:.4f}  (recall for diabetic class)")
    print(f"  Specificity: {spec_05:.4f}  (recall for non-diabetic class)")
    print(f"  Precision  : {prec_05:.4f}")
    print(f"  Confusion  :\n{cm_05}")

    print(f"\n  ── At threshold = {thr:.3f} (sensitivity-optimised) ────────")
    print(f"  F1-Score   : {f1_opt:.4f}")
    print(f"  F2-Score   : {f2_opt:.4f}")
    print(f"  Sensitivity: {sens_opt:.4f}  ← target ≥ 0.75")
    print(f"  Specificity: {spec_opt:.4f}")
    print(f"  Precision  : {prec_opt:.4f}")
    print(f"  Confusion  :\n{cm_opt}")

    # ── IMP 2: Full operating point table ────────────────────────────────
    print_sensitivity_operating_points(
        labels, probs,
        thresholds=[0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        opt_threshold=thr,
    )

    # ── Pillar 2: Interpretability ────────────────────────────────────────
    import time
    t0 = time.perf_counter()
    active_rules  = model.get_active_rules()
    rule_time_ms  = (time.perf_counter() - t0) * 1000
    print(f"\n  Active rules after pruning : {len(active_rules)}")
    print(f"  Rule inference time        : {rule_time_ms:.3f} ms")

    # ── FIX 1: Correct clinical narrative ────────────────────────────────
    print_clinical_report(model, feature_names)

    # ── Plots ─────────────────────────────────────────────────────────────
    if save_plots:
        _plot_roc(labels, probs, auc, output_dir, opt_threshold=thr)
        _plot_pr(labels, probs, ap, output_dir)
        _plot_calibration(labels, raw_probs, probs, output_dir, T,
                          opt_threshold=thr)      # IMP 3
        _plot_rule_weights(model, output_dir)
        _plot_sensitivity_curve(labels, probs, output_dir, opt_threshold=thr)

    return {
        # Threshold-agnostic
        "auc": auc, "ap": ap,
        # At 0.5
        "f1_05": f1_05, "f2_05": f2_05,
        "sensitivity_05": sens_05, "specificity_05": spec_05,
        "precision_05": prec_05, "cm_05": cm_05,
        # At optimal threshold
        "threshold_opt": thr,
        "f1_opt": f1_opt, "f2_opt": f2_opt,
        "sensitivity_opt": sens_opt, "specificity_opt": spec_opt,
        "precision_opt": prec_opt, "cm_opt": cm_opt,
        # Other
        "n_active_rules": len(active_rules),
        "rule_inference_ms": rule_time_ms,
        "temperature": T,
    }


# ─────────────────────────────────────────────
# 5.  CROSS-POPULATION VALIDATION
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

    auc_pidd_on_diabd   = _get_auc(model_pidd,   X_diabd, y_diabd)
    auc_kanfis_on_diabd = _get_auc(model_kanfis,  X_diabd, y_diabd)
    delta = auc_kanfis_on_diabd - auc_pidd_on_diabd

    print("\n" + "═"*65)
    print("  Cross-Population Validation (Pima Bias Quantification)")
    print("═"*65)
    print(f"  PIDD-trained model on DiaBD  : AUC = {auc_pidd_on_diabd:.4f}")
    print(f"  KANFIS (DiaBD-trained)       : AUC = {auc_kanfis_on_diabd:.4f}")
    print(f"  Improvement (ΔAUC)           : {delta:+.4f}")
    if delta > 0:
        print("  ✓ KANFIS demonstrates improved cross-population generalizability")
    else:
        print("  ✗ Further training or feature adjustment required")


# ─────────────────────────────────────────────
# 6.  PLOT HELPERS
# ─────────────────────────────────────────────
def _plot_roc(labels, probs, auc, output_dir, opt_threshold=None):
    fpr, tpr, thresholds = roc_curve(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="#2563EB", lw=2, label=f"KANFIS (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)

    # Mark the operating point at the optimal threshold
    if opt_threshold is not None:
        preds = (probs >= opt_threshold).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        tn = np.sum((preds == 0) & (labels == 0))
        sens_pt = tp / (tp + fn + 1e-9)
        spec_pt = tn / (tn + fp + 1e-9)
        plt.scatter([1 - spec_pt], [sens_pt], color="#DC2626", zorder=5, s=80,
                    label=f"Threshold={opt_threshold:.3f} "
                          f"(sens={sens_pt:.3f}, spec={spec_pt:.3f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — KANFIS Diabetes Diagnostics")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/roc_curve.png", dpi=150)
    plt.close()
    print("  [Plot] ROC curve saved.")


def _plot_pr(labels, probs, ap, output_dir):
    prec, rec, _ = precision_recall_curve(labels, probs)
    plt.figure(figsize=(6, 5))
    plt.step(rec, prec, color="#16A34A", lw=2, where="post",
             label=f"KANFIS (AP = {ap:.4f})")
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pr_curve.png", dpi=150)
    plt.close()
    print("  [Plot] PR curve saved.")


def _plot_calibration(labels, raw_probs, cal_probs, output_dir,
                      T: float, opt_threshold=None):
    """IMP 3 — marks the optimal operating threshold."""
    frac_raw, mean_raw = calibration_curve(labels, raw_probs, n_bins=10)
    frac_cal, mean_cal = calibration_curve(labels, cal_probs,  n_bins=10)

    plt.figure(figsize=(6, 5))
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
    plt.plot(mean_raw, frac_raw, "s--", color="#94A3B8",
             label=f"KANFIS raw (T=1.00)", alpha=0.7)
    plt.plot(mean_cal, frac_cal, "s-",  color="#DC2626",
             label=f"KANFIS calibrated (T={T:.2f})")

    if opt_threshold is not None:
        plt.axvline(x=opt_threshold, color="#F59E0B", lw=1.5, linestyle=":",
                    label=f"Opt. threshold={opt_threshold:.3f}")

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction Positives")
    plt.title("Calibration Curve — Raw vs Temperature-Scaled")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/calibration.png", dpi=150)
    plt.close()
    print("  [Plot] Calibration curve saved.")


def _plot_rule_weights(model: KANFIS, output_dir: str):
    # FIX: Calculate effective rule weights for the 2-layer MLP rule head
    W_hid = model.rule_head.hidden.weight.detach().cpu().numpy()
    W_out = model.rule_head.output.weight.detach().cpu().numpy()
    weights = np.dot(W_out, W_hid).squeeze(0)
    
    indices = np.arange(len(weights))
    colors  = ["#2563EB" if w > 0 else "#DC2626" for w in weights]
    
    plt.figure(figsize=(10, 4))
    plt.bar(indices, weights, color=colors, edgecolor="white", linewidth=0.5)
    plt.axhline(0, color="black", lw=0.8)
    plt.xlabel("Rule Index")
    plt.ylabel("Consequent Weight")
    plt.title("KANFIS Rule Weights (after L1 Pruning)\nBlue=Diabetic Risk, Red=Protective")
    blue_p = mpatches.Patch(color="#2563EB", label="Positive (diabetic risk)")
    red_p  = mpatches.Patch(color="#DC2626", label="Negative (protective)")
    plt.legend(handles=[blue_p, red_p])
    plt.tight_layout()
    plt.savefig(f"{output_dir}/rule_weights.png", dpi=150)
    plt.close()
    print("  [Plot] Rule weights saved.")


def _plot_sensitivity_curve(labels, probs, output_dir, opt_threshold=None):
    """Plot sensitivity and specificity vs threshold — the clinical trade-off curve."""
    thresholds = np.linspace(0.1, 0.9, 100)
    sensitivities, specificities = [], []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        sensitivities.append(recall_score(labels, preds, pos_label=1, zero_division=0))
        specificities.append(recall_score(labels, preds, pos_label=0, zero_division=0))

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds, sensitivities, color="#DC2626", lw=2, label="Sensitivity (TPR)")
    plt.plot(thresholds, specificities, color="#2563EB", lw=2, label="Specificity (TNR)")
    plt.axhline(0.75, color="#DC2626", lw=1, linestyle=":", alpha=0.6,
                label="Target sensitivity = 0.75")

    if opt_threshold is not None:
        plt.axvline(opt_threshold, color="#F59E0B", lw=1.5, linestyle="--",
                    label=f"Optimal threshold = {opt_threshold:.3f}")

    plt.xlabel("Decision Threshold")
    plt.ylabel("Rate")
    plt.title("Sensitivity vs Specificity by Threshold\n(Clinical Operating Point Selection)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/sensitivity_curve.png", dpi=150)
    plt.close()
    print("  [Plot] Sensitivity/Specificity curve saved.")