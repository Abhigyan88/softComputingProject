"""
inference.py — KANFIS Per-Patient Inference & Clinical Narrative Generator
===========================================================================
Usage:
    python inference.py \
        --model    results/kanfis_final.pt \
        --scaler   results/scaler.pkl \
        --features results/feature_names.json \
        --csv      data/test_patients.csv \
        --threshold 0.4
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
import torch
import joblib
import os

try:
    from kanfis_model import KANFIS, build_kanfis
except ImportError:
    sys.exit(
        "ERROR: Cannot import kanfis_model.py. "
        "Run this script from the project root folder."
    )


# ──────────────────────────────────────────────────────────────────────────
# 1.  LINGUISTIC LABEL MAP
# ──────────────────────────────────────────────────────────────────────────
LINGUISTIC_MAP = {
    "glucose":               {(-4, -1.5): "Normal",             (-1.5, 0.5): "Elevated",               (0.5, 4): "Critically Elevated"},
    "log_glucose":           {(-4, -1.5): "Normal",             (-1.5, 0.5): "Slightly Elevated",       (0.5, 4): "Critically Elevated"},
    "bmi":                   {(-4, -1.0): "Underweight",        (-1.0, 1.0): "Normal Weight",           (1.0, 4): "Obese"},
    "log_bmi":               {(-4, -1.0): "Underweight",        (-1.0, 1.0): "Normal Weight",           (1.0, 4): "Obese"},
    "GlucoseBMI":            {(-4, -0.5): "Low Metabolic Load", (-0.5, 0.5): "Moderate",                (0.5, 4): "High Metabolic Load"},
    "MetabolicBurden":       {(-4, -0.5): "Low",                (-0.5, 0.5): "Moderate",                (0.5, 4): "High"},
    "AgeBMI":                {(-4, -0.5): "Young/Lean",         (-0.5, 0.5): "Moderate",                (0.5, 4): "Elderly Obese"},
    "systolic_bp":           {(-4, -1.0): "Normal",             (-1.0, 1.0): "Elevated",                (1.0, 4): "Hypertensive"},
    "diastolic_bp":          {(-4, -1.0): "Normal",             (-1.0, 1.0): "Moderately High",         (1.0, 4): "Severely High"},
    "hypertensive":          {(-4,  0.5): "Absent",             (0.5, 4):    "Present"},
    "HyperGlucose":          {(-4,  0.5): "Low Vascular-Glucose Risk", (0.5, 4): "High Vascular-Glucose Risk"},
    "VascularRiskScore":     {(-4,  0.0): "Low Risk",           (0.0, 1.5):  "Moderate Risk",           (1.5, 4): "High Risk"},
    "stroke":                {(-4,  0.5): "No History",         (0.5, 4):    "History Present"},
    "cardiovascular_disease":{(-4,  0.5): "No History",         (0.5, 4):    "History Present"},
    "CardioFlag":            {(-4,  0.5): "No Comorbidities",   (0.5, 4):    "Comorbidities Present"},
    "family_diabetes":       {(-4,  0.5): "No Family History",  (0.5, 4):    "Family History Present"},
    "family_hypertension":   {(-4,  0.5): "No Family History",  (0.5, 4):    "Family History Present"},
    "age":                   {(-4, -1.0): "Young",              (-1.0, 1.0): "Middle-Aged",             (1.0, 4): "Elderly"},
    "gender":                {(-4,  0.0): "Female",             (0.0, 4):    "Male"},
    "pulse_rate":            {(-4, -1.0): "Low",                (-1.0, 1.0): "Normal",                  (1.0, 4): "Elevated"},
}
DEFAULT_TERM = "Abnormal"


def _z_to_linguistic(feature_name: str, z_value: float) -> str:
    mapping = LINGUISTIC_MAP.get(feature_name, {})
    for (low, high), term in mapping.items():
        if low <= z_value < high:
            return term
    return DEFAULT_TERM


# ──────────────────────────────────────────────────────────────────────────
# 2.  COLUMN NORMALISATION
#     Handles the real DiaBD CSV which has columns like __bmi, __weight
# ──────────────────────────────────────────────────────────────────────────
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace and leading/trailing underscores from column names.
    e.g.  '__bmi'    → 'bmi'
          '__weight' → 'weight'
          ' age '   → 'age'
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.strip("_")
    return df


def resolve_bmi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a 'bmi' column exists.
    Priority:
      1. 'bmi' already present → use it
      2. 'weight' + 'height' present → compute bmi = weight / height²
         (assumes weight in kg, height in metres)
    """
    df = df.copy()
    if "bmi" not in df.columns:
        if "weight" in df.columns and "height" in df.columns:
            h = pd.to_numeric(df["height"], errors="coerce")
            w = pd.to_numeric(df["weight"], errors="coerce")
            # Guard against height accidentally stored in cm
            h = h.where(h < 10, h / 100)   # if > 10, assume cm → convert
            df["bmi"] = w / (h ** 2)
            print("  [Columns] 'bmi' computed from height & weight.")
        else:
            # Last resort: fill with population median so pipeline doesn't crash
            print("  [Columns] WARNING: 'bmi' column not found and cannot be "
                  "computed. Filling with 25.0 (median estimate).")
            df["bmi"] = 25.0
    return df


# ──────────────────────────────────────────────────────────────────────────
# 3.  FEATURE ENGINEERING  (exact mirror of data_preprocessing.py v4)
# ──────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicates engineer_comorbidity_features() + the load_dataset cleaning
    from data_preprocessing.py v4.  Call after normalize_columns().
    """
    df = df.copy()

    # ── Gender encoding ──────────────────────────────────────────────────
    if "gender" in df.columns:
        if df["gender"].dtype == object:
            df["gender"] = (
                df["gender"].astype(str).str.strip().str.lower()
                .map({"female": 0, "male": 1, "f": 0, "m": 1})
                .fillna(0)
            )
        df["gender"] = pd.to_numeric(df["gender"], errors="coerce").fillna(0).astype(float)

    # ── Replace impossible zeros with NaN, then fill with median ────────
    zero_impossible = ["glucose", "systolic_bp", "diastolic_bp", "bmi", "pulse_rate"]
    for col in zero_impossible:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace(0, np.nan)
            df[col] = df[col].fillna(df[col].median())

    # Ensure all remaining raw columns are numeric
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # ── FIX 1 (v4): Log transforms ──────────────────────────────────────
    if "glucose" in df.columns:
        df["log_glucose"] = np.log1p(df["glucose"].fillna(df["glucose"].median()))
    if "bmi" in df.columns:
        df["log_bmi"] = np.log1p(df["bmi"].fillna(df["bmi"].median()))

    # ── FIX 2 (v4): Interaction features ────────────────────────────────
    if "glucose" in df.columns and "bmi" in df.columns:
        df["GlucoseBMI"] = df["glucose"].fillna(0) * df["bmi"].fillna(0) / 100

    if "age" in df.columns and "bmi" in df.columns:
        df["AgeBMI"] = df["age"].fillna(0) * df["bmi"].fillna(0) / 1000

    if "hypertensive" in df.columns and "glucose" in df.columns:
        df["HyperGlucose"] = df["hypertensive"].fillna(0) * df["glucose"].fillna(0)

    # ── v3 features ──────────────────────────────────────────────────────
    if "age" in df.columns and "diastolic_bp" in df.columns:
        df["VascularRiskScore"] = (df["age"] * df["diastolic_bp"]) / 1e3

    if "bmi" in df.columns and "glucose" in df.columns:
        df["MetabolicBurden"] = df["bmi"] * df["glucose"] / 100

    if all(c in df.columns for c in ["cardiovascular_disease", "stroke", "hypertensive"]):
        df["CardioFlag"] = (
            df["cardiovascular_disease"].fillna(0)
            + df["stroke"].fillna(0)
            + df["hypertensive"].fillna(0)
        ).clip(0, 3)

    return df


# ──────────────────────────────────────────────────────────────────────────
# 4.  MODEL LOADER
# ──────────────────────────────────────────────────────────────────────────
def load_model(path: str) -> KANFIS:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        sys.exit(f"ERROR: Could not load model from '{path}'. Details: {e}")

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        # BUG 1 FIX: reconstruct with the correct n_rules.
        # Prefer the stored key; fall back to inferring from the centres tensor shape
        # so checkpoints saved before this fix also load correctly.
        n_rules = ckpt.get(
            "n_rules",
            ckpt["state_dict"]["fuzzy_layer.centres"].shape[0],
        )
        model = build_kanfis(ckpt["n_features"], ckpt["group_map"], n_rules=n_rules)
        model.load_state_dict(ckpt["state_dict"])
        print(f"  [Model]    Loaded — n_features={ckpt['n_features']}, "
              f"n_rules={n_rules}, "
              f"groups={list(ckpt['group_map'].keys())}")
        return model
    elif isinstance(ckpt, KANFIS):
        print(f"  [Model]    Raw KANFIS object loaded from '{path}'")
        return ckpt
    else:
        sys.exit("ERROR: Unrecognised checkpoint format.")


# ──────────────────────────────────────────────────────────────────────────
# 5.  PER-SAMPLE RULE EXTRACTOR
# ──────────────────────────────────────────────────────────────────────────
def explain_sample(
    model: KANFIS,
    x_tensor: torch.Tensor,
    feature_names: list,
    x_z: np.ndarray,
    threshold: float = 0.5,
    top_k_rules: int = 3,
    temperature: float = 1.0,   # BUG 4 FIX: accept temperature from checkpoint
) -> dict:
    model.eval()
    with torch.no_grad():
        logit, firing_strengths = model(x_tensor, return_rules=True)
        # BUG 4 FIX: divide by temperature before sigmoid to match training evaluation.
        # Without this, probabilities diverge from what was reported during training.
        prob = torch.sigmoid(logit / max(temperature, 1e-6)).item()

    firing       = firing_strengths.squeeze(0).cpu().numpy()
    rule_weights = model.rule_head.rule_weights.weight.detach().cpu().numpy().squeeze(0)
    centres      = model.fuzzy_layer.centres.detach().cpu().numpy()

    contributions    = firing * rule_weights
    sorted_rule_idxs = np.argsort(-np.abs(contributions))

    top_rules = []
    for r_idx in sorted_rule_idxs[:top_k_rules]:
        rule_centres = centres[r_idx]
        antecedents  = []

        for f_idx, fname in enumerate(feature_names):
            if f_idx >= len(rule_centres):
                break
            centre_z  = float(rule_centres[f_idx])
            patient_z = float(x_z[f_idx])
            if abs(centre_z) > 0.4 or abs(patient_z) > 0.4:
                antecedents.append({
                    "feature":       fname,
                    "term":          _z_to_linguistic(fname, patient_z),
                    "patient_z":     patient_z,
                    "rule_centre_z": centre_z,
                })

        if not antecedents:
            strongest = int(np.argmax(np.abs(rule_centres[:len(feature_names)])))
            antecedents.append({
                "feature":       feature_names[strongest],
                "term":          _z_to_linguistic(feature_names[strongest], float(x_z[strongest])),
                "patient_z":     float(x_z[strongest]),
                "rule_centre_z": float(rule_centres[strongest]),
            })

        top_rules.append({
            "rule_idx":     int(r_idx),
            "weight":       float(rule_weights[r_idx]),
            "firing":       float(firing[r_idx]),
            "contribution": float(contributions[r_idx]),
            "antecedents":  antecedents,
        })

    return {
        "prob":       prob,
        "prediction": "DIABETIC" if prob >= threshold else "NON-DIABETIC",
        "top_rules":  top_rules,
    }


# ──────────────────────────────────────────────────────────────────────────
# 6.  REPORT FORMATTER
# ──────────────────────────────────────────────────────────────────────────
def format_patient_report(patient_idx: int, raw_row: pd.Series,
                          result: dict, threshold: float) -> str:
    prob       = result["prob"]
    prediction = result["prediction"]
    top_rules  = result["top_rules"]

    filled     = int(round(prob * 20))
    bar        = "█" * filled + "░" * (20 - filled)
    risk_label = (
        "🔴 HIGH RISK"     if prob >= 0.65       else
        "🟡 MODERATE RISK" if prob >= threshold  else
        "🟢 LOW RISK"
    )

    # Try to show bmi from either raw column name
    bmi_val     = raw_row.get("bmi", raw_row.get("__bmi", "?"))
    glucose_val = raw_row.get("glucose", "?")

    lines = [
        "",
        "─" * 68,
        f"  PATIENT {patient_idx + 1}  |  "
        f"Age: {int(raw_row.get('age', 0))}  |  "
        f"Gender: {raw_row.get('gender', '?')}  |  "
        f"Glucose: {glucose_val}  |  "
        f"BMI: {bmi_val}",
        "─" * 68,
        f"  Prediction   : {prediction}",
        f"  Probability  : {prob:.3f}  [{bar}]  {risk_label}",
        f"  Threshold    : {threshold:.2f}",
        "",
        "  ── Fuzzy Rule Explanations (top contributing rules) ──",
    ]

    for i, rule in enumerate(top_rules):
        direction  = "↑ RISK" if rule["weight"] > 0 else "↓ PROTECTIVE"
        conditions = "\n         AND  ".join(
            f"{a['feature']} is {a['term']}" for a in rule["antecedents"]
        )
        lines += [
            "",
            f"  Rule {i+1}  (Rule #{rule['rule_idx']}):",
            f"    IF   {conditions}",
            f"    THEN Diabetes risk is {direction}",
            f"         Rule weight      = {rule['weight']:+.4f}",
            f"         Firing strength  = {rule['firing']:.4f}",
            f"         Net contribution = {rule['contribution']:+.4f}",
        ]

    total_contrib = sum(r["contribution"] for r in top_rules)
    summary = ("These factors collectively INCREASE risk."
               if total_contrib > 0
               else "These factors collectively do NOT support a diabetes diagnosis.")
    lines += ["", f"  Summary: {summary}"]
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────
# 7.  MAIN
# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="KANFIS per-patient inference with fuzzy rule explanations"
    )
    parser.add_argument("--model",     type=str, required=True,
                        help="Path to kanfis_final.pt")
    parser.add_argument("--scaler",    type=str, required=True,
                        help="Path to scaler.pkl saved during training")
    parser.add_argument("--features",  type=str, required=True,
                        help="Path to feature_names.json saved during training")
    parser.add_argument("--all_features", type=str, default=None,
                        help="Path to all_feature_names.json (auto-detected if omitted)")
    parser.add_argument("--csv",       type=str, required=True,
                        help="Path to patient CSV file")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold (check training_history.csv for optimal)")
    parser.add_argument("--top_rules", type=int,   default=3,
                        help="Number of top rules to show per patient")
    args = parser.parse_args()

    print("\n" + "█" * 68)
    print("  KANFIS — Clinical Inference & Narrative Generator")
    print("█" * 68)

    # ── Load model ───────────────────────────────────────────────────────
    model = load_model(args.model)
    model.eval()

    # BUG 1+4 FIX: read temperature and optimal threshold persisted in the checkpoint.
    # Fall back to sensible defaults so old checkpoints still work.
    try:
        ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
        temperature   = float(ckpt.get("temperature",   1.0))
        ckpt_opt_thr  = ckpt.get("opt_threshold", None)
    except Exception:
        temperature, ckpt_opt_thr = 1.0, None

    # CLI --threshold takes priority; otherwise use the checkpoint's optimal threshold.
    effective_threshold = args.threshold
    if args.threshold == 0.5 and ckpt_opt_thr is not None:
        effective_threshold = ckpt_opt_thr
        print(f"  [Threshold] Using checkpoint optimal threshold: {effective_threshold:.3f} "
              f"(override with --threshold)")
    else:
        print(f"  [Threshold] {effective_threshold:.3f} (from --threshold flag)")

    print(f"  [Temperature] T={temperature:.4f}")

    # ── Load scaler ──────────────────────────────────────────────────────
    try:
        scaler = joblib.load(args.scaler)
        print(f"  [Scaler]   Loaded from '{args.scaler}'")
    except Exception as e:
        sys.exit(f"ERROR: Could not load scaler from '{args.scaler}'. Details: {e}")

    # ── Load feature names ───────────────────────────────────────────────
    try:
        with open(args.features) as f:
            feature_names = json.load(f)
        print(f"  [Features] {len(feature_names)} selected: {feature_names}")
    except Exception as e:
        sys.exit(f"ERROR: Could not load feature names from '{args.features}'. Details: {e}")

    # ── Load all_feature_names (full pre-selection list the scaler expects) ──
    # SCALER FIX: the scaler was fitted on ALL engineered features, not just the
    # selected subset.  We need the full ordered list to build the complete matrix
    # before calling scaler.transform(), then slice to the selected columns.
    all_features_path = args.all_features
    if all_features_path is None:
        # Auto-detect: look for all_feature_names.json next to feature_names.json
        candidate = os.path.join(os.path.dirname(args.features), "all_feature_names.json")
        if os.path.exists(candidate):
            all_features_path = candidate
    if all_features_path and os.path.exists(all_features_path):
        try:
            with open(all_features_path) as f:
                all_feature_names = json.load(f)
            print(f"  [AllFeatures] {len(all_feature_names)} total (scaler expects these)")
        except Exception as e:
            sys.exit(f"ERROR: Could not load all_feature_names from '{all_features_path}'. Details: {e}")
    else:
        # Fallback for checkpoints produced before this fix: assume scaler was
        # fitted only on the selected features (old incorrect behaviour).
        all_feature_names = None
        print("  [AllFeatures] all_feature_names.json not found — "
              "assuming scaler was fitted on selected features only.\n"
              "  Re-run training to generate all_feature_names.json and fix "
              "any scaler dimension mismatch.")

    if len(feature_names) != model.n_features:
        sys.exit(
            f"ERROR: feature_names.json has {len(feature_names)} entries but "
            f"model expects {model.n_features}. "
            f"Both files must be from the same training run."
        )

    # ── Load CSV ─────────────────────────────────────────────────────────
    try:
        df_raw = pd.read_csv(args.csv)
    except Exception as e:
        sys.exit(f"ERROR: Cannot read CSV '{args.csv}'. Details: {e}")

    print(f"\n  Loaded {len(df_raw)} patient(s).")
    print(f"  Raw columns: {list(df_raw.columns)}")

    # ── Normalise column names ───────────────────────────────────────────
    # Strips whitespace AND leading/trailing underscores so that
    # '__bmi' → 'bmi',  '__weight' → 'weight',  etc.
    df_clean = normalize_columns(df_raw)
    print(f"  Clean columns: {list(df_clean.columns)}")

    # ── Resolve bmi ──────────────────────────────────────────────────────
    df_clean = resolve_bmi(df_clean)

    # ── Feature engineering ──────────────────────────────────────────────
    df_eng = engineer_features(df_clean)

    # BUG 2 FIX: detect glucose unit mismatch.
    # diabd.csv stores glucose as HbA1c % (typical range 4–14).
    # test_patients.csv uses blood glucose mg/dL (typical range 70–400).
    # The scaler was fitted on HbA1c; passing mg/dL values through it produces
    # z-scores in the range +15σ to +100σ, making all predictions meaningless.
    if "glucose" in df_eng.columns:
        gmax = df_eng["glucose"].dropna().max()
        if gmax > 30:
            print(
                f"\n  *** WARNING: 'glucose' max value is {gmax:.1f}, which looks like "
                f"blood glucose mg/dL.\n"
                f"  ***          The scaler expects HbA1c % (typical range 4–14).\n"
                f"  ***          Predictions will be INVALID unless you convert:\n"
                f"  ***          HbA1c ≈ (mg/dL + 46.7) / 28.7   (DCCT approximation)\n"
                f"  ***          or retrain the model on mg/dL data.\n"
            )

    # ── Select exactly the features used during training ─────────────────
    missing = [f for f in feature_names if f not in df_eng.columns]
    if missing:
        sys.exit(
            f"ERROR: The following features from feature_names.json could not "
            f"be found or engineered from the CSV:\n  {missing}\n"
            f"Available columns after engineering: {list(df_eng.columns)}"
        )

    # ── Select and scale features ────────────────────────────────────────
    # SCALER FIX: the scaler was fitted on all engineered features before
    # XGB/RFECV selection.  We must:
    #   1. Build the complete engineered matrix (all_feature_names columns)
    #   2. Transform with the scaler (correct dimensionality)
    #   3. Slice out only the selected feature_names columns
    if all_feature_names is not None:
        missing_all = [f for f in all_feature_names if f not in df_eng.columns]
        if missing_all:
            sys.exit(
                f"ERROR: The following features expected by the scaler are missing "
                f"from the engineered CSV:\n  {missing_all}\n"
                f"Available columns: {list(df_eng.columns)}"
            )
        X_all_raw = df_eng[all_feature_names].values.astype(np.float32)
        if np.isnan(X_all_raw).any():
            X_all_raw = np.nan_to_num(X_all_raw, nan=0.0)
        X_all_z = scaler.transform(X_all_raw).astype(np.float32)
        # Slice to the selected feature columns
        selected_indices = [all_feature_names.index(f) for f in feature_names]
        X_z = X_all_z[:, selected_indices]
    else:
        # Legacy path (no all_feature_names.json) — may raise ValueError if
        # scaler was fitted on more features than feature_names contains.
        X_raw = df_eng[feature_names].values.astype(np.float32)
        if np.isnan(X_raw).any():
            nan_cols = [feature_names[i] for i in range(X_raw.shape[1])
                        if np.isnan(X_raw[:, i]).any()]
            print(f"  WARNING: NaN values found in {nan_cols}. Filling with 0.")
            X_raw = np.nan_to_num(X_raw, nan=0.0)
        X_z = scaler.transform(X_raw).astype(np.float32)

    # ── Inference loop ───────────────────────────────────────────────────
    print(f"\n{'█' * 68}")
    print("  PER-PATIENT RESULTS")
    print(f"{'█' * 68}")

    results_summary = []

    for i in range(len(df_raw)):
        x_tensor = torch.FloatTensor(X_z[i:i+1])
        raw_row  = df_raw.iloc[i]   # use original for display

        result = explain_sample(
            model, x_tensor, feature_names, X_z[i],
            threshold=effective_threshold,
            top_k_rules=args.top_rules,
            temperature=temperature,      # BUG 4 FIX
        )

        print(format_patient_report(i, raw_row, result, effective_threshold))

        gt = str(raw_row.get("diabetic", "?")).strip().lower()
        gt_label = ("DIABETIC"     if gt in ("yes", "1", "true")  else
                    "NON-DIABETIC" if gt in ("no",  "0", "false") else "?")
        results_summary.append({
            "patient":      i + 1,
            "age":          raw_row.get("age",     "?"),
            "gender":       raw_row.get("gender",  "?"),
            "glucose":      raw_row.get("glucose", "?"),
            "prob":         f"{result['prob']:.3f}",
            "prediction":   result["prediction"],
            "ground_truth": gt_label,
            "correct":      "✓" if result["prediction"] == gt_label else "✗",
        })

    # ── Summary table ────────────────────────────────────────────────────
    print("\n\n" + "═" * 68)
    print("  SUMMARY TABLE")
    print("═" * 68)
    print(f"  {'Pt':>3}  {'Age':>4}  {'Sex':<8}  {'Glucose':>8}  "
          f"{'Prob':>6}  {'Prediction':<14}  {'GT':<14}  {'OK':>3}")
    print("  " + "─" * 65)
    for r in results_summary:
        print(
            f"  {r['patient']:>3}  {str(r['age']):>4}  {str(r['gender']):<8}  "
            f"{str(r['glucose']):>8}  {r['prob']:>6}  "
            f"{r['prediction']:<14}  {r['ground_truth']:<14}  {r['correct']:>3}"
        )

    n_correct  = sum(1 for r in results_summary if r["correct"] == "✓")
    n_labelled = sum(1 for r in results_summary if r["ground_truth"] != "?")
    if n_labelled:
        print(f"\n  Accuracy on labelled rows: {n_correct}/{n_labelled} "
              f"({100 * n_correct / n_labelled:.1f}%)  "
              f"[threshold={effective_threshold:.3f}, T={temperature:.4f}]")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()