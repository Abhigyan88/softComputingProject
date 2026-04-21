"""
main.py
=======
Orchestrates the complete KANFIS research pipeline:

  Phase 1 → Data Harmonization & Preprocessing
  Phase 2 → Model Construction
  Phase 3 → Training with Sparse Pruning
  Phase 4 → Clinical Validation & Narrative Generation

Usage:
  python main.py --diabd  path/to/diabd.csv
  python main.py --diabd  path/to/diabd.csv  --pidd path/to/pima.csv  --cross_pop
  python main.py --diabd  path/to/diabd.csv  --ablation
"""

import argparse
import os
import sys
import torch
import numpy as np

from data_preprocessing import run_preprocessing_pipeline, DIABD_COLS, PIDD_COLS
from kanfis_model import build_kanfis
from train import train_kanfis, cross_validate_kanfis, run_ablation_study, save_model
from evaluate import full_evaluation, cross_population_test


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="KANFIS — Kolmogorov-Arnold Neuro-Fuzzy Inference System for Diabetes Diagnostics"
    )
    p.add_argument("--diabd",      type=str, required=True,
                   help="Path to DiaBD CSV file (primary dataset)")
    p.add_argument("--pidd",       type=str, default=None,
                   help="Path to PIDD CSV (optional, for cross-population test)")
    p.add_argument("--cross_pop",  action="store_true",
                   help="Run cross-population Pima Bias validation")
    p.add_argument("--ablation",   action="store_true",
                   help="Run ablation study (KANFIS vs DNN, RF, XGBoost)")
    p.add_argument("--cv",         action="store_true",
                   help="Run 5-fold cross-validation")
    p.add_argument("--epochs",     type=int, default=150)
    p.add_argument("--n_rules",    type=int, default=10)
    p.add_argument("--l1_lambda",  type=float, default=1e-3)
    p.add_argument("--output_dir", type=str, default="./outputs")
    return p.parse_args()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "█"*65)
    print("  KANFIS — Diabetes Diagnostics")
    print("  Abhigyan Pandey (22075001) & Shivansh Kandpal (22075083)")
    print("█"*65)

    # ══════════════════════════════════════════════
    # PHASE 1 — PREPROCESSING (DiaBD primary)
    # ══════════════════════════════════════════════
    print("\n▶ Phase 1: Data Harmonisation & Preprocessing")
    data = run_preprocessing_pipeline(
        csv_path=args.diabd,
        schema=DIABD_COLS,
        balance_strategy="smote_tomek",
        do_feature_engineering=True,
        do_feature_selection=True,
    )

    X_train       = data["X_train"]
    y_train       = data["y_train"]
    X_test        = data["X_test"]
    y_test        = data["y_test"]
    feature_names = data["feature_names"]
    group_map     = data["group_map"]

    # ══════════════════════════════════════════════
    # PHASE 2+3 — TRAINING
    # ══════════════════════════════════════════════
    print("\n▶ Phase 2+3: Architectural Construction & Training")
    model, history = train_kanfis(
        X_train, y_train, X_test, y_test,
        group_map=group_map,
        n_rules=args.n_rules,
        l1_lambda=args.l1_lambda,
        epochs=args.epochs,
        batch_size=64,
        lr=1e-3,
        patience=20,
    )
    save_model(model, os.path.join(args.output_dir, "kanfis_final.pt"))
    _save_history(history, os.path.join(args.output_dir, "training_history.csv"))

    # ══════════════════════════════════════════════
    # PHASE 4 — CLINICAL VALIDATION
    # ══════════════════════════════════════════════
    print("\n▶ Phase 4: Clinical Validation & Narrative Generation")
    metrics = full_evaluation(
        model, X_test, y_test,
        feature_names=feature_names,
        save_plots=True,
        output_dir=args.output_dir,
    )

    # ══════════════════════════════════════════════
    # OPTIONAL: 5-FOLD CROSS-VALIDATION
    # ══════════════════════════════════════════════
    if args.cv:
        print("\n▶ Running 5-Fold Cross-Validation...")
        # Combine train + test for full CV
        X_all = np.vstack([X_train, X_test])
        y_all = np.concatenate([y_train, y_test])
        cv_metrics = cross_validate_kanfis(
            X_all, y_all, group_map,
            k=5,
            n_rules=args.n_rules,
            l1_lambda=args.l1_lambda,
            epochs=args.epochs,
        )
        _print_cv_summary(cv_metrics)

    # ══════════════════════════════════════════════
    # OPTIONAL: ABLATION STUDY
    # ══════════════════════════════════════════════
    if args.ablation:
        print("\n▶ Running Ablation Study (KANFIS vs baselines)...")
        ablation_results = run_ablation_study(
            X_train, y_train, X_test, y_test, group_map
        )
        _save_ablation_results(ablation_results, args.output_dir)

    # ══════════════════════════════════════════════
    # OPTIONAL: CROSS-POPULATION TEST (Pima Bias)
    # ══════════════════════════════════════════════
    if args.cross_pop and args.pidd:
        print("\n▶ Running Cross-Population Validation (Pima Bias test)...")
        pidd_data = run_preprocessing_pipeline(
            csv_path=args.pidd,
            schema=PIDD_COLS,
            balance_strategy="smote_tomek",
            do_feature_engineering=False,
            do_feature_selection=True,
        )
        # Train a model on PIDD using the same input size as DiaBD for fair comparison
        n_pidd_features = pidd_data["X_train"].shape[1]
        pidd_group_map  = pidd_data["group_map"]

        print("  Training PIDD-baseline model...")
        model_pidd, _ = train_kanfis(
            pidd_data["X_train"], pidd_data["y_train"],
            pidd_data["X_test"],  pidd_data["y_test"],
            group_map=pidd_group_map,
            n_rules=args.n_rules,
            epochs=args.epochs,
        )

        # NOTE: Cross-population test requires the PIDD model to accept DiaBD features.
        # In practice this requires a shared feature alignment step.
        # Here we test each model on its own held-out set and compare generalisability.
        print("  [Note] For full cross-population test, align PIDD & DiaBD feature spaces.")
        print("         Refer to Research Plan §Phase 4 for feature alignment strategy.")
        cross_population_test(model_pidd, X_test, y_test, model)

    print(f"\n✓ All outputs saved to: {args.output_dir}/")
    print("  • kanfis_final.pt      — trained model weights")
    print("  • training_history.csv — epoch-by-epoch metrics")
    print("  • roc_curve.png        — ROC curve")
    print("  • pr_curve.png         — Precision-Recall curve")
    print("  • calibration.png      — Calibration curve")
    print("  • rule_weights.png     — Rule weight bar chart")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _save_history(history, path):
    import csv
    if not history:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)
    print(f"  [Save] Training history → {path}")


def _print_cv_summary(cv_metrics):
    print("\n  5-Fold CV Summary:")
    for k in ["auc", "f1", "sensitivity", "specificity"]:
        print(f"    {k:12}: {cv_metrics[k]:.4f} ± {cv_metrics[k+'_std']:.4f}")


def _save_ablation_results(results, output_dir):
    import csv
    path = os.path.join(output_dir, "ablation_results.csv")
    rows = [{"model": name, **{k: v for k, v in m.items() if not isinstance(v, np.ndarray)}}
            for name, m in results.items()]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  [Save] Ablation results → {path}")


if __name__ == "__main__":
    main()
