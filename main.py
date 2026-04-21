"""
main.py  (v2 — improvements applied)
=======
Orchestrates the complete KANFIS research pipeline:

  Phase 1 → Data Harmonization & Preprocessing
  Phase 2 → Model Construction
  Phase 3 → Training with Sparse Pruning
  Phase 4 → Clinical Validation & Narrative Generation

IMPROVEMENT CHANGELOG vs v1:
  IMP 1 — train_kanfis now returns 3 values: (model, history, ts)
           ts = TemperatureScaling module fitted on val set after training.
           All downstream evaluation uses calibrated probabilities.
  IMP 2 — full_evaluation receives ts for calibrated plots and metrics.
  IMP 3 — patience set to 40 everywhere (was 20 in main.py / 30 in train.py).
  IMP 4 — New --focal_gamma and --diversity_weight CLI args expose the
           key hyperparameters from kanfis_model.py improvements.
  IMP 5 — New --impute_strategy flag: 'mice' (default) or 'knn'.
  IMP 6 — New --no_xgb_filter flag to disable XGBoost pre-filtering.
  IMP 7 — Temperature T is logged and saved to training_history.csv metadata.

Usage:
  python main.py --diabd  path/to/diabd.csv
  python main.py --diabd  path/to/diabd.csv  --pidd path/to/pima.csv  --cross_pop
  python main.py --diabd  path/to/diabd.csv  --ablation
  python main.py --diabd  path/to/diabd.csv  --epochs 150 --focal_gamma 2.0
"""

import argparse
import os
import sys
import torch
import numpy as np

from data_preprocessing import (
    run_preprocessing_pipeline, DIABD_COLS, PIDD_COLS
)
from kanfis_model import build_kanfis
from train import train_kanfis, cross_validate_kanfis, run_ablation_study, save_model
from evaluate import full_evaluation, cross_population_test


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="KANFIS — Kolmogorov-Arnold Neuro-Fuzzy Inference System"
    )
    p.add_argument("--diabd",       type=str, required=True,
                   help="Path to DiaBD CSV file (primary dataset)")
    p.add_argument("--pidd",        type=str, default=None,
                   help="Path to PIDD CSV (optional, for cross-population test)")
    p.add_argument("--cross_pop",   action="store_true",
                   help="Run cross-population Pima Bias validation")
    p.add_argument("--ablation",    action="store_true",
                   help="Run ablation study (KANFIS vs DNN, RF, XGBoost)")
    p.add_argument("--cv",          action="store_true",
                   help="Run 5-fold cross-validation")
    p.add_argument("--epochs",      type=int,   default=150)
    p.add_argument("--n_rules",     type=int,   default=10)
    p.add_argument("--l1_lambda",   type=float, default=1e-3)
    # IMP 4: Focal loss and diversity weight are now CLI-configurable
    p.add_argument("--focal_gamma", type=float, default=2.0,
                   help="Focal loss gamma (0=BCE, 2=default focal, higher=more focus on hard samples)")
    p.add_argument("--diversity_weight", type=float, default=0.1,
                   help="Rule diversity regularisation weight (0 to disable)")
    # IMP 5: Imputation strategy
    p.add_argument("--impute_strategy", type=str, default="mice",
                   choices=["mice", "knn"],
                   help="Missing value imputation: 'mice' (default) or 'knn'")
    # IMP 6: XGB pre-filter toggle
    p.add_argument("--no_xgb_filter", action="store_true",
                   help="Disable XGBoost feature pre-filtering")
    p.add_argument("--output_dir",  type=str,   default="./outputs")
    return p.parse_args()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "█"*65)
    print("  KANFIS — Diabetes Diagnostics  (v2)")
    print("  Abhigyan Pandey (22075001) & Shivansh Kandpal (22075083)")
    print("█"*65)
    print(f"\n  Config: epochs={args.epochs} | n_rules={args.n_rules} | "
          f"focal_γ={args.focal_gamma} | diversity_w={args.diversity_weight}")
    print(f"  Impute: {args.impute_strategy} | XGB filter: {not args.no_xgb_filter}")

    # ══════════════════════════════════════════════
    # PHASE 1 — PREPROCESSING
    # ══════════════════════════════════════════════
    print("\n▶ Phase 1: Data Harmonisation & Preprocessing")
    data = run_preprocessing_pipeline(
        csv_path=args.diabd,
        schema=DIABD_COLS,
        balance_strategy="smote_tomek",
        impute_strategy=args.impute_strategy,          # IMP 5
        do_feature_engineering=True,
        do_xgb_prefilter=not args.no_xgb_filter,      # IMP 6
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
    # IMP 1 — train_kanfis returns (model, history, ts)
    model, history, ts = train_kanfis(
        X_train, y_train, X_test, y_test,
        group_map=group_map,
        n_rules=args.n_rules,
        l1_lambda=args.l1_lambda,
        focal_gamma=args.focal_gamma,           # IMP 4
        diversity_weight=args.diversity_weight, # IMP 4
        epochs=args.epochs,
        batch_size=64,
        lr=1e-3,
        patience=40,    # IMP 3: unified patience
    )

    save_model(model, os.path.join(args.output_dir, "kanfis_final.pt"))
    # IMP 7: save temperature T alongside history
    _save_history(history, os.path.join(args.output_dir, "training_history.csv"),
                  temperature=ts.temperature.item())

    # ══════════════════════════════════════════════
    # PHASE 4 — CLINICAL VALIDATION
    # ══════════════════════════════════════════════
    print("\n▶ Phase 4: Clinical Validation & Narrative Generation")
    # IMP 2 — pass ts so all metrics use calibrated probabilities
    metrics = full_evaluation(
        model, X_test, y_test,
        feature_names=feature_names,
        ts=ts,            # IMP 2
        save_plots=True,
        output_dir=args.output_dir,
    )

    # ══════════════════════════════════════════════
    # OPTIONAL: 5-FOLD CROSS-VALIDATION
    # ══════════════════════════════════════════════
    if args.cv:
        print("\n▶ Running 5-Fold Cross-Validation...")
        X_all = np.vstack([X_train, X_test])
        y_all = np.concatenate([y_train, y_test])
        cv_metrics = cross_validate_kanfis(
            X_all, y_all, group_map,
            k=5,
            n_rules=args.n_rules,
            l1_lambda=args.l1_lambda,
            focal_gamma=args.focal_gamma,
            diversity_weight=args.diversity_weight,
            epochs=args.epochs,
            patience=40,  # IMP 3
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
            impute_strategy=args.impute_strategy,
            do_feature_engineering=False,
            do_xgb_prefilter=not args.no_xgb_filter,
            do_feature_selection=True,
        )
        pidd_group_map = pidd_data["group_map"]

        print("  Training PIDD-baseline model...")
        model_pidd, _, _pidd_ts = train_kanfis(
            pidd_data["X_train"], pidd_data["y_train"],
            pidd_data["X_test"],  pidd_data["y_test"],
            group_map=pidd_group_map,
            n_rules=args.n_rules,
            focal_gamma=args.focal_gamma,
            epochs=args.epochs,
            patience=40,
        )

        print("  [Note] For full cross-population test, align PIDD & DiaBD feature spaces.")
        cross_population_test(model_pidd, X_test, y_test, model)

    print(f"\n✓ All outputs saved to: {args.output_dir}/")
    print("  • kanfis_final.pt      — trained model weights")
    print("  • training_history.csv — epoch-by-epoch metrics (+ temperature T)")
    print("  • roc_curve.png        — ROC curve (calibrated probs)")
    print("  • pr_curve.png         — Precision-Recall curve (calibrated)")
    print("  • calibration.png      — Raw vs temperature-calibrated curves")
    print("  • rule_weights.png     — Rule weight bar chart")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _save_history(history, path, temperature: float = 1.0):
    """IMP 7: temperature T is written as a metadata row."""
    import csv
    if not history:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)
    # Append temperature as a comment line
    with open(path, "a") as f:
        f.write(f"# temperature_scaling_T={temperature:.6f}\n")
    print(f"  [Save] Training history → {path}  (T={temperature:.4f})")


def _print_cv_summary(cv_metrics):
    print("\n  5-Fold CV Summary:")
    for k in ["auc", "f1", "sensitivity", "specificity"]:
        print(f"    {k:12}: {cv_metrics[k]:.4f} ± {cv_metrics[k+'_std']:.4f}")


def _save_ablation_results(results, output_dir):
    import csv
    path = os.path.join(output_dir, "ablation_results.csv")
    rows = [
        {"model": name, **{k: v for k, v in m.items() if not isinstance(v, np.ndarray)}}
        for name, m in results.items()
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  [Save] Ablation results → {path}")


if __name__ == "__main__":
    main()