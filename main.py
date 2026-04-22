"""
main.py  (v6 — precision & interpretability optimized)
=======
Orchestrates the complete KANFIS research pipeline.

CHANGELOG vs v4/v5:
  UPD 1 — Defaults updated to rescue precision and prevent rule collapse:
           l1_lambda: 5e-4 -> 5e-5
           focal_gamma: 2.0 -> 3.0
           alpha_pos: 0.75 -> 0.5
           diversity_weight: 0.05 -> 0.2
           spread_weight: 0.5 -> 0.0
           epochs: 200 -> 250
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
        description="KANFIS — Kolmogorov-Arnold Neuro-Fuzzy Inference System (v6 orchestrator)"
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
    p.add_argument("--epochs",      type=int,   default=250)          
    p.add_argument("--n_rules",     type=int,   default=10)
    p.add_argument("--l1_lambda",   type=float, default=5e-5)         
    p.add_argument("--focal_gamma", type=float, default=3.0)
    p.add_argument("--alpha_pos",   type=float, default=0.5)
    p.add_argument("--min_sensitivity", type=float, default=0.75)
    p.add_argument("--diversity_weight", type=float, default=0.2)    
    p.add_argument("--spread_weight",    type=float, default=0.0)     
    p.add_argument("--impute_strategy", type=str, default="mice",
                   choices=["mice", "knn"])
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
    print("  KANFIS — Diabetes Diagnostics  (v6 model / orchestrator)")
    print("  Abhigyan Pandey (22075001) & Shivansh Kandpal (22075083)")
    print("█"*65)
    print(f"\n  Config: epochs={args.epochs} | n_rules={args.n_rules} | "
          f"focal_γ={args.focal_gamma} | α_pos={args.alpha_pos}")
    print(f"  λ_L1={args.l1_lambda} | diversity_w={args.diversity_weight} | "
          f"spread_w={args.spread_weight}")
    print(f"  Sensitivity target: ≥{args.min_sensitivity:.2f} | "
          f"Impute: {args.impute_strategy} | "
          f"XGB filter: {not args.no_xgb_filter}")

    # ══════════════════════════════════════════════
    # PHASE 1 — PREPROCESSING
    # ══════════════════════════════════════════════
    print("\n▶ Phase 1: Data Harmonisation & Preprocessing (v4 — richer features)")
    data = run_preprocessing_pipeline(
        csv_path=args.diabd,
        schema=DIABD_COLS,
        balance_strategy="smote_tomek",
        impute_strategy=args.impute_strategy,
        do_feature_engineering=True,
        do_xgb_prefilter=not args.no_xgb_filter,
        do_feature_selection=True,
    )

    X_train       = data["X_train"]
    y_train       = data["y_train"]
    X_test        = data["X_test"]
    y_test        = data["y_test"]
    feature_names = data["feature_names"]
    group_map     = data["group_map"]

    print(f"\n  Selected features ({len(feature_names)}): {feature_names}")

    # ══════════════════════════════════════════════
    # PHASE 2+3 — TRAINING
    # ══════════════════════════════════════════════
    print("\n▶ Phase 2+3: Architectural Construction & Training (v6 model)")
    model, history, ts, opt_thr = train_kanfis(
        X_train, y_train, X_test, y_test,
        group_map=group_map,
        n_rules=args.n_rules,
        l1_lambda=args.l1_lambda,
        focal_gamma=args.focal_gamma,
        alpha_pos=args.alpha_pos,
        diversity_weight=args.diversity_weight,
        spread_weight=args.spread_weight,            
        epochs=args.epochs,
        batch_size=64,
        lr=1e-3,
        patience=50,
        min_sensitivity=args.min_sensitivity,
    )

    save_model(model, os.path.join(args.output_dir, "kanfis_final.pt"))
    _save_history(
        history,
        os.path.join(args.output_dir, "training_history.csv"),
        temperature=ts.temperature.item(),
        opt_threshold=opt_thr,
    )

    # ══════════════════════════════════════════════
    # PHASE 4 — CLINICAL VALIDATION
    # ══════════════════════════════════════════════
    print("\n▶ Phase 4: Clinical Validation & Narrative Generation")
    metrics = full_evaluation(
        model, X_test, y_test,
        feature_names=feature_names,
        ts=ts,
        opt_threshold=opt_thr,
        save_plots=True,
        output_dir=args.output_dir,
    )

    _print_summary(metrics, opt_thr)

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
            alpha_pos=args.alpha_pos,
            diversity_weight=args.diversity_weight,
            spread_weight=args.spread_weight,
            epochs=args.epochs,
            patience=50,
            min_sensitivity=args.min_sensitivity,
        )

    # ══════════════════════════════════════════════
    # OPTIONAL: ABLATION STUDY
    # ══════════════════════════════════════════════
    if args.ablation:
        print("\n▶ Running Ablation Study (KANFIS vs baselines)...")
        ablation_results = run_ablation_study(
            X_train, y_train, X_test, y_test, group_map,
            alpha_pos=args.alpha_pos,
            min_sensitivity=args.min_sensitivity,
            spread_weight=args.spread_weight,
        )
        _save_ablation_results(ablation_results, args.output_dir)

    # ══════════════════════════════════════════════
    # OPTIONAL: CROSS-POPULATION TEST
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
        model_pidd, _, _pidd_ts, _pidd_thr = train_kanfis(
            pidd_data["X_train"], pidd_data["y_train"],
            pidd_data["X_test"],  pidd_data["y_test"],
            group_map=pidd_group_map,
            n_rules=args.n_rules,
            focal_gamma=args.focal_gamma,
            alpha_pos=args.alpha_pos,
            spread_weight=args.spread_weight,
            epochs=args.epochs,
            patience=50,
            min_sensitivity=args.min_sensitivity,
        )

        print("  [Note] For full cross-population test, align PIDD & DiaBD feature spaces.")
        cross_population_test(model_pidd, X_test, y_test, model)

    print(f"\n✓ All outputs saved to: {args.output_dir}/")
    print("  • kanfis_final.pt          — trained model weights")
    print("  • training_history.csv     — per-epoch metrics (includes spread)")
    print("  • roc_curve.png            — ROC + optimal operating point")
    print("  • pr_curve.png             — Precision-Recall curve")
    print("  • calibration.png          — Raw vs calibrated + threshold line")
    print("  • rule_weights.png         — Rule weight bar chart")
    print("  • sensitivity_curve.png    — Sensitivity/Specificity vs threshold")
    if args.ablation:
        print("  • ablation_results.csv     — Ablation comparison table")


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _save_history(history, path, temperature: float = 1.0,
                  opt_threshold: float = None):
    import csv
    if not history:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)
    with open(path, "a") as f:
        f.write(f"# temperature_scaling_T={temperature:.6f}\n")
        if opt_threshold is not None:
            f.write(f"# sensitivity_opt_threshold={opt_threshold:.6f}\n")
    print(f"  [Save] Training history → {path}  "
          f"(T={temperature:.4f}, thr={opt_threshold:.3f})")


def _print_summary(metrics: dict, opt_thr: float):
    print("\n" + "═"*65)
    print("  KANFIS v6 — Final Results Summary")
    print("═"*65)
    print(f"  ROC-AUC   : {metrics['auc']:.4f}")
    print(f"  Avg Prec  : {metrics['ap']:.4f}")
    print(f"\n  {'Metric':<15} {'@0.50 (default)':>17} "
          f"{'@{:.3f} (optimal)'.format(opt_thr):>20}")
    print(f"  {'─'*55}")
    for label, k05, kopt in [
        ("Sensitivity",  "sensitivity_05", "sensitivity_opt"),
        ("Specificity",  "specificity_05", "specificity_opt"),
        ("Precision",    "precision_05",   "precision_opt"),
        ("F1-Score",     "f1_05",          "f1_opt"),
        ("F2-Score",     "f2_05",          "f2_opt"),
    ]:
        v05  = metrics.get(k05, 0.0)
        vopt = metrics.get(kopt, 0.0)
        print(f"  {label:<15} {v05:>17.4f} {vopt:>20.4f}")
    print(f"  {'─'*55}")
    δ_sens = metrics.get('sensitivity_opt', 0) - metrics.get('sensitivity_05', 0)
    δ_spec = metrics.get('specificity_opt', 0) - metrics.get('specificity_05', 0)
    print(f"  Threshold shift Δ: sensitivity {δ_sens:+.4f}, "
          f"specificity {δ_spec:+.4f}")


def _save_ablation_results(results, output_dir):
    import csv
    path = os.path.join(output_dir, "ablation_results.csv")
    rows = [
        {"model": name, **{k: v for k, v in m.items()
                           if not isinstance(v, np.ndarray)}}
        for name, m in results.items()
    ]
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"  [Save] Ablation results → {path}")


if __name__ == "__main__":
    main()