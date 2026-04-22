"""
train.py  (v4 — firing-diversity aware)
========
Phase 3: Training Dynamics and Sparse Pruning for KANFIS.

CHANGELOG vs v3:

  FIX 1 — Warmup extended: 30 → 60 epochs.
           With the new gated IT2 layer (v5 model), the feature gates start
           uniform and need time to differentiate before L1 pruning begins.
           Extending warmup prevents L1 from zeroing gates before they learn.

  FIX 2 — Scheduler upgraded: CosineAnnealingLR replaced with OneCycleLR.
           OneCycleLR provides a peak-LR phase that helps escape the plateau
           observed in v3 (AUC stuck at 0.689 from epoch 3).  The peak LR
           of 5e-3 is 5× the base LR, providing enough gradient magnitude to
           reshape the feature gates from their uniform initialisation.

  FIX 3 — Training diagnostics: firing variance and output probability
           variance are printed every 10 epochs.  This allows immediate
           detection of any future output-collapse regression.
           Target: firing_var > 0.01, prob_var > 0.01 throughout training.

  FIX 4 — L1 warmup scale multiplied by 0.5 (max L1 contribution capped at
           ~10% of typical BCE rather than ~100%).  With the new mean L1
           penalty (FIX 6 in model), the raw L1 value is smaller but we
           also halve the scale to be safe.

  FIX 5 — spread_weight parameter plumbed through all training functions.
           Passed to build_kanfis → KANFIS.composite_loss (FIX 5 in model).

  Retained from v3:
    IMP 1 — F2-score early stopping.
    IMP 2 — post_train_threshold_search().
    IMP 3 — alpha_pos parameter.
    IMP 4 — 4-value return signature.
    IMP 5 — Sensitivity/specificity printed every 10 epochs.
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, fbeta_score,
    accuracy_score, recall_score, precision_score,
)
from kanfis_model import KANFIS, build_kanfis, calibrate_temperature


# ─────────────────────────────────────────────
# 1.  TRAINING LOOP
# ─────────────────────────────────────────────
def train_one_epoch(
    model: KANFIS,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    epoch: int = 1,
    warmup_epochs: int = 60,          # FIX 1: extended from 30 → 60
) -> dict:
    """
    FIX 1: L1 scale ramps 0 → 0.5 over warmup_epochs.
    FIX 4: Max L1 scale capped at 0.5 (was 1.0) to prevent L1 domination.
    """
    model.train()
    total_loss = bce_sum = l1_sum = div_sum = spread_sum = 0.0
    # FIX 4: cap at 0.5 instead of 1.0
    l1_scale = 0.5 * min(1.0, (epoch - 1) / max(warmup_epochs - 1, 1))

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimiser.zero_grad()

        logit = model(X_batch)
        loss, breakdown = model.composite_loss(logit, y_batch, l1_scale=l1_scale)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        total_loss  += loss.item()
        bce_sum     += breakdown["bce"]
        l1_sum      += breakdown["l1"]
        div_sum     += breakdown.get("diversity", 0.0)
        spread_sum  += breakdown.get("spread", 0.0)

    n = len(loader)
    return {
        "loss":      total_loss / n,
        "bce":       bce_sum / n,
        "l1":        l1_sum / n,
        "diversity": div_sum / n,
        "spread":    spread_sum / n,
    }


# ─────────────────────────────────────────────
# DIAGNOSTIC HELPERS  (FIX 3)
# ─────────────────────────────────────────────
@torch.no_grad()
def _compute_firing_diagnostics(model: KANFIS, loader: DataLoader,
                                 device: torch.device) -> dict:
    """
    FIX 3 — Compute firing variance and output probability variance.
    These should both be > 0.01 for a healthy model.
    If they collapse to ~0, the IT2 firing is uniform again.
    """
    model.eval()
    all_centroids, all_probs = [], []

    for X_batch, _ in loader:
        X_batch = X_batch.to(device)
        logit, fz = model(X_batch, return_rules=True)
        all_centroids.append(fz["centroid"].cpu())
        all_probs.append(torch.sigmoid(logit).cpu())

    centroids = torch.cat(all_centroids, dim=0)  # (N, n_rules)
    probs     = torch.cat(all_probs, dim=0).squeeze(-1)

    return {
        "firing_var_mean": centroids.var(dim=0).mean().item(),
        "prob_var":        probs.var().item(),
        "prob_min":        probs.min().item(),
        "prob_max":        probs.max().item(),
    }


@torch.no_grad()
def evaluate(
    model: KANFIS,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """
    Evaluate model at a given decision threshold.
    Returns F2-score alongside standard metrics.
    """
    model.eval()
    all_logits, all_labels = [], []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        logit   = model(X_batch).squeeze(-1).cpu()
        all_logits.append(logit)
        all_labels.append(y_batch)

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()

    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)

    return {
        "auc":         roc_auc_score(labels, probs),
        "f1":          f1_score(labels, preds, zero_division=0),
        "f2":          fbeta_score(labels, preds, beta=2, zero_division=0),
        "accuracy":    accuracy_score(labels, preds),
        "sensitivity": recall_score(labels, preds, pos_label=1, zero_division=0),
        "specificity": recall_score(labels, preds, pos_label=0, zero_division=0),
        "precision":   precision_score(labels, preds, zero_division=0),
        "probs":       probs,
        "labels":      labels,
    }


# ─────────────────────────────────────────────
# 2.  SENSITIVITY-CONSTRAINED THRESHOLD SEARCH
# ─────────────────────────────────────────────
def post_train_threshold_search(
    probs: np.ndarray,
    labels: np.ndarray,
    min_sensitivity: float = 0.75,
) -> tuple:
    """
    Find the decision threshold that achieves at least min_sensitivity
    while maximising specificity.  Falls back to Youden's J if target
    sensitivity is unreachable.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(labels, probs)

    candidates = [
        (float(thr), float(tpr_i), float(1 - fpr_i))
        for tpr_i, fpr_i, thr in zip(tpr, fpr, thresholds)
        if tpr_i >= min_sensitivity
    ]

    if candidates:
        best = max(candidates, key=lambda c: c[2])
        print(
            f"  [ThresholdSearch] target sens≥{min_sensitivity:.2f} → "
            f"threshold={best[0]:.3f}  "
            f"(sensitivity={best[1]:.3f}, specificity={best[2]:.3f})"
        )
        return best
    else:
        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))
        thr  = float(thresholds[best_idx])
        sens = float(tpr[best_idx])
        spec = float(1 - fpr[best_idx])
        print(
            f"  [ThresholdSearch] ⚠ Could not reach sensitivity={min_sensitivity:.2f}. "
            f"Falling back to Youden's J → "
            f"threshold={thr:.3f}  (sens={sens:.3f}, spec={spec:.3f})"
        )
        return thr, sens, spec


# ─────────────────────────────────────────────
# 3.  FULL TRAINING RUN  (single split)
# ─────────────────────────────────────────────
def train_kanfis(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    group_map:        dict,
    n_rules:          int   = 10,
    l1_lambda:        float = 5e-4,    # FIX: reduced from 1e-3
    focal_gamma:      float = 2.0,
    alpha_pos:        float = 0.75,
    diversity_weight: float = 0.05,    # FIX: reduced from 0.1
    spread_weight:    float = 0.5,     # FIX 5: new
    epochs:           int   = 200,     # FIX: increased default from 150
    batch_size:       int   = 64,
    lr:               float = 1e-3,
    patience:         int   = 50,      # FIX: increased from 40
    min_sensitivity:  float = 0.75,
    device_str:       str   = "auto",
) -> tuple:
    """
    Train KANFIS on one train/val split.

    Returns: (model, history, temperature_scaler, optimal_threshold)
      - model:              Best checkpoint by F2-score
      - history:            Per-epoch metrics list
      - temperature_scaler: TemperatureScaling fitted on val set
      - optimal_threshold:  Sensitivity-constrained threshold
    """
    device = _get_device(device_str)
    print(f"  [Train] Device: {device}")

    X_train = np.array(X_train, dtype=np.float32)
    X_val   = np.array(X_val,   dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    y_val   = np.array(y_val,   dtype=np.float32)

    n_features = X_train.shape[1]
    model = build_kanfis(
        n_features=n_features,
        group_map=group_map,
        n_rules=n_rules,
        l1_lambda=l1_lambda,
        focal_gamma=focal_gamma,
        alpha_pos=alpha_pos,
        diversity_weight=diversity_weight,
        spread_weight=spread_weight,
    ).to(device)

    train_ds  = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_ds    = TensorDataset(torch.FloatTensor(X_val),   torch.FloatTensor(y_val))
    train_ldr = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ldr   = DataLoader(val_ds,   batch_size=batch_size)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # FIX 2: OneCycleLR replaces CosineAnnealingLR
    # Provides a strong initial LR push (5× base) to escape the flat region,
    # then anneals smoothly.  pct_start=0.3 means 30% of training at peak LR.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser,
        max_lr=lr * 5,
        steps_per_epoch=len(train_ldr),
        epochs=epochs,
        pct_start=0.3,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=1e3,
    )

    best_f2    = -1.0
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0
    history    = []

    print(f"  [Train] epochs={epochs} | batch={batch_size} | "
          f"patience={patience} | warmup=60 | OneCycleLR")

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_ldr, optimiser, device, epoch, warmup_epochs=60
        )
        scheduler.step()

        val_metrics = evaluate(model, val_ldr, device, threshold=0.5)

        row = {
            "epoch":            epoch,
            "train_loss":       train_metrics["loss"],
            "train_bce":        train_metrics["bce"],
            "train_l1":         train_metrics["l1"],
            "train_diversity":  train_metrics["diversity"],
            "train_spread":     train_metrics["spread"],
            "val_auc":          val_metrics["auc"],
            "val_f1":           val_metrics["f1"],
            "val_f2":           val_metrics["f2"],
            "val_accuracy":     val_metrics["accuracy"],
            "val_sensitivity":  val_metrics["sensitivity"],
            "val_specificity":  val_metrics["specificity"],
            "val_precision":    val_metrics["precision"],
        }
        history.append(row)

        if val_metrics["f2"] > best_f2:
            best_f2    = val_metrics["f2"]
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        # FIX 3: Print firing diagnostics every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            diag = _compute_firing_diagnostics(model, val_ldr, device)
            elapsed = time.time() - t0
            print(
                f"  Ep {epoch:>4}/{epochs}  "
                f"loss={train_metrics['loss']:.4f}  "
                f"AUC={val_metrics['auc']:.4f}  "
                f"F2={val_metrics['f2']:.4f}  "
                f"Sens={val_metrics['sensitivity']:.3f}  "
                f"Spec={val_metrics['specificity']:.3f}  "
                f"| firing_var={diag['firing_var_mean']:.4f}  "
                f"prob_var={diag['prob_var']:.4f}  "
                f"p∈[{diag['prob_min']:.2f},{diag['prob_max']:.2f}]  "
                f"({elapsed:.0f}s)"
            )

        if no_improve >= patience:
            print(f"  [EarlyStop] No improvement for {patience} epochs → stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    print(f"  [Train] Best val F2 = {best_f2:.4f}")

    # Post-training: sensitivity-constrained threshold search on val set
    val_ds2    = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    val_ldr2   = DataLoader(val_ds2, batch_size=batch_size)
    val_eval   = evaluate(model, val_ldr2, device, threshold=0.5)
    opt_thr, opt_sens, opt_spec = post_train_threshold_search(
        val_eval["probs"], val_eval["labels"], min_sensitivity=min_sensitivity
    )

    # Temperature calibration
    ts = calibrate_temperature(model, X_val, y_val, device)

    return model, history, ts, opt_thr


# ─────────────────────────────────────────────
# 4.  STRATIFIED K-FOLD CROSS-VALIDATION
# ─────────────────────────────────────────────
def cross_validate_kanfis(
    X: np.ndarray,
    y: np.ndarray,
    group_map: dict,
    k: int = 5,
    **train_kwargs,
) -> dict:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics = []

    print(f"\n{'='*60}")
    print(f"  Stratified {k}-Fold Cross-Validation")
    print(f"{'='*60}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold_idx}/{k} ---")
        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]

        model, _, _ts, opt_thr = train_kanfis(
            X_tr, y_tr, X_vl, y_vl, group_map, **train_kwargs
        )

        device  = next(model.parameters()).device
        val_ds  = TensorDataset(torch.FloatTensor(X_vl), torch.FloatTensor(y_vl))
        val_ldr = DataLoader(val_ds, batch_size=64)

        m_05  = evaluate(model, val_ldr, device, threshold=0.5)
        m_opt = evaluate(model, val_ldr, device, threshold=opt_thr)

        print(
            f"  Fold {fold_idx} @0.50 → AUC={m_05['auc']:.4f}  "
            f"F2={m_05['f2']:.4f}  "
            f"Sens={m_05['sensitivity']:.4f}  "
            f"Spec={m_05['specificity']:.4f}"
        )
        print(
            f"  Fold {fold_idx} @{opt_thr:.2f} → "
            f"Sens={m_opt['sensitivity']:.4f}  "
            f"Spec={m_opt['specificity']:.4f}"
        )
        fold_metrics.append(m_opt)

    aggregated = {}
    for key in ["auc", "f1", "f2", "accuracy", "sensitivity", "specificity"]:
        vals = [m[key] for m in fold_metrics]
        aggregated[key]          = np.mean(vals)
        aggregated[f"{key}_std"] = np.std(vals)

    print(f"\n{'─'*60}")
    print(f"  CV Results ({k} folds, at sensitivity-optimised threshold):")
    for key in ["auc", "f2", "sensitivity", "specificity"]:
        print(
            f"    {key:12s}: {aggregated[key]:.4f} ± {aggregated[key+'_std']:.4f}"
        )
    return aggregated


# ─────────────────────────────────────────────
# 5.  ABLATION STUDY BASELINES
# ─────────────────────────────────────────────
def run_ablation_study(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    group_map: dict,
    alpha_pos: float = 0.75,
    min_sensitivity: float = 0.75,
    spread_weight: float = 0.5,        # FIX 5
) -> dict:
    """Evaluates KANFIS against standard baselines."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score, f1_score
    try:
        from xgboost import XGBClassifier
        has_xgb = True
    except ImportError:
        has_xgb = False

    results = {}

    # 5a. KANFIS
    print("\n[Ablation] Training KANFIS...")
    kanfis_model, _, _ts, opt_thr = train_kanfis(
        X_train, y_train, X_test, y_test,
        group_map, n_rules=10, l1_lambda=5e-4, epochs=150,
        patience=50, alpha_pos=alpha_pos,
        min_sensitivity=min_sensitivity,
        spread_weight=spread_weight,
    )
    device   = next(kanfis_model.parameters()).device
    test_ds  = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_ldr = DataLoader(test_ds, batch_size=64)

    m_05  = evaluate(kanfis_model, test_ldr, device, threshold=0.5)
    m_opt = evaluate(kanfis_model, test_ldr, device, threshold=opt_thr)
    results["KANFIS@0.5"]          = m_05
    results[f"KANFIS@{opt_thr:.2f}"] = m_opt

    # 5b. MLP baseline
    print("[Ablation] Training MLP...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=300,
        random_state=42, early_stopping=True,
    )
    mlp.fit(X_train, y_train)
    mlp_probs = mlp.predict_proba(X_test)[:, 1]
    mlp_thr, _, _ = post_train_threshold_search(mlp_probs, y_test, min_sensitivity)
    mlp_preds = (mlp_probs >= mlp_thr).astype(int)
    results[f"MLP@{mlp_thr:.2f}"] = {
        "auc":         roc_auc_score(y_test, mlp_probs),
        "f2":          fbeta_score(y_test, mlp_preds, beta=2, zero_division=0),
        "f1":          f1_score(y_test, mlp_preds, zero_division=0),
        "sensitivity": _sensitivity(y_test, mlp_preds),
        "specificity": _specificity(y_test, mlp_preds),
    }

    # 5c. Random Forest
    print("[Ablation] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1,
        class_weight="balanced",   # handles class imbalance
    )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_thr, _, _ = post_train_threshold_search(rf_probs, y_test, min_sensitivity)
    rf_preds = (rf_probs >= rf_thr).astype(int)
    results[f"RF@{rf_thr:.2f}"] = {
        "auc":         roc_auc_score(y_test, rf_probs),
        "f2":          fbeta_score(y_test, rf_preds, beta=2, zero_division=0),
        "f1":          f1_score(y_test, rf_preds, zero_division=0),
        "sensitivity": _sensitivity(y_test, rf_preds),
        "specificity": _specificity(y_test, rf_preds),
    }

    # 5d. XGBoost
    if has_xgb:
        print("[Ablation] Training XGBoost...")
        # Compute scale_pos_weight for class imbalance
        neg, pos = np.bincount(y_train.astype(int))
        xgb = XGBClassifier(
            n_estimators=200, eval_metric="logloss",
            random_state=42, scale_pos_weight=neg / max(pos, 1),
        )
        xgb.fit(X_train, y_train)
        xgb_probs = xgb.predict_proba(X_test)[:, 1]
        xgb_thr, _, _ = post_train_threshold_search(
            xgb_probs, y_test, min_sensitivity
        )
        xgb_preds = (xgb_probs >= xgb_thr).astype(int)
        results[f"XGB@{xgb_thr:.2f}"] = {
            "auc":         roc_auc_score(y_test, xgb_probs),
            "f2":          fbeta_score(y_test, xgb_preds, beta=2, zero_division=0),
            "f1":          f1_score(y_test, xgb_preds, zero_division=0),
            "sensitivity": _sensitivity(y_test, xgb_preds),
            "specificity": _specificity(y_test, xgb_preds),
        }

    print(f"\n{'─'*68}")
    print(f"  {'Model':<22} {'AUC':>7} {'F2':>7} {'Sens':>7} {'Spec':>7}")
    print(f"{'─'*68}")
    for name, m in results.items():
        print(
            f"  {name:<22} {m['auc']:>7.4f} {m.get('f2',0):>7.4f} "
            f"{m['sensitivity']:>7.4f} {m['specificity']:>7.4f}"
        )
    print(f"{'─'*68}")
    return results


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _sensitivity(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=1, zero_division=0)

def _specificity(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0, zero_division=0)

def _get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)

def save_model(model: KANFIS, path: str) -> None:
    torch.save({
        "state_dict": model.state_dict(),
        "group_map":  model.group_map,
        "n_features": model.n_features,
    }, path)
    print(f"  [Save] Model saved to {path}")

def load_model(path: str) -> KANFIS:
    ckpt  = torch.load(path, map_location="cpu")
    model = build_kanfis(ckpt["n_features"], ckpt["group_map"])
    model.load_state_dict(ckpt["state_dict"])
    print(f"  [Load] Model loaded from {path}")
    return model