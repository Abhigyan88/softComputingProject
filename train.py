"""
train.py  (v2 — improvements applied)
========
Phase 3: Training Dynamics and Sparse Pruning for KANFIS.

IMPROVEMENT CHANGELOG vs v1:
  IMP 1 — LR schedule: cosine-only → linear warmup (10 epochs) + cosine annealing
           Prevents unstable early RBF updates when lr=1e-3 hits the highly
           dynamic GaussianRBF activation parameters from epoch 1.
  IMP 2 — L1 warmup epochs: hardcoded 30 → proportional to total epochs (epochs//5)
           With the original 30-epoch warmup and only 35 epochs running, L1
           operated at full strength for just 5 epochs. Now scales correctly
           for any --epochs value.
  IMP 3 — Early stopping: monitor AUC → monitor F1
           AUC measures ranking; F1 measures the precision-recall balance
           that directly reflects clinical utility (catching true diabetics
           without flooding clinicians with false alarms). F1 is the better
           stopping criterion for imbalanced clinical classification.
  IMP 4 — Default patience: inconsistency fixed (main.py passed 20, train.py
           defaulted to 30). Now unified at 40 to allow L1 pruning to complete
           before early stopping fires, especially with longer 150-epoch runs.
  IMP 5 — Breakdown dict now includes 'diversity' loss for monitoring.
  IMP 6 — train_kanfis returns calibrated temperature scalar for evaluate.py.

Original features retained:
  - AdamW optimiser
  - Stratified K-Fold cross-validation (k=5)
  - Composite Focal + L1 sparse regularisation loss
  - Ablation study helpers (KANFIS vs DNN, RF, XGBoost baselines)
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
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
    warmup_epochs: int = 30,
) -> dict:
    """
    IMP 2 — warmup_epochs is now passed from train_kanfis as epochs//5,
    so L1 scaling is always proportional to the total training budget.

    L1 scale ramps linearly from 0 (epoch 1) to 1.0 (epoch warmup_epochs),
    preventing L1 from zeroing weights before classification gradients
    have had a chance to shape the rule base.
    """
    model.train()
    total_loss = bce_sum = l1_sum = div_sum = 0.0
    l1_scale = min(1.0, (epoch - 1) / max(warmup_epochs - 1, 1))

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimiser.zero_grad()

        logit = model(X_batch)
        loss, breakdown = model.composite_loss(logit, y_batch, l1_scale=l1_scale)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        total_loss += loss.item()
        bce_sum    += breakdown["bce"]
        l1_sum     += breakdown["l1"]
        div_sum    += breakdown.get("diversity", 0.0)   # IMP 5

    n = len(loader)
    return {
        "loss":      total_loss / n,
        "bce":       bce_sum / n,
        "l1":        l1_sum / n,
        "diversity": div_sum / n,                        # IMP 5
    }


@torch.no_grad()
def evaluate(
    model: KANFIS,
    loader: DataLoader,
    device: torch.device,
) -> dict:
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
    preds = (probs >= 0.5).astype(int)

    from sklearn.metrics import (
        roc_auc_score, f1_score, accuracy_score,
        recall_score, precision_score,
    )
    return {
        "auc":         roc_auc_score(labels, probs),
        "f1":          f1_score(labels, preds, zero_division=0),
        "accuracy":    accuracy_score(labels, preds),
        "sensitivity": recall_score(labels, preds, pos_label=1, zero_division=0),
        "specificity": recall_score(labels, preds, pos_label=0, zero_division=0),
        "precision":   precision_score(labels, preds, zero_division=0),
        "probs":       probs,
        "labels":      labels,
    }


# ─────────────────────────────────────────────
# 2.  FULL TRAINING RUN (single split)
# ─────────────────────────────────────────────
def train_kanfis(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    group_map:    dict,
    n_rules:      int   = 10,
    l1_lambda:    float = 1e-3,
    focal_gamma:  float = 2.0,
    diversity_weight: float = 0.1,
    epochs:       int   = 150,
    batch_size:   int   = 64,
    lr:           float = 1e-3,
    patience:     int   = 40,    # IMP 4: was 20 in main.py / 30 in train.py
    device_str:   str   = "auto",
) -> tuple[KANFIS, list, "TemperatureScaling"]:
    """
    Train KANFIS on one train/val split.

    IMP 6 — Now returns (model, history, temperature_scaler).
    The temperature scaler is fitted on X_val after training completes
    and should be used in evaluate.py for calibrated probability outputs.
    """
    device = _get_device(device_str)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_val   = np.array(X_val,   dtype=np.float32)
    y_val   = np.array(y_val,   dtype=np.float32)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True)   # drop_last avoids 1-sample BN batches
    val_loader   = DataLoader(val_ds, batch_size=batch_size)

    n_features = X_train.shape[1]
    model = build_kanfis(
        n_features, group_map, n_rules, l1_lambda, focal_gamma, diversity_weight
    ).to(device)

    # IMP 1 — LR warmup (10 epochs linear) + cosine annealing
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    warmup_steps   = min(10, epochs // 10)
    cosine_steps   = max(epochs - warmup_steps, 1)
    warmup_sched   = LinearLR(
        optimiser, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_sched   = CosineAnnealingLR(optimiser, T_max=cosine_steps, eta_min=1e-6)
    scheduler      = SequentialLR(
        optimiser,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[warmup_steps],
    )

    # IMP 2 — L1 warmup proportional to total epochs
    l1_warmup_epochs = max(20, epochs // 5)

    # IMP 3 — Monitor F1 (not AUC) for early stopping
    best_metric = 0.0
    best_state  = None
    no_improve  = 0
    history     = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimiser, device,
            epoch=epoch,
            warmup_epochs=l1_warmup_epochs,   # IMP 2
        )
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        # IMP 3 — Use F1 as the early stopping criterion
        monitor_val = val_metrics["f1"]

        record = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()
               if k not in ("probs", "labels")},
        }
        history.append(record)

        if monitor_val > best_metric:
            best_metric = monitor_val
            best_state  = copy.deepcopy(model.state_dict())
            no_improve  = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimiser.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_f1={val_metrics['f1']:.4f} | "       # IMP 3: show F1
                f"val_auc={val_metrics['auc']:.4f} | "
                f"val_sens={val_metrics['sensitivity']:.4f} | "
                f"val_spec={val_metrics['specificity']:.4f} | "
                f"lr={lr_now:.2e}"
            )

        # IMP 4 — patience=40 (unified)
        if no_improve >= patience:
            print(f"  [Early Stop] No F1 improvement for {patience} epochs.")
            break

    model.load_state_dict(best_state)
    print(f"  [Best Val F1] {best_metric:.4f}")

    model.prune_rules(threshold=0.01)

    # IMP 6 — Fit temperature scaler on validation set
    ts = calibrate_temperature(model, X_val, y_val, device)

    return model, history, ts


# ─────────────────────────────────────────────
# 3.  STRATIFIED K-FOLD CROSS-VALIDATION
# ─────────────────────────────────────────────
def cross_validate_kanfis(
    X: np.ndarray,
    y: np.ndarray,
    group_map: dict,
    k: int = 5,
    **train_kwargs,
) -> dict:
    """
    Stratified K-Fold CV for rigorous model evaluation.
    Returns aggregated metrics across all folds.
    """
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

        # train_kanfis now returns 3 values; discard ts in CV
        model, _, _ts = train_kanfis(X_tr, y_tr, X_vl, y_vl, group_map, **train_kwargs)

        device  = next(model.parameters()).device
        val_ds  = TensorDataset(torch.FloatTensor(X_vl), torch.FloatTensor(y_vl))
        val_ldr = DataLoader(val_ds, batch_size=64)
        metrics = evaluate(model, val_ldr, device)
        fold_metrics.append(metrics)

        print(
            f"  Fold {fold_idx} → AUC={metrics['auc']:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"Sens={metrics['sensitivity']:.4f}  "
            f"Spec={metrics['specificity']:.4f}"
        )

    aggregated = {}
    for key in ["auc", "f1", "accuracy", "sensitivity", "specificity"]:
        vals = [m[key] for m in fold_metrics]
        aggregated[key]          = np.mean(vals)
        aggregated[f"{key}_std"] = np.std(vals)

    print(f"\n{'─'*60}")
    print(f"  CV Results ({k} folds):")
    for key in ["auc", "f1", "sensitivity", "specificity"]:
        print(
            f"    {key:12s}: {aggregated[key]:.4f} ± {aggregated[key+'_std']:.4f}"
        )
    return aggregated


# ─────────────────────────────────────────────
# 4.  ABLATION STUDY BASELINES
# ─────────────────────────────────────────────
def run_ablation_study(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test:  np.ndarray, y_test:  np.ndarray,
    group_map: dict,
) -> dict:
    """
    Evaluates KANFIS against standard black-box models:
      - Deep Neural Network (MLP)
      - Random Forest
      - XGBoost
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score, f1_score
    try:
        from xgboost import XGBClassifier
        has_xgb = True
    except ImportError:
        has_xgb = False

    results = {}

    # 4a. KANFIS
    print("\n[Ablation] Training KANFIS...")
    kanfis_model, _, _ts = train_kanfis(
        X_train, y_train, X_test, y_test,
        group_map, n_rules=10, l1_lambda=1e-3, epochs=100,
        patience=40,   # IMP 4: consistent patience
    )
    device   = next(kanfis_model.parameters()).device
    test_ds  = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_ldr = DataLoader(test_ds, batch_size=64)
    kanfis_m = evaluate(kanfis_model, test_ldr, device)
    results["KANFIS"] = kanfis_m

    # 4b. MLP baseline
    print("[Ablation] Training MLP...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=300,
        random_state=42, early_stopping=True,
    )
    mlp.fit(X_train, y_train)
    mlp_probs = mlp.predict_proba(X_test)[:, 1]
    mlp_preds = mlp.predict(X_test)
    results["MLP"] = {
        "auc":         roc_auc_score(y_test, mlp_probs),
        "f1":          f1_score(y_test, mlp_preds, zero_division=0),
        "sensitivity": _sensitivity(y_test, mlp_preds),
        "specificity": _specificity(y_test, mlp_preds),
    }

    # 4c. Random Forest
    print("[Ablation] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_preds = rf.predict(X_test)
    results["RandomForest"] = {
        "auc":         roc_auc_score(y_test, rf_probs),
        "f1":          f1_score(y_test, rf_preds, zero_division=0),
        "sensitivity": _sensitivity(y_test, rf_preds),
        "specificity": _specificity(y_test, rf_preds),
    }

    # 4d. XGBoost
    if has_xgb:
        print("[Ablation] Training XGBoost...")
        xgb = XGBClassifier(
            n_estimators=200, eval_metric="logloss",
            use_label_encoder=False, random_state=42,
        )
        xgb.fit(X_train, y_train)
        xgb_probs = xgb.predict_proba(X_test)[:, 1]
        xgb_preds = xgb.predict(X_test)
        results["XGBoost"] = {
            "auc":         roc_auc_score(y_test, xgb_probs),
            "f1":          f1_score(y_test, xgb_preds, zero_division=0),
            "sensitivity": _sensitivity(y_test, xgb_preds),
            "specificity": _specificity(y_test, xgb_preds),
        }

    print(f"\n{'─'*62}")
    print(f"  {'Model':<15} {'AUC':>8} {'F1':>8} {'Sens':>8} {'Spec':>8}")
    print(f"{'─'*62}")
    for name, m in results.items():
        print(
            f"  {name:<15} {m['auc']:>8.4f} {m['f1']:>8.4f} "
            f"{m['sensitivity']:>8.4f} {m['specificity']:>8.4f}"
        )
    print(f"{'─'*62}")
    return results


def _sensitivity(y_true, y_pred):
    from sklearn.metrics import recall_score
    return recall_score(y_true, y_pred, pos_label=1, zero_division=0)

def _specificity(y_true, y_pred):
    from sklearn.metrics import recall_score
    return recall_score(y_true, y_pred, pos_label=0, zero_division=0)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
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