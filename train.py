"""
train.py
========
Phase 3: Training Dynamics and Sparse Pruning for KANFIS.

Features:
  - AdamW optimiser with cosine LR annealing
  - Stratified K-Fold cross-validation (k=5)
  - Composite BCE + L1 sparse regularisation loss
  - Early stopping on validation AUC
  - Ablation study helpers (KANFIS vs DNN, RF, XGBoost baselines)
"""

import copy
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from kanfis_model import KANFIS, build_kanfis


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
    warmup_epochs: L1 penalty is linearly ramped from 0 → full λ over this
    many epochs.  Without warm-up, L1=10.47 at epoch 1 zeroes all rule
    weights before any learning occurs, trapping the model at AUC=0.5.
    """
    model.train()
    total_loss = bce_sum = l1_sum = 0.0
    # Linearly scale L1 lambda: 0 at epoch 1, full value at epoch warmup_epochs
    l1_scale = min(1.0, (epoch - 1) / max(warmup_epochs - 1, 1))

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimiser.zero_grad()

        logit = model(X_batch)
        loss, breakdown = model.composite_loss(logit, y_batch, l1_scale=l1_scale)

        loss.backward()
        # Gradient clipping — stabilises KAN RBF parameter updates
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        total_loss += loss.item()
        bce_sum    += breakdown["bce"]
        l1_sum     += breakdown["l1"]

    n = len(loader)
    return {"loss": total_loss / n, "bce": bce_sum / n, "l1": l1_sum / n}


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

    probs = 1 / (1 + np.exp(-logits))   # sigmoid
    preds = (probs >= 0.5).astype(int)

    from sklearn.metrics import (
        roc_auc_score, f1_score, accuracy_score,
        recall_score, precision_score
    )
    return {
        "auc":       roc_auc_score(labels, probs),
        "f1":        f1_score(labels, preds, zero_division=0),
        "accuracy":  accuracy_score(labels, preds),
        "sensitivity": recall_score(labels, preds, pos_label=1, zero_division=0),
        "specificity": recall_score(labels, preds, pos_label=0, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "probs":     probs,
        "labels":    labels,
    }


# ─────────────────────────────────────────────
# 2.  FULL TRAINING RUN (single split)
# ─────────────────────────────────────────────
def train_kanfis(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    group_map: dict,
    n_rules: int = 10,
    l1_lambda: float = 1e-3,
    epochs: int = 150,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 30,
    device_str: str = "auto",
) -> tuple[KANFIS, list]:
    """
    Train KANFIS on one train/val split.

    Returns trained model + history list of dicts.
    """
    device = _get_device(device_str)

    # Ensure numeric dtypes — labels can come out as object after SMOTE
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.float32)
    X_val   = np.array(X_val,   dtype=np.float32)
    y_val   = np.array(y_val,   dtype=np.float32)

    # DataLoaders
    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)

    n_features = X_train.shape[1]
    model = build_kanfis(n_features, group_map, n_rules, l1_lambda).to(device)

    # AdamW + cosine annealing LR schedule
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=epochs, eta_min=1e-6
    )

    best_auc   = 0.0
    best_state = None
    no_improve = 0
    history    = []

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimiser, device,
                                            epoch=epoch, warmup_epochs=30)
        val_metrics   = evaluate(model, val_loader, device)
        scheduler.step()

        record = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items() if k not in ("probs","labels")},
        }
        history.append(record)

        if val_metrics["auc"] > best_auc:
            best_auc   = val_metrics["auc"]
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{epochs} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_auc={val_metrics['auc']:.4f} | "
                f"val_sens={val_metrics['sensitivity']:.4f} | "
                f"val_spec={val_metrics['specificity']:.4f}"
            )

        if no_improve >= patience:
            print(f"  [Early Stop] No improvement for {patience} epochs.")
            break

    # Restore best weights
    model.load_state_dict(best_state)
    print(f"  [Best Val AUC] {best_auc:.4f}")

    # Post-training sparse pruning
    model.prune_rules(threshold=0.01)

    return model, history


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
    skf  = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_metrics = []

    print(f"\n{'='*60}")
    print(f"  Stratified {k}-Fold Cross-Validation")
    print(f"{'='*60}")

    # Ensure numeric dtypes
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold_idx}/{k} ---")
        X_tr, X_vl = X[train_idx], X[val_idx]
        y_tr, y_vl = y[train_idx], y[val_idx]

        model, _ = train_kanfis(X_tr, y_tr, X_vl, y_vl, group_map, **train_kwargs)

        device = next(model.parameters()).device
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

    # Aggregate
    aggregated = {}
    for key in ["auc", "f1", "accuracy", "sensitivity", "specificity"]:
        vals = [m[key] for m in fold_metrics]
        aggregated[key]             = np.mean(vals)
        aggregated[f"{key}_std"]    = np.std(vals)

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

    Provides evidence for Research Plan §Evaluation Pillar 1 (Predictive Fidelity).
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
    kanfis_model, _ = train_kanfis(
        X_train, y_train, X_test, y_test,
        group_map, n_rules=10, l1_lambda=1e-3, epochs=100
    )
    device   = next(kanfis_model.parameters()).device
    test_ds  = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    test_ldr = DataLoader(test_ds, batch_size=64)
    kanfis_m = evaluate(kanfis_model, test_ldr, device)
    results["KANFIS"] = kanfis_m

    # 4b. MLP baseline
    print("[Ablation] Training MLP...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=300, random_state=42, early_stopping=True
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
            n_estimators=200, use_label_encoder=False,
            eval_metric="logloss", random_state=42
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

    # Print summary table
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
    print(f"[Save] Model saved to {path}")


def load_model(path: str) -> KANFIS:
    ckpt  = torch.load(path, map_location="cpu")
    model = build_kanfis(ckpt["n_features"], ckpt["group_map"])
    model.load_state_dict(ckpt["state_dict"])
    print(f"[Load] Model loaded from {path}")
    return model