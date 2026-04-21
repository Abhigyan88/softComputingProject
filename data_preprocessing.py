"""
data_preprocessing.py
=====================
Phase 1: Data Harmonization, Imputation, Balancing, and Normalization
for the KANFIS diabetes diagnostics project.

Supports:
  - DiaBD (primary)          — Bangladeshi cohort, addresses Pima Bias
  - PIDD (baseline)          — Legacy Pima Indian dataset
  - Diabetic Dataset 2019    — Tigga & Garg, for class-balance validation
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1.  COLUMN SCHEMAS  (adapt to actual CSV headers)
# ─────────────────────────────────────────────
DIABD_COLS = {
    # Actual column names from diabd CSV (lowercase with underscores)
    "features": [
        "age", "gender",
        "pulse_rate",
        "systolic_bp", "diastolic_bp",
        "glucose", "bmi",
        "family_diabetes", "hypertensive",
        "family_hypertension", "cardiovascular_disease", "stroke",
    ],
    "target": "diabetic",
    "groups": {
        "metabolic":      ["glucose", "bmi", "family_diabetes"],
        "cardiovascular": ["systolic_bp", "diastolic_bp", "hypertensive",
                           "family_hypertension", "cardiovascular_disease", "stroke"],
        "demographic":    ["age", "gender", "pulse_rate"],
    },
}

PIDD_COLS = {
    # Actual column names from Pima CSV (title-case)
    "features": [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
    ],
    "target": "Outcome",
    "groups": {
        "metabolic":   ["Glucose", "Insulin", "BMI", "SkinThickness"],
        "hormonal":    ["DiabetesPedigreeFunction", "Pregnancies"],
        "demographic": ["Age", "BloodPressure"],
    },
}


# ─────────────────────────────────────────────
# 2.  LOADER
# ─────────────────────────────────────────────
def load_dataset(csv_path: str, schema: dict) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load a CSV, separate features from target, and return both.
    Handles:
      - Leading/trailing whitespace in column headers (e.g. '  Age')
      - Gender text column → binary 0/1 encoding
      - 'diabetic' Yes/No target → 0/1
      - Physiologically impossible zero values → NaN
    """
    df = pd.read_csv(csv_path)

    # Strip whitespace from all column names
    df.columns = df.columns.str.strip()

    # Encode gender: Female=0, Male=1
    if "gender" in df.columns:
        df["gender"] = df["gender"].str.strip().str.lower().map(
            {"female": 0, "male": 1, "f": 0, "m": 1}
        ).fillna(0).astype(int)

    # Encode 'diabetic' target: Yes=1, No=0 (DiaBD uses text labels)
    target_col = schema["target"]
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].str.strip().str.lower().map(
            {"yes": 1, "no": 0, "1": 1, "0": 0}
        ).fillna(0).astype(int)

    # Physiologically impossible zeros → NaN for continuous clinical vars
    zero_impossible = ["Glucose", "glucose", "BloodPressure", "BloodPressure",
                       "systolic_bp", "diastolic_bp", "SkinThickness",
                       "Insulin", "BMI", "bmi", "pulse_rate"]
    for col in zero_impossible:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    X = df[schema["features"]]
    y = df[schema["target"]]
    return X, y


# ─────────────────────────────────────────────
# 3.  KNN IMPUTATION
# ─────────────────────────────────────────────
def knn_impute(X: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """
    Replace NaN entries with KNN-imputed values.
    Uses 5 nearest neighbours; preserves multivariate distribution
    much better than mean/median substitution.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_imp = imputer.fit_transform(X)
    return pd.DataFrame(X_imp, columns=X.columns), imputer


# ─────────────────────────────────────────────
# 4.  CLASS BALANCING
# ─────────────────────────────────────────────
def balance_classes(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "smote_tomek",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Address class imbalance via constrained synthetic oversampling.

    'smote_tomek'  — SMOTE oversampling + Tomek undersampling (recommended)
    'adasyn'       — Adaptive Synthetic sampling
    'smote'        — Pure SMOTE

    The key hyperparameter: k_neighbors is kept small (default 5) to constrain
    synthetic generation to high-density minority neighbourhoods, preventing
    the diffuse noise that blurs fuzzy membership boundaries.
    """
    if strategy == "smote_tomek":
        sampler = SMOTETomek(
            smote=SMOTE(k_neighbors=5, random_state=random_state),
            random_state=random_state,
        )
    elif strategy == "adasyn":
        sampler = ADASYN(n_neighbors=5, random_state=random_state)
    elif strategy == "smote":
        sampler = SMOTE(k_neighbors=5, random_state=random_state)
    else:
        raise ValueError(f"Unknown balancing strategy: {strategy}")

    X_res, y_res = sampler.fit_resample(X, y)
    y_res = np.array(y_res, dtype=np.int64)   # SMOTE can return object dtype
    print(f"[Balance] Before: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"[Balance] After : {dict(zip(*np.unique(y_res, return_counts=True)))}")
    return X_res, y_res


# ─────────────────────────────────────────────
# 5.  Z-SCORE NORMALIZATION
# ─────────────────────────────────────────────
def normalize(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit StandardScaler on training data only (prevents data leakage),
    then transform both splits.  Mean=0, Var=1 ensures KAN rational
    basis functions operate in a well-conditioned, non-saturated regime.
    """
    scaler = StandardScaler()
    X_train_n = scaler.fit_transform(X_train)
    X_test_n  = scaler.transform(X_test)
    return X_train_n, X_test_n, scaler


# ─────────────────────────────────────────────
# 6.  FEATURE SELECTION  (Boruta / RFE)
# ─────────────────────────────────────────────
def select_features_rfe(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_features_range: tuple = (5, 8),
    cv: int = 5,
) -> tuple[np.ndarray, list[str], list[int]]:
    """
    Recursive Feature Elimination with Cross-Validation (RFECV) backed by
    a GradientBoostingClassifier as the estimator.
    Returns X reduced to the selected features, their names, and indices.

    Target: 5-8 features to prevent rule explosion in the fuzzy system
    while retaining diagnostic power.
    """
    estimator = GradientBoostingClassifier(n_estimators=100, random_state=42)
    selector  = RFECV(
        estimator,
        min_features_to_select=n_features_range[0],
        cv=StratifiedKFold(cv, shuffle=True, random_state=42),
        scoring="roc_auc",
        n_jobs=-1,
    )
    selector.fit(X, y)

    selected_mask   = selector.support_
    selected_names  = [feature_names[i] for i, s in enumerate(selected_mask) if s]
    selected_indices = [i for i, s in enumerate(selected_mask) if s]

    # Cap at max 8 features to preserve fuzzy rule clarity
    if len(selected_names) > n_features_range[1]:
        importances = selector.estimator_.feature_importances_
        # importances is already sized to selected features — index directly
        ranked = np.argsort(importances)[::-1][: n_features_range[1]]
        selected_indices = [selected_indices[r] for r in ranked]
        selected_names   = [selected_names[r]   for r in ranked]

    print(f"[Feature Selection] {len(selected_names)} features selected: {selected_names}")
    return X[:, selected_indices], selected_names, selected_indices


# ─────────────────────────────────────────────
# 7.  ENGINEERED COMORBIDITY FEATURES
# ─────────────────────────────────────────────
def engineer_comorbidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct latent interaction features that capture multifactorial
    diabetes progression risk — beyond simple metabolic markers.
    """
    df = df.copy()

    # Vascular risk score: Age × DiastolicBP interaction
    if all(c in df.columns for c in ["age", "diastolic_bp"]):
        df["VascularRiskScore"] = (
            df["age"] * df["diastolic_bp"]
        ) / 1e3

    # Metabolic burden: BMI × Glucose composite
    if all(c in df.columns for c in ["bmi", "glucose"]):
        df["MetabolicBurden"] = df["bmi"] * df["glucose"] / 100

    # Cardiovascular complication flag
    if all(c in df.columns for c in ["cardiovascular_disease", "stroke", "hypertensive"]):
        df["CardioFlag"] = (
            df["cardiovascular_disease"].fillna(0)
            + df["stroke"].fillna(0)
            + df["hypertensive"].fillna(0)
        ).clip(0, 3)

    return df


# ─────────────────────────────────────────────
# 8.  FULL PIPELINE
# ─────────────────────────────────────────────
def run_preprocessing_pipeline(
    csv_path: str,
    schema: dict = DIABD_COLS,
    balance_strategy: str = "smote_tomek",
    do_feature_engineering: bool = True,
    do_feature_selection: bool = True,
) -> dict:
    """
    End-to-end preprocessing:
      Load → Engineer → Impute → Split → Balance → Normalize → Select

    Returns a dict with train/test splits, scaler, selected feature names,
    and the group indices required by Group-KAN.
    """
    print("\n" + "="*60)
    print("  KANFIS Preprocessing Pipeline")
    print("="*60)

    # 8a. Load
    X_df, y = load_dataset(csv_path, schema)

    # Guaranteed target encoding — handles Yes/No strings, bool, or already-int.
    # Pandas 2.x StringDtype allows .astype(np.int64) without raising even when
    # the values are 'Yes'/'No', so we must check the actual values, not just dtype.
    def _encode_target(s):
        # Fast path: already a proper numeric dtype with no string content
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            # Double-check there are no stray string values hiding as objects
            return s.astype(np.int64)
        # Slow path: map text/bool labels → 0/1
        return (
            s.astype(str).str.strip().str.lower()
             .map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
             .fillna(0).astype(np.int64)
        )

    y = _encode_target(y)

    # 8b. Feature engineering (comorbidity interactions)
    if do_feature_engineering:
        X_df = engineer_comorbidity_features(X_df)
        # Update feature list to include engineered columns
        feature_names = list(X_df.columns)
    else:
        feature_names = schema["features"]

    # 8c. KNN Imputation
    X_df_imp, imputer = knn_impute(X_df)
    X_arr = X_df_imp.values
    y_arr = y.values

    # 8d. Stratified 80/20 split (before balancing to prevent leakage)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.2, stratify=y_arr, random_state=42
    )

    # 8e. Balance training set only
    X_train_bal, y_train_bal = balance_classes(X_train, y_train, strategy=balance_strategy)

    # 8f. Z-score normalization
    X_train_n, X_test_n, scaler = normalize(X_train_bal, X_test)

    # 8g. Feature selection
    selected_names = feature_names
    selected_idx   = list(range(len(feature_names)))
    if do_feature_selection:
        X_train_n, selected_names, selected_idx = select_features_rfe(
            X_train_n, y_train_bal, feature_names
        )
        X_test_n = X_test_n[:, selected_idx]

    # 8h. Build group index map for Group-KAN
    group_map = _build_group_map(selected_names, schema.get("groups", {}))

    print(f"\n[Pipeline] Final shapes → Train: {X_train_n.shape}, Test: {X_test_n.shape}")

    return {
        "X_train": X_train_n.astype(np.float32),
        "y_train": np.array(y_train_bal, dtype=np.float32),
        "X_test":  X_test_n.astype(np.float32),
        "y_test":  np.array(y_test,      dtype=np.float32),
        "feature_names": selected_names,
        "group_map":     group_map,
        "scaler":        scaler,
        "imputer":       imputer,
    }


def _build_group_map(selected_names: list[str], schema_groups: dict) -> dict:
    """
    Map selected feature names back to domain groups required by Group-KAN.
    Returns {group_name: [local_indices_in_selected_names]}.
    """
    group_map = {}
    for group_name, group_features in schema_groups.items():
        indices = [i for i, n in enumerate(selected_names) if n in group_features]
        if indices:
            group_map[group_name] = indices
    # Catch any engineered features not in original groups
    all_assigned = {i for idxs in group_map.values() for i in idxs}
    leftover = [i for i in range(len(selected_names)) if i not in all_assigned]
    if leftover:
        group_map["engineered"] = leftover
    return group_map


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python data_preprocessing.py <path_to_diabd.csv>")
    else:
        result = run_preprocessing_pipeline(sys.argv[1])
        print("\nGroup map:", result["group_map"])
        print("Feature names:", result["feature_names"])