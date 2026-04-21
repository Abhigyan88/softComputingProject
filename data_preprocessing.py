"""
data_preprocessing.py  (v2 — improvements applied)
=====================
Phase 1: Data Harmonization, Imputation, Balancing, and Normalization
for the KANFIS diabetes diagnostics project.

IMPROVEMENT CHANGELOG vs v1:
  IMP 1 — MICE imputation added alongside KNN (better for correlated clinical vars)
  IMP 2 — XGBoost feature pre-filtering added (required by research plan §Feature Curation)
  IMP 3 — SMOTE k_neighbors reduced to 3 (tighter neighbourhood = less fuzzy-boundary blur)
  IMP 4 — run_preprocessing_pipeline now runs XGB pre-filter BEFORE RFECV to save compute
  IMP 5 — _build_group_map extended to handle engineered features explicitly

Supports:
  - DiaBD (primary)          — Bangladeshi cohort, addresses Pima Bias
  - PIDD (baseline)          — Legacy Pima Indian dataset
  - Diabetic Dataset 2019    — Tigga & Garg, for class-balance validation
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer   # noqa — required for MICE
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1.  COLUMN SCHEMAS  (adapt to actual CSV headers)
# ─────────────────────────────────────────────
DIABD_COLS = {
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
      - Leading/trailing whitespace in column headers
      - Gender text column → binary 0/1 encoding
      - 'diabetic' Yes/No target → 0/1
      - Physiologically impossible zero values → NaN
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if "gender" in df.columns:
        df["gender"] = df["gender"].str.strip().str.lower().map(
            {"female": 0, "male": 1, "f": 0, "m": 1}
        ).fillna(0).astype(int)

    target_col = schema["target"]
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].str.strip().str.lower().map(
            {"yes": 1, "no": 0, "1": 1, "0": 0}
        ).fillna(0).astype(int)

    zero_impossible = [
        "Glucose", "glucose", "BloodPressure",
        "systolic_bp", "diastolic_bp", "SkinThickness",
        "Insulin", "BMI", "bmi", "pulse_rate",
    ]
    for col in zero_impossible:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)

    X = df[schema["features"]]
    y = df[schema["target"]]
    return X, y


# ─────────────────────────────────────────────
# 3.  IMPUTATION  (IMP 1: MICE added)
# ─────────────────────────────────────────────
def impute_missing(
    X: pd.DataFrame,
    strategy: str = "mice",
    n_neighbors: int = 5,
) -> tuple[pd.DataFrame, object]:
    """
    IMP 1 — MICE (Multivariate Imputation by Chained Equations) replaces
    simple KNN as the default strategy.

    Why MICE over KNN for clinical data?
      - MICE models each feature as a function of all other features,
        preserving the multivariate conditional distributions critical
        for cardiovascular comorbidity features (BP, stroke, CVD).
      - KNN can underestimate variance in sparse regions of the feature
        space (common in small clinical datasets like DiaBD's 225 non-diabetics).

    strategy: 'mice' (default) | 'knn'
    """
    if strategy == "mice":
        imputer = IterativeImputer(
            max_iter=10,
            random_state=42,
            initial_strategy="median",   # robust to clinical outliers
            imputation_order="roman",
        )
        print("  [Impute] Strategy: MICE (Iterative Imputer, 10 iterations)")
    else:
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
        print(f"  [Impute] Strategy: KNN (k={n_neighbors})")

    X_imp = imputer.fit_transform(X)
    return pd.DataFrame(X_imp, columns=X.columns), imputer


# ─────────────────────────────────────────────
# 4.  CLASS BALANCING  (IMP 3: k_neighbors=3)
# ─────────────────────────────────────────────
def balance_classes(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "smote_tomek",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    IMP 3 — k_neighbors reduced from 5 → 3.

    Why smaller k?
      - DiaBD minority class (non-diabetic, n=225) is small.
      - k=5 generates synthetic samples that can interpolate across
        biologically distinct sub-groups, blurring the Gaussian MF
        boundaries in the IT2 fuzzy layer.
      - k=3 constrains synthesis to the immediate high-density
        neighbourhood of each minority sample as specified in §Phase 1
        of the research plan.

    'smote_tomek'  — SMOTE oversampling + Tomek link removal (recommended)
    'adasyn'       — Adaptive density-based synthesis
    'smote'        — Pure SMOTE
    """
    print(f"  [Balance] Strategy: {strategy}")
    print(f"  [Balance] Before: {dict(zip(*np.unique(y, return_counts=True)))}")

    if strategy == "smote_tomek":
        sampler = SMOTETomek(
            smote=SMOTE(k_neighbors=3, random_state=random_state),
            random_state=random_state,
        )
    elif strategy == "adasyn":
        sampler = ADASYN(n_neighbors=3, random_state=random_state)
    elif strategy == "smote":
        sampler = SMOTE(k_neighbors=3, random_state=random_state)
    else:
        raise ValueError(f"Unknown balancing strategy: {strategy}")

    X_res, y_res = sampler.fit_resample(X, y)
    y_res = np.array(y_res, dtype=np.int64)
    print(f"  [Balance] After : {dict(zip(*np.unique(y_res, return_counts=True)))}")
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
    then transform both splits. Mean=0, Var=1 is critical for KAN
    Gaussian RBF centres which are initialised in [-3, 3].
    """
    scaler = StandardScaler()
    X_train_n = scaler.fit_transform(X_train)
    X_test_n  = scaler.transform(X_test)
    return X_train_n, X_test_n, scaler


# ─────────────────────────────────────────────
# 6.  XGB FEATURE PRE-FILTER  (IMP 2: new function)
# ─────────────────────────────────────────────
def xgb_feature_prefilter(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    threshold: str = "median",
) -> tuple[np.ndarray, list[str], list[int]]:
    """
    IMP 2 — XGBoost-driven gradient boosting feature importance pre-filtering.

    Research plan §Feature Curation explicitly states:
      "XGBoost-driven gradient boosting feature importance pre-filtering
       must be utilized to identify the most critical clinical indicators
       before they are fed into the KANFIS architecture."

    This runs BEFORE RFECV to aggressively remove irrelevant features,
    saving compute and protecting the fuzzy rule base from noise.

    threshold: 'median' keeps features above median importance (halves count).
               'mean'   keeps features above mean importance.
               float    keeps features above that exact importance value.
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  [XGB Filter] xgboost not installed — skipping pre-filter.")
        return X, feature_names, list(range(len(feature_names)))

    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    xgb.fit(X, y)

    selector      = SelectFromModel(xgb, threshold=threshold, prefit=True)
    X_filtered    = selector.transform(X)
    selected_mask = selector.get_support()

    selected_names   = [f for f, m in zip(feature_names, selected_mask) if m]
    selected_indices = [i for i, m in enumerate(selected_mask) if m]

    print(f"  [XGB Filter] {X.shape[1]} features → {X_filtered.shape[1]} selected")
    print(f"  [XGB Filter] Kept: {selected_names}")
    return X_filtered, selected_names, selected_indices


# ─────────────────────────────────────────────
# 7.  RFECV FEATURE SELECTION
# ─────────────────────────────────────────────
def select_features_rfe(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_features_range: tuple = (5, 8),
    cv: int = 5,
) -> tuple[np.ndarray, list[str], list[int]]:
    """
    Recursive Feature Elimination with Cross-Validation (RFECV).
    Runs AFTER XGB pre-filter, so input is already cleaned.
    Target: 5-8 features to prevent rule explosion in fuzzy system.
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

    selected_mask    = selector.support_
    selected_names   = [feature_names[i] for i, s in enumerate(selected_mask) if s]
    selected_indices = [i for i, s in enumerate(selected_mask) if s]

    # Cap at max 8 to preserve fuzzy rule clarity
    if len(selected_names) > n_features_range[1]:
        importances = selector.estimator_.feature_importances_
        ranked = np.argsort(importances)[::-1][: n_features_range[1]]
        selected_indices = [selected_indices[r] for r in ranked]
        selected_names   = [selected_names[r]   for r in ranked]

    print(f"  [RFE] {len(selected_names)} features selected: {selected_names}")
    return X[:, selected_indices], selected_names, selected_indices


# ─────────────────────────────────────────────
# 8.  ENGINEERED COMORBIDITY FEATURES
# ─────────────────────────────────────────────
def engineer_comorbidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct latent interaction features capturing multifactorial
    diabetes progression risk beyond simple metabolic markers.

    These three composites are explicitly required by the research plan
    and map to the LINGUISTIC_MAP in evaluate.py.
    """
    df = df.copy()

    # Vascular risk: Age × DiastolicBP — interaction for autonomic neuropathy risk
    if all(c in df.columns for c in ["age", "diastolic_bp"]):
        df["VascularRiskScore"] = (df["age"] * df["diastolic_bp"]) / 1e3

    # Metabolic burden: BMI × Glucose composite — insulin resistance proxy
    if all(c in df.columns for c in ["bmi", "glucose"]):
        df["MetabolicBurden"] = df["bmi"] * df["glucose"] / 100

    # Cardiovascular complication flag: sum of CVD + Stroke + Hypertension
    if all(c in df.columns for c in ["cardiovascular_disease", "stroke", "hypertensive"]):
        df["CardioFlag"] = (
            df["cardiovascular_disease"].fillna(0)
            + df["stroke"].fillna(0)
            + df["hypertensive"].fillna(0)
        ).clip(0, 3)

    return df


# ─────────────────────────────────────────────
# 9.  FULL PIPELINE  (IMP 4: XGB pre-filter integrated)
# ─────────────────────────────────────────────
def run_preprocessing_pipeline(
    csv_path: str,
    schema: dict = DIABD_COLS,
    balance_strategy: str = "smote_tomek",
    impute_strategy: str = "mice",           # IMP 1: default changed to MICE
    do_feature_engineering: bool = True,
    do_xgb_prefilter: bool = True,           # IMP 2: XGBoost pre-filter flag
    do_feature_selection: bool = True,
) -> dict:
    """
    End-to-end preprocessing pipeline:
      Load → Engineer → Impute → Split → Balance → Normalize
        → XGB Pre-Filter (NEW) → RFECV Selection → Group Map

    Returns a dict with train/test splits, scaler, selected feature names,
    and the group indices required by Group-KAN.
    """
    print("\n" + "="*60)
    print("  KANFIS Preprocessing Pipeline  (v2)")
    print("="*60)

    # 9a. Load raw data
    X_df, y = load_dataset(csv_path, schema)

    def _encode_target(s):
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            return s.astype(np.int64)
        return (
            s.astype(str).str.strip().str.lower()
             .map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
             .fillna(0).astype(np.int64)
        )

    y = _encode_target(y)

    # 9b. Feature engineering (comorbidity interactions)
    if do_feature_engineering:
        X_df = engineer_comorbidity_features(X_df)
        feature_names = list(X_df.columns)
    else:
        feature_names = list(schema["features"])

    # 9c. MICE / KNN imputation  (IMP 1)
    X_df_imp, imputer = impute_missing(X_df, strategy=impute_strategy)
    X_arr = X_df_imp.values
    y_arr = y.values

    # 9d. Stratified 80/20 split (BEFORE balancing to prevent leakage)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr, test_size=0.2, stratify=y_arr, random_state=42
    )

    # 9e. Balance training set only
    X_train_bal, y_train_bal = balance_classes(
        X_train, y_train, strategy=balance_strategy
    )

    # 9f. Z-score normalization (fit on balanced train only)
    X_train_n, X_test_n, scaler = normalize(X_train_bal, X_test)

    # 9g. XGBoost feature pre-filter  (IMP 2 — runs BEFORE RFECV)
    current_names = feature_names
    if do_xgb_prefilter:
        X_train_n, current_names, xgb_idx = xgb_feature_prefilter(
            X_train_n, y_train_bal, current_names
        )
        X_test_n = X_test_n[:, xgb_idx]

    # 9h. RFECV fine-grained feature selection
    selected_names = current_names
    selected_idx   = list(range(len(current_names)))
    if do_feature_selection:
        X_train_n, selected_names, selected_idx = select_features_rfe(
            X_train_n, y_train_bal, current_names
        )
        X_test_n = X_test_n[:, selected_idx]

    # 9i. Build group index map for Group-KAN
    group_map = _build_group_map(selected_names, schema.get("groups", {}))

    print(f"\n  [Pipeline] Final shapes → Train: {X_train_n.shape}, Test: {X_test_n.shape}")
    print(f"  [Pipeline] Groups: { {k: len(v) for k, v in group_map.items()} }")

    return {
        "X_train":      X_train_n.astype(np.float32),
        "y_train":      np.array(y_train_bal, dtype=np.float32),
        "X_test":       X_test_n.astype(np.float32),
        "y_test":       np.array(y_test, dtype=np.float32),
        "feature_names": selected_names,
        "group_map":    group_map,
        "scaler":       scaler,
        "imputer":      imputer,
    }


# ─────────────────────────────────────────────
# 10.  GROUP MAP BUILDER  (IMP 5: handles engineered features)
# ─────────────────────────────────────────────
def _build_group_map(selected_names: list[str], schema_groups: dict) -> dict:
    """
    IMP 5 — Engineered features (VascularRiskScore, MetabolicBurden, CardioFlag)
    are now explicitly routed to the 'engineered' group rather than being
    silently dropped or misassigned.

    Returns {group_name: [local_indices_in_selected_names]}.
    """
    # Explicit engineered feature group
    engineered_features = {"VascularRiskScore", "MetabolicBurden", "CardioFlag"}

    group_map = {}
    for group_name, group_features in schema_groups.items():
        indices = [i for i, n in enumerate(selected_names) if n in group_features]
        if indices:
            group_map[group_name] = indices

    # Engineered group: only if any engineered features survived selection
    eng_indices = [
        i for i, n in enumerate(selected_names) if n in engineered_features
    ]
    if eng_indices:
        group_map["engineered"] = eng_indices

    # Safety net: any feature not yet assigned
    all_assigned = {i for idxs in group_map.values() for i in idxs}
    leftover = [i for i in range(len(selected_names)) if i not in all_assigned]
    if leftover:
        group_map.setdefault("misc", []).extend(leftover)

    # Ensure at least one group exists even with a minimal feature set
    if not group_map:
        group_map["all"] = list(range(len(selected_names)))

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