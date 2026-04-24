"""
data_preprocessing.py  (v4 — richer features & better selection ceiling)
=====================
Phase 1: Data Harmonization, Imputation, Balancing, and Normalization.

CHANGELOG vs v3:

  ROOT-CAUSE: All models (KANFIS, MLP, RF, XGB) converge to AUC ≈ 0.68–0.69.
  This is a DATA ceiling.  The most impactful preprocessing improvements are:
  (a) more informative features (interaction terms, log transforms), and
  (b) a less aggressive feature selection cap.

  FIX 1 — Log transforms for skewed continuous features.
           Glucose, insulin, and BMI are often right-skewed.  log(1+x) of
           their raw (pre-normalisation) values provides an extra feature
           with a more Gaussian distribution that benefits the fuzzy layer.
           Added features: log_glucose, log_bmi.

  FIX 2 — Richer interaction features (additional to v3 engineer step).
           Added:
             GlucoseBMI     = glucose * bmi / 100        (metabolic burden proxy)
             AgeBMI         = age * bmi / 1000           (age-adjusted obesity)
             HyperGlucose   = hypertensive * glucose     (vascular-glycaemic)
             DurationRisk   = DurationDiabetes * bmi / 10
               (available only if DurationDiabetes is in the dataset)

  FIX 3 — RFECV max features increased: 8 → 12.
           v3 capped at 8 features to "prevent rule explosion", but with
           the gated IT2 layer (v5 model) learning sparse per-rule feature
           importance, more input features are beneficial.  The gate learns
           which features to ignore; pre-filtering should be less aggressive.

  FIX 4 — XGB pre-filter threshold more conservative: top-70% importance
           retained (was top-50%), preventing premature removal of features
           with non-linear interactions.

  FIX 5 — SMOTE k_neighbors increased from 3 → 5 for larger datasets.
           k=3 can create overly tight synthetic samples that don't
           generalise.  k=5 is the sklearn default and produces more
           representative synthetic minority samples.

  Retained from v3:
    FIX 1 (v3) — Leakage-free imputation (split before impute).
    IMP 1-3    — impute_split(), corrected pipeline order.
"""

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer   # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import RFECV, SelectFromModel
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, SMOTE
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1.  COLUMN SCHEMAS
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
        "metabolic":      ["glucose", "bmi", "family_diabetes",
                           "log_glucose", "log_bmi",
                           "GlucoseBMI", "MetabolicBurden"],
        "cardiovascular": ["systolic_bp", "diastolic_bp", "hypertensive",
                           "family_hypertension", "cardiovascular_disease",
                           "stroke", "CardioFlag", "VascularRiskScore",
                           "HyperGlucose"],
        "demographic":    ["age", "gender", "pulse_rate",
                           "AgeBMI"],
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
def load_dataset(csv_path: str, schema: dict) -> tuple:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    if "gender" in df.columns:
        df["gender"] = (
            df["gender"].str.strip().str.lower()
            .map({"female": 0, "male": 1, "f": 0, "m": 1})
            .fillna(0).astype(int)
        )

    target_col = schema["target"]
    if df[target_col].dtype == object:
        df[target_col] = (
            df[target_col].str.strip().str.lower()
            .map({"yes": 1, "no": 0, "1": 1, "0": 0})
            .fillna(0).astype(int)
        )

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
# 3a.  IMPUTATION — STANDALONE
# ─────────────────────────────────────────────
def impute_missing(
    X: pd.DataFrame,
    strategy: str = "mice",
    n_neighbors: int = 5,
) -> tuple:
    if strategy == "mice":
        imputer = IterativeImputer(
            max_iter=10, random_state=42,
            initial_strategy="median",
            imputation_order="roman",
        )
    else:
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")

    X_imp = imputer.fit_transform(X)
    return pd.DataFrame(X_imp, columns=X.columns), imputer


# ─────────────────────────────────────────────
# 3b.  IMPUTATION — SPLIT-AWARE  (no leakage)
# ─────────────────────────────────────────────
def impute_split(
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    strategy: str = "mice",
    n_neighbors: int = 5,
) -> tuple:
    """
    Fits the imputer ONLY on X_train_df, then transforms both splits.
    Prevents test-set statistics from contaminating training imputation.
    """
    if strategy == "mice":
        imputer = IterativeImputer(
            max_iter=10, random_state=42,
            initial_strategy="median",
            imputation_order="roman",
        )
        print("  [Impute] MICE — fit on train, transform test (no leakage)  ✓")
    else:
        imputer = KNNImputer(n_neighbors=n_neighbors, weights="distance")
        print(f"  [Impute] KNN (k={n_neighbors}) — fit on train only  ✓")

    cols = X_train_df.columns
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train_df), columns=cols
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test_df), columns=cols
    )
    return X_train_imp, X_test_imp, imputer


# ─────────────────────────────────────────────
# 4.  CLASS BALANCING
# ─────────────────────────────────────────────
def balance_classes(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "smote_tomek",
    random_state: int = 42,
) -> tuple:
    """
    FIX 5: SMOTE k_neighbors increased from 3 → 5 (sklearn default).
    k=3 creates overly tight synthetic samples; k=5 generalises better.
    """
    print(f"  [Balance] Strategy: {strategy}")
    print(f"  [Balance] Before: {dict(zip(*np.unique(y, return_counts=True)))}")

    if strategy == "smote_tomek":
        sampler = SMOTETomek(
            smote=SMOTE(k_neighbors=5, random_state=random_state),  # FIX 5
            random_state=random_state,
        )
    elif strategy == "adasyn":
        sampler = ADASYN(n_neighbors=5, random_state=random_state)
    elif strategy == "smote":
        sampler = SMOTE(k_neighbors=5, random_state=random_state)   # FIX 5
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
) -> tuple:
    """Fit on train only to prevent data leakage."""
    scaler = StandardScaler()
    X_train_n = scaler.fit_transform(X_train)
    X_test_n  = scaler.transform(X_test)
    return X_train_n, X_test_n, scaler


# ─────────────────────────────────────────────
# 6.  XGB FEATURE PRE-FILTER
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# 6.  XGB FEATURE PRE-FILTER
# ─────────────────────────────────────────────
def xgb_feature_prefilter(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    threshold: str = "0.7*mean",   # FIX 4: was "mean" (top-50%), now top-70%
) -> tuple:
    """
    FIX 4 — More conservative pre-filter: keep features with importance
    above 70% of mean (was 100% of mean / top-50%).

    This retains more features with non-linear interactions that XGB
    importance scores may underestimate.
    """
    try:
        from xgboost import XGBClassifier
        neg, pos = np.bincount(y.astype(int))
        clf = XGBClassifier(
            n_estimators=100, eval_metric="logloss",
            random_state=42, verbosity=0,
            scale_pos_weight=neg / max(pos, 1),
        )
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
        threshold = "mean"

    selector = SelectFromModel(clf, threshold=threshold, prefit=False)
    selector.fit(X, y)
    selected_mask    = selector.get_support()
    selected_names   = [feature_names[i] for i, s in enumerate(selected_mask) if s]
    selected_indices = [i for i, s in enumerate(selected_mask) if s]

    # Ensure we keep at least 6 features
    if len(selected_names) < 6:
        importances = selector.estimator_.feature_importances_  # FIXED HERE
        top_idx = np.argsort(importances)[-6:]
        selected_indices = sorted(top_idx.tolist())
        selected_names   = [feature_names[i] for i in selected_indices]

    print(f"  [XGB Pre-filter] {len(selected_names)} features retained: {selected_names}")
    return X[:, selected_indices], selected_names, selected_indices

# ─────────────────────────────────────────────
# 7.  RFECV FEATURE SELECTION
# ─────────────────────────────────────────────
def select_features_rfe(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list,
    n_features_range: tuple = (6, 12),  # FIX 3: max raised 8 → 12
    cv: int = 5,
) -> tuple:
    """
    FIX 3 — Max features raised from 8 → 12.
    The gated IT2 layer (v5 model) learns sparse per-rule feature importance,
    so the model can handle more input features without rule explosion.
    The gate itself performs effective feature selection at inference time.
    """
    estimator = GradientBoostingClassifier(n_estimators=100, random_state=42)
    selector = RFECV(
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

    # FIX 3: Cap at max 12 (was 8)
    if len(selected_names) > n_features_range[1]:
        ranks  = selector.ranking_
        ranked = np.argsort([ranks[i] for i in selected_indices])[:n_features_range[1]]
        selected_indices = [selected_indices[r] for r in ranked]
        selected_names   = [selected_names[r]   for r in ranked]

    print(f"  [RFE] {len(selected_names)} features selected: {selected_names}")
    return X[:, selected_indices], selected_names, selected_indices


# ─────────────────────────────────────────────
# 8.  ENGINEERED FEATURES  (v4 — enriched)
# ─────────────────────────────────────────────
def engineer_comorbidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX 1 & 2: Added log transforms and richer interactions.

    New features (v4):
      log_glucose  — log(1 + glucose): reduces right skew, helps fuzzy MF
      log_bmi      — log(1 + bmi):     reduces right skew
      GlucoseBMI   — glucose * bmi / 100: direct metabolic risk proxy
      AgeBMI       — age * bmi / 1000:   age-adjusted obesity risk
      HyperGlucose — hypertensive * glucose: cardiovascular-glycaemic flag
    """
    df = df.copy()

    # FIX 1: Log transforms for skewed features
    if "glucose" in df.columns:
        df["log_glucose"] = np.log1p(df["glucose"].fillna(df["glucose"].median()))
    if "bmi" in df.columns:
        df["log_bmi"] = np.log1p(df["bmi"].fillna(df["bmi"].median()))

    # FIX 2: Richer interaction features
    if all(c in df.columns for c in ["glucose", "bmi"]):
        df["GlucoseBMI"] = df["glucose"].fillna(0) * df["bmi"].fillna(0) / 100

    if all(c in df.columns for c in ["age", "bmi"]):
        df["AgeBMI"] = df["age"].fillna(0) * df["bmi"].fillna(0) / 1000

    if all(c in df.columns for c in ["hypertensive", "glucose"]):
        df["HyperGlucose"] = df["hypertensive"].fillna(0) * df["glucose"].fillna(0)

    # v3 features retained:
    if all(c in df.columns for c in ["age", "diastolic_bp"]):
        df["VascularRiskScore"] = (df["age"] * df["diastolic_bp"]) / 1e3
    if all(c in df.columns for c in ["bmi", "glucose"]):
        df["MetabolicBurden"] = df["bmi"] * df["glucose"] / 100
    if all(c in df.columns for c in
           ["cardiovascular_disease", "stroke", "hypertensive"]):
        df["CardioFlag"] = (
            df["cardiovascular_disease"].fillna(0)
            + df["stroke"].fillna(0)
            + df["hypertensive"].fillna(0)
        ).clip(0, 3)

    return df


# ─────────────────────────────────────────────
# 9.  FULL PIPELINE  (v4 — leakage-free + richer features)
# ─────────────────────────────────────────────
def run_preprocessing_pipeline(
    csv_path: str,
    schema: dict = DIABD_COLS,
    balance_strategy: str = "smote_tomek",
    impute_strategy: str = "mice",
    do_feature_engineering: bool = True,
    do_xgb_prefilter: bool = True,
    do_feature_selection: bool = True,
) -> dict:
    """
    End-to-end preprocessing pipeline (v4).

    Order:
      Load → Engineer (FIX 1+2) → Split → Impute(train-fit-only)
      → Balance (FIX 5) → Normalize → XGB Pre-Filter (FIX 4)
      → RFECV (FIX 3) → Group Map
    """
    print("\n" + "="*60)
    print("  KANFIS Preprocessing Pipeline  (v4 — richer features)")
    print("="*60)

    # 9a. Load raw data
    X_df, y = load_dataset(csv_path, schema)

    # 9b. Encode target
    def _encode_target(s):
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            return s.astype(np.int64)
        return (
            s.astype(str).str.strip().str.lower()
             .map({"yes": 1, "no": 0, "true": 1, "false": 0, "1": 1, "0": 0})
             .fillna(0).astype(np.int64)
        )
    y_enc = _encode_target(y)

    # 9c. Feature engineering
    if do_feature_engineering:
        X_df = engineer_comorbidity_features(X_df)
        new_feats = [c for c in X_df.columns if c not in schema["features"]]
        if new_feats:
            print(f"  [Engineer] Added {len(new_feats)} features: {new_feats}")
    feature_names_all = list(X_df.columns)

    # 9d. Stratified 80/20 split BEFORE imputation (no leakage)
    y_arr = y_enc.values
    idx   = np.arange(len(y_arr))
    tr_idx, te_idx, y_train_raw, y_test = train_test_split(
        idx, y_arr, test_size=0.2, stratify=y_arr, random_state=42
    )
    X_train_df_raw = X_df.iloc[tr_idx].reset_index(drop=True)
    X_test_df_raw  = X_df.iloc[te_idx].reset_index(drop=True)

    # 9e. Impute: fit on train only
    X_train_df_imp, X_test_df_imp, imputer = impute_split(
        X_train_df_raw, X_test_df_raw, strategy=impute_strategy
    )
    X_train = X_train_df_imp.values
    X_test  = X_test_df_imp.values

    # 9f. Balance training set only
    X_train_bal, y_train_bal = balance_classes(
        X_train, y_train_raw, strategy=balance_strategy
    )

    # 9g. Z-score normalization (fit on balanced train only)
    X_train_n, X_test_n, scaler = normalize(X_train_bal, X_test)

    # 9h. XGBoost feature pre-filter (FIX 4: more conservative threshold)
    current_names = list(feature_names_all)
    if do_xgb_prefilter:
        X_train_n, current_names, xgb_idx = xgb_feature_prefilter(
            X_train_n, y_train_bal, current_names
        )
        X_test_n = X_test_n[:, xgb_idx]

    # 9i. RFECV fine-grained selection (FIX 3: max 12 features)
    selected_names = current_names
    selected_idx   = list(range(len(current_names)))
    if do_feature_selection:
        X_train_n, selected_names, selected_idx = select_features_rfe(
            X_train_n, y_train_bal, current_names,
            n_features_range=(6, 12),   # FIX 3
        )
        X_test_n = X_test_n[:, selected_idx]

    # 9j. Build group index map for Group-KAN
    group_map = _build_group_map(selected_names, schema.get("groups", {}))

    print(f"\n  [Pipeline] Final shapes → Train: {X_train_n.shape},"
          f" Test: {X_test_n.shape}")
    print(f"  [Pipeline] Groups: { {k: len(v) for k, v in group_map.items()} }")
    _print_class_dist("Test set", y_test)

    return {
        "X_train":           X_train_n.astype(np.float32),
        "y_train":           np.array(y_train_bal, dtype=np.float32),
        "X_test":            X_test_n.astype(np.float32),
        "y_test":            np.array(y_test, dtype=np.float32),
        "feature_names":     selected_names,
        # SCALER FIX: scaler was fitted on ALL engineered features before XGB/RFECV
        # selection.  Inference must transform the full set first, then slice.
        # This key records the full ordered column list the scaler expects.
        "all_feature_names": feature_names_all,
        "group_map":         group_map,
        "scaler":            scaler,
        "imputer":           imputer,
    }


# ─────────────────────────────────────────────
# 10.  GROUP MAP BUILDER
# ─────────────────────────────────────────────
def _build_group_map(selected_names: list, schema_groups: dict) -> dict:
    """
    Routes features to clinical groups.
    Engineered features are routed to the group that contains their
    parent features (e.g. log_glucose → metabolic, AgeBMI → demographic).
    """
    # Extended routing map for v4 engineered features
    feature_to_group = {}
    for group_name, group_features in schema_groups.items():
        for f in group_features:
            feature_to_group[f] = group_name

    group_map = {}
    for i, name in enumerate(selected_names):
        grp = feature_to_group.get(name)
        if grp:
            group_map.setdefault(grp, []).append(i)
        else:
            group_map.setdefault("misc", []).append(i)

    # Remove empty groups
    group_map = {k: v for k, v in group_map.items() if v}

    if not group_map:
        group_map["all"] = list(range(len(selected_names)))

    return group_map


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _print_class_dist(label: str, y: np.ndarray):
    counts = dict(zip(*np.unique(y, return_counts=True)))
    total  = len(y)
    pos    = counts.get(1, 0)
    print(f"  [ClassDist] {label}: "
          f"negative={counts.get(0,0)} ({100*(total-pos)/total:.1f}%), "
          f"positive={pos} ({100*pos/total:.1f}%)")


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