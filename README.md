# KANFIS — Kolmogorov-Arnold Neuro-Fuzzy Inference System
### Interpretable & Generalised Diabetes Diagnostics
**Abhigyan Pandey (22075001) & Shivansh Kandpal (22075083)**

---

## Architecture Overview

```
Input Features (normalised)
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  GROUP KAN LAYER  (Gaussian RBF edges, domain groups) │
│  ┌────────────┐  ┌────────────────┐  ┌──────────────┐│
│  │ Metabolic  │  │ Cardiovascular │  │ Demographic  ││
│  │ BMI,Glucose│  │ BP, Stroke,CVD │  │ Age, Gender  ││
│  │ Insulin    │  │ Hypertension   │  │ Duration     ││
│  └────────────┘  └────────────────┘  └──────────────┘│
│        Shared RBF activations per group               │
│        → O(n) parameter scaling                       │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  INTERVAL TYPE-2 FUZZY ANTECEDENT LAYER               │
│  • Learnable Gaussian MF (mean, σ_upper, σ_lower)     │
│  • Footprint of Uncertainty → models clinical noise   │
│  • Additive aggregation (NOT product T-norm)          │
│  • Rule complexity: O(n)  vs  O(mⁿ) for ANFIS         │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  SPARSE RULE HEAD (L1 regularised)                    │
│  • BCE + λ·L1 composite loss                          │
│  • Prunes redundant rules to 5–7 dominant rules       │
│  • Generates IF-THEN clinical narratives              │
└──────────────────────────────────────────────────────┘
    │
    ▼
Binary Output: Diabetic / Non-Diabetic
+ Clinical Narrative (explicit fuzzy rules)
```

---

## Research Gaps Addressed

| Gap | Solution |
|-----|----------|
| **Interpretability** — ANFIS rule explosion O(mⁿ) | KAN additive decomposition → O(n) |
| **Pima Bias** — PIDD-only training | DiaBD Bangladeshi cohort, cross-pop validation |
| **Class Imbalance** — false negatives in screening | SMOTE-Tomek with constrained k-neighbours |
| **Computational inefficiency** — B-spline KANs | Gaussian RBF + Group weight sharing |

---

## File Structure

```
project/
├── data_preprocessing.py  # Phase 1: imputation, balancing, normalisation, feature selection
├── kanfis_model.py        # Phase 2: GroupKANLayer, IT2FuzzyLayer, SparseRuleHead, KANFIS
├── train.py               # Phase 3: training loop, K-fold CV, ablation study
├── evaluate.py            # Phase 4: metrics, rule extraction, clinical narratives
├── main.py                # Orchestration entry point
├── requirements.txt
├── README.md
├── data/
│   ├── diabd.csv          # DiaBD dataset (Bangladeshi cohort)
│   └── pima.csv           # PIDD dataset (Pima Indians)
└── results/
    ├── ablation_results.csv
    ├── kanfis_final.pt
    └── training_history.csv
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

### Basic training + evaluation
```bash
python main.py --diabd data/diabd.csv
```

### With 5-fold cross-validation
```bash
python main.py --diabd data/diabd.csv --cv
```

### With ablation study (KANFIS vs DNN / RF / XGBoost)
```bash
python main.py --diabd data/diabd.csv --ablation
```

### Full pipeline including Pima Bias cross-population test
```bash
python main.py \
  --diabd  data/diabd.csv \
  --pidd   data/pima.csv \
  --cross_pop \
  --ablation \
  --cv \
  --epochs 150 \
  --n_rules 10 \
  --l1_lambda 0.001 \
  --output_dir ./results
```

---

## Dataset Setup

The datasets are already provided in the `data/` folder.

### Primary — DiaBD (Bangladeshi cohort)
Located at: `data/diabd.csv`  
Download from: https://www.kaggle.com/datasets/hassinarman/diabetes-bangladesh  
or: https://data.mendeley.com/datasets/rn9m3zb7nt

Expected columns: `Age, Gender, FastingGlucose, RandomGlucose, BMI, Insulin,
SkinThickness, SystolicBP, DiastolicBP, Hypertension, HeartDisease,
IschemicStroke, VisionComplications, DurationDiabetes, Outcome`

### Baseline — PIDD
Located at: `data/pima.csv`  
Download from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database  
Expected columns: `Pregnancies, Glucose, BloodPressure, SkinThickness,
Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`

> **Note:** Rename CSV columns to match the schemas in `data_preprocessing.py`
> if your downloaded files use different header names.

---

## Outputs

| File | Description |
|------|-------------|
| `kanfis_final.pt` | Trained model weights |
| `training_history.csv` | Per-epoch loss and metrics |
| `roc_curve.png` | ROC-AUC curve |
| `pr_curve.png` | Precision-Recall curve |
| `calibration.png` | Calibration (reliability) plot |
| `rule_weights.png` | Consequent rule weight chart |
| `ablation_results.csv` | Ablation study comparison table |

---

## Example Clinical Narrative Output

```
Rule 1:
  IF   FastingGlucose is Critically Elevated AND
       DiastolicBP is Moderately High AND
       IschemicStroke is History Present
  THEN Risk of Type-2 Diabetes is HIGH
       (Consequent Weight = +0.8241, Confidence ∝ 0.8241)

Rule 2:
  IF   BMI is Obese AND
       Age is Elderly AND
       CardioFlag is Comorbidities Present
  THEN Risk of Type-2 Diabetes is HIGH
       (Consequent Weight = +0.6103, Confidence ∝ 0.6103)
```

---

## Evaluation Pillars (Research Plan)

1. **Predictive Fidelity** — ROC-AUC, F1, Precision-Recall, calibration vs XGBoost/MLP
2. **Interpretability** — Active rule count, rule inference time vs SHAP O(n²)
3. **Epidemiological Generalizability** — DiaBD cross-cohort validation, Pima Bias delta AUC
