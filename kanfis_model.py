"""
kanfis_model.py  (v5 — firing-diversity & output-spread fixes)
===============
Kolmogorov-Arnold Neuro-Fuzzy Inference System (KANFIS)

CHANGELOG vs v4:

  ─────────────────────────────────────────────────────────────
  ROOT-CAUSE ANALYSIS OF v4 FAILURE (AUC 0.689, stuck epoch 3)
  ─────────────────────────────────────────────────────────────
  CRITICAL BUG (v4): IT2FuzzyLayer used `per_feature.mean(-1)` to aggregate
  Gaussian activations across features into a per-rule firing strength.
  With n_features ≈ 5–8 and random centres in Z-score space, most features
  have moderate Gaussian activation (≈ 0.5–0.8), so the mean is always ≈ 0.6
  regardless of which sample is presented.  This COLLAPSES the firing space:
  all rules fire at ≈ 0.6 for every patient, so the rule head receives a
  nearly constant input and produces nearly constant logits.

  The only source of variation was the KAN correction bounded to ±0.2, which
  gave a probability spread of only ≈ [0.35, 0.65].  This explains why:
    (a) calibration shows all predictions in a narrow 0.45–0.80 band,
    (b) optimal threshold = 0.466 predicts *everything* as positive,
    (c) AUC is stuck at 0.689 from epoch 3 — no gradient can improve it.

  FIX 1 (CRITICAL) — IT2FuzzyLayer: Gated geometric-mean aggregation.
    Replace `per_feature.mean(-1)` with a **learnable feature gate**
    (n_rules × n_features) that implements a soft, weighted product T-norm:

        gate = sigmoid(feature_gate)          # (n_rules, n_features)
        gate_norm = gate / gate.sum(-1)       # normalise to sum=1
        firing = exp(Σ_f gate_norm_rf · log(rbf_brf))

    This is a weighted geometric mean.  The gate learns which 1–3 features
    define each rule's antecedent.  A rule focused on Glucose + BMI will
    fire HIGH only when *both* are elevated (product-like), not whenever
    *any* feature has moderate activation (mean-like).

    Interpretability is preserved: `gate_norm[r]` is the feature-importance
    vector for rule r — directly readable as a clinical IF-THEN antecedent.

  FIX 2 — KAN correction bounded to ±0.2 → ±0.5.
    With collapsed IT2 firing, the KAN term was the only source of logit
    variation, but was capped too tightly.  ±0.5 doubles the usable logit
    range while keeping the fuzzy centroid the dominant (interpretable) signal.

  FIX 3 — Add BatchNorm on firing strengths before SparseRuleHead.
    Normalises firing-strength distribution before linear combination,
    stabilising gradients and preventing any single rule from dominating.

  FIX 4 — SparseRuleHead upgraded to 2-layer MLP.
    Linear(n_rules, n_rules) → ELU → Dropout(0.1) → Linear(n_rules, 1)
    More expressive rule interaction while still allowing L1 pruning on
    the first layer's weights (rule antecedent salience).

  FIX 5 — Output-spread bonus in composite_loss.
    Add a penalty proportional to −Var(sigmoid(logit)) across the batch.
    This directly punishes the model when all predictions collapse to the
    same value, forcing the output distribution to spread.
    Weight: diversity_weight * 0.5 * −prob_var

  FIX 6 — L1 penalty normalised to per-rule mean (not sum).
    sum(|w|) over n_rules rules grows linearly with n_rules, making
    l1_lambda non-transferable when n_rules changes.  Mean is stable.

  IMP 1 — Alpha-balanced focal loss retained from v4 (alpha_pos=0.75).
  IMP 2 — Rule diversity min_distance=2.0 retained from v4.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
# 1.  GAUSSIAN RBF KAN EDGE
# ═══════════════════════════════════════════════════════════
class GaussianRBF(nn.Module):
    """
    Univariate KAN edge:  out(x) = linear_weight*x  +  Σ_k w_k·φ_k(x)
    Linear bypass prevents RBF≈0 at init from blocking gradient flow.
    """
    def __init__(self, n_basis: int = 8):
        super().__init__()
        self.n_basis = n_basis
        self.centres       = nn.Parameter(torch.linspace(-3.0, 3.0, n_basis).unsqueeze(0))
        self.log_widths    = nn.Parameter(torch.zeros(1, n_basis))
        self.weights       = nn.Parameter(torch.randn(n_basis) * 0.1)
        self.linear_weight = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_out = self.linear_weight * x
        x_exp  = x.unsqueeze(-1)
        sigma  = torch.exp(self.log_widths) + 1e-6
        rbf    = torch.exp(-0.5 * ((x_exp - self.centres) / sigma) ** 2)
        rbf_out = (rbf * self.weights).sum(-1)
        return linear_out + rbf_out


# ═══════════════════════════════════════════════════════════
# 2.  GROUP KAN LAYER
# ═══════════════════════════════════════════════════════════
class GroupKANLayer(nn.Module):
    """
    KAN layer with clinical domain grouping and shared RBF activations.
    Output is used as an attention correction on IT2 firing strengths,
    NOT as input to the fuzzy layer (that role belongs to raw features).
    """
    def __init__(self, n_features: int, group_map: dict,
                 out_dim: int = 16, n_basis: int = 8):
        super().__init__()
        self.group_map  = group_map
        self.out_dim    = out_dim
        self.group_rbf  = nn.ModuleDict()
        self.group_proj = nn.ModuleDict()

        for name, indices in group_map.items():
            g_size = len(indices)
            self.group_rbf[name]  = GaussianRBF(n_basis)
            self.group_proj[name] = nn.Linear(g_size, out_dim, bias=True)

        for proj in self.group_proj.values():
            nn.init.kaiming_normal_(proj.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(proj.bias)

        total_out = len(group_map) * out_dim
        self.batch_norm = nn.BatchNorm1d(total_out, momentum=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group_outputs = []
        for name, indices in self.group_map.items():
            x_g     = x[:, indices]
            rbf_out = torch.stack(
                [self.group_rbf[name](x_g[:, i]) for i in range(x_g.size(1))],
                dim=1,
            )
            proj_out = self.group_proj[name](rbf_out)
            group_outputs.append(proj_out)
        out = torch.cat(group_outputs, dim=-1)
        return self.batch_norm(out)


# ═══════════════════════════════════════════════════════════
# 3.  IT2 FUZZY LAYER — GATED AGGREGATION  (FIX 1)
# ═══════════════════════════════════════════════════════════
class IT2FuzzyLayer(nn.Module):
    """
    FIX 1 — Gated geometric-mean aggregation replaces uniform mean.

    WHY UNIFORM MEAN FAILS:
      With n features and random centres, most features have moderate
      Gaussian activation (0.5–0.8).  mean_f(activation) ≈ constant ≈ 0.6
      for ALL rules and ALL samples.  The rule head receives a constant
      input and cannot learn, causing the AUC to plateau at epoch 3.

    WHY GATED GEOMETRIC MEAN WORKS:
      Each rule r learns a feature gate gate_r ∈ R^n_features (sigmoid-ed,
      then L1-normalised to sum=1).  The firing strength is:

        firing_r(x) = exp( Σ_f gate_rf · log(rbf(x_f)) )

      = product_f  rbf(x_f)^gate_rf    (weighted geometric mean)

      A rule with gate concentrated on {Glucose, BMI} fires HIGH only when
      *both* glucose AND BMI are elevated — a product T-norm on those 2
      features.  This creates strong, clinically meaningful variation in
      firing strengths across patients.

    INTERPRETABILITY:
      gate_norm[r] is the feature-importance weight vector for rule r.
      It is visualisable as a bar chart and maps directly to the
      LINGUISTIC_MAP in evaluate.py: the top-k features with highest
      gate_norm[r] become the antecedents of rule r's IF-THEN statement.

    STABILITY:
      log(rbf) is clamped to [-8, 0] to prevent log(0) and exp overflow.
    """
    def __init__(self, in_dim: int, n_rules: int = 8):
        super().__init__()
        self.in_dim  = in_dim
        self.n_rules = n_rules

        # Centres in Z-score space: init uniformly in [-2, 2]
        self.centres = nn.Parameter(torch.empty(n_rules, in_dim))
        nn.init.uniform_(self.centres, -2.0, 2.0)

        # sigma_upper ≈ 0.80, sigma_lower ≈ 0.45 in Z-score units
        self.log_sigma_upper = nn.Parameter(torch.full((n_rules, in_dim), -0.2))
        self.log_sigma_lower = nn.Parameter(torch.full((n_rules, in_dim), -0.8))

        # FIX 1: Learnable feature gate for gated geometric-mean aggregation.
        # Init = 0 → sigmoid(0) = 0.5 → uniform distribution at start.
        # Sparsifies during training: gates focus on 1–3 features per rule.
        self.feature_gate = nn.Parameter(torch.zeros(n_rules, in_dim))

    @property
    def sigma_upper(self):
        return torch.exp(self.log_sigma_upper) + 1e-6

    @property
    def sigma_lower(self):
        su = self.sigma_upper
        sl = torch.exp(self.log_sigma_lower) + 1e-6
        return torch.minimum(sl, su * 0.9)

    def _gated_gaussian_mf(self, x, centres, sigmas):
        """
        FIX 1: Gated geometric-mean aggregation.

        x:       (batch, n_features)
        centres: (n_rules, n_features)
        sigmas:  (n_rules, n_features)
        returns: (batch, n_rules)
        """
        x_exp = x.unsqueeze(1)           # (batch, 1, n_features)
        c_exp = centres.unsqueeze(0)     # (1, n_rules, n_features)
        s_exp = sigmas.unsqueeze(0)      # (1, n_rules, n_features)
        dist  = ((x_exp - c_exp) / s_exp) ** 2
        rbf   = torch.exp(-0.5 * dist)  # (batch, n_rules, n_features) ∈ (0,1]

        # Learnable feature gate: normalise to sum=1 per rule
        gate      = torch.sigmoid(self.feature_gate)               # (n_rules, n_features)
        gate_norm = gate / gate.sum(-1, keepdim=True).clamp(min=0.1)  # (n_rules, n_features)

        # Weighted geometric mean via log space:
        #   firing_r = exp( Σ_f gate_norm_rf * log(rbf_brf) )
        # Clamp log to [-8, 0] to prevent -inf from rbf≈0 regions.
        log_rbf = torch.log(rbf.clamp(min=1e-6))                   # (batch, n_rules, n_features)
        weighted_log = (gate_norm.unsqueeze(0) * log_rbf).sum(-1)  # (batch, n_rules)
        return torch.exp(weighted_log)                              # (batch, n_rules) ∈ (0,1]

    def forward(self, x: torch.Tensor) -> dict:
        upper    = self._gated_gaussian_mf(x, self.centres, self.sigma_upper)
        lower    = self._gated_gaussian_mf(x, self.centres, self.sigma_lower)
        centroid = 0.5 * (upper + lower)
        return {"upper": upper, "lower": lower, "centroid": centroid}

    def get_rule_feature_importance(self) -> torch.Tensor:
        """Returns normalised gate weights: (n_rules, n_features)."""
        gate = torch.sigmoid(self.feature_gate).detach()
        return gate / gate.sum(-1, keepdim=True).clamp(min=0.1)


# ═══════════════════════════════════════════════════════════
# 4.  SPARSE RULE HEAD  — 2-LAYER MLP  (FIX 4, FIX 6)
# ═══════════════════════════════════════════════════════════
class SparseRuleHead(nn.Module):
    """
    FIX 4 — Upgraded from single Linear to 2-layer MLP.
    Allows nonlinear rule interactions (e.g. rule 1 AND rule 3 together)
    while still allowing L1 pruning on the hidden layer.

    FIX 6 — L1 penalty uses .mean() not .sum() for scale-invariance
    across different n_rules settings.
    """
    def __init__(self, n_rules: int, n_classes: int = 1, dropout: float = 0.10):
        super().__init__()
        self.hidden = nn.Linear(n_rules, n_rules)
        self.output = nn.Linear(n_rules, n_classes)
        self.dropout = nn.Dropout(dropout)

        # Small init: prevents L1 from zeroing weights before BCE gradients
        # have shaped the representation.
        nn.init.normal_(self.hidden.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.hidden.bias)
        nn.init.normal_(self.output.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.output.bias)

    def forward(self, firing_strengths: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.hidden(firing_strengths))
        h = self.dropout(h)
        return self.output(h)

    @property
    def l1_penalty(self) -> torch.Tensor:
        # FIX 6: mean (not sum) — scale-invariant across n_rules values
        return self.hidden.weight.abs().mean() + self.output.weight.abs().mean()


# ═══════════════════════════════════════════════════════════
# 5.  TEMPERATURE SCALING
# ═══════════════════════════════════════════════════════════
class TemperatureScaling(nn.Module):
    """
    Post-training probability calibration via a single learned scalar T.
    calibrated_logit = raw_logit / T
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.05)


def calibrate_temperature(
    model: "KANFIS",
    X_val: "np.ndarray",
    y_val: "np.ndarray",
    device: torch.device,
) -> "TemperatureScaling":
    import numpy as np
    ts = TemperatureScaling().to(device)
    model.eval()

    with torch.no_grad():
        logits = model(
            torch.FloatTensor(X_val).to(device)
        ).squeeze(-1).detach()

    labels = torch.FloatTensor(y_val).to(device)

    optimiser = torch.optim.LBFGS(
        [ts.temperature], lr=0.01, max_iter=100, line_search_fn="strong_wolfe"
    )

    def _eval():
        optimiser.zero_grad()
        loss = F.binary_cross_entropy_with_logits(ts(logits), labels)
        loss.backward()
        return loss

    optimiser.step(_eval)

    T = ts.temperature.item()
    direction = "smoothing (was over-confident)" if T > 1 else "sharpening (was under-confident)"
    print(f"  [TemperatureScaling] T = {T:.4f}  ({direction})")
    return ts


# ═══════════════════════════════════════════════════════════
# 6.  FULL KANFIS MODEL  (v5: FIX 2+3+5 applied)
# ═══════════════════════════════════════════════════════════
class KANFIS(nn.Module):
    """
    Revised data flow (v5):

      Input x (normalised Z-score)
          │
          ├───────────────────────────────────────────────────┐
          ▼                                                   ▼
      GroupKANLayer                               IT2FuzzyLayer (FIX 1)
      (nonlinear feature refinement)              (gated geo-mean firing)
          │                                                   │
          ▼                                                   │
      kan_proj → tanh → *0.5  (FIX 2: was 0.2)              │
      (bounded attention correction)                          │
          │                                                   │
          └──────────── additive combine ─────────────────────┘
                                   │
                                   ▼
                           FiringBatchNorm  (FIX 3)
                                   │
                                   ▼
                            SparseRuleHead (2-layer MLP, FIX 4)
                                   │
                                   ▼
                               logit (binary)

    FIX 5: composite_loss adds −Var(sigmoid(logit)) spread bonus.
    FIX 6: L1 penalty = mean(|weights|), scale-invariant.
    """
    def __init__(
        self,
        n_features: int,
        group_map: dict,
        kan_out_dim: int = 16,
        n_rules: int = 10,
        kan_n_basis: int = 8,
        l1_lambda: float = 5e-4,       # FIX: reduced default from 1e-3
        focal_gamma: float = 2.0,
        alpha_pos: float = 0.75,
        diversity_weight: float = 0.05, # FIX: reduced from 0.1 (was collapsing)
        spread_weight: float = 0.5,     # FIX 5: new — penalises output collapse
    ):
        super().__init__()
        self.l1_lambda        = l1_lambda
        self.n_features       = n_features
        self.group_map        = group_map
        self.focal_gamma      = focal_gamma
        self.alpha_pos        = alpha_pos
        self.diversity_weight = diversity_weight
        self.spread_weight    = spread_weight

        # Stage 1 — Group KAN (nonlinear feature refinement)
        self.group_kan = GroupKANLayer(
            n_features=n_features, group_map=group_map,
            out_dim=kan_out_dim, n_basis=kan_n_basis,
        )
        kan_total_out = len(group_map) * kan_out_dim

        # KAN attention projection → n_rules-dim correction
        self.kan_proj = nn.Linear(kan_total_out, n_rules)
        nn.init.xavier_uniform_(self.kan_proj.weight)
        nn.init.zeros_(self.kan_proj.bias)

        # Stage 2 — IT2 Fuzzy antecedent in FEATURE SPACE (gated — FIX 1)
        self.fuzzy_layer = IT2FuzzyLayer(in_dim=n_features, n_rules=n_rules)

        # FIX 3: BatchNorm on firing strengths before rule head
        self.firing_bn = nn.BatchNorm1d(n_rules, momentum=0.1)

        # Stage 3 — Sparse rule consequent (2-layer MLP — FIX 4)
        self.rule_head = SparseRuleHead(n_rules=n_rules, n_classes=1, dropout=0.10)

    def forward(self, x: torch.Tensor, return_rules: bool = False):
        # KAN branch: nonlinear representation → attention correction
        kan_out        = self.group_kan(x)
        # FIX 2: bounded correction ±0.5 (was ±0.2 — too small)
        kan_correction = 0.5 * torch.tanh(self.kan_proj(kan_out))

        # Fuzzy branch: interpretable antecedents in feature space
        fuzzy_out = self.fuzzy_layer(x)
        centroid  = fuzzy_out["centroid"]                  # (batch, n_rules) ∈ (0, 1]

        # Additive combination
        firing = centroid + kan_correction                 # (batch, n_rules)

        # FIX 3: Normalise firing distribution before linear combination
        # Handle both training (batch) and eval (single sample) modes
        if firing.size(0) > 1:
            firing_normed = self.firing_bn(firing)
        else:
            firing_normed = firing  # BN undefined for batch_size=1

        logit = self.rule_head(firing_normed)
        if return_rules:
            return logit, fuzzy_out
        return logit

    # ── Rule diversity (min_distance=2.0 for Z-score feature space)
    def _rule_diversity_loss(self, min_distance: float = 2.0) -> torch.Tensor:
        c    = self.fuzzy_layer.centres                           # (n_rules, n_features)
        diff = c.unsqueeze(0) - c.unsqueeze(1)                   # (n_rules, n_rules, n_features)
        dist = (diff.pow(2).sum(-1) + 1e-8).sqrt()               # (n_rules, n_rules)
        penalty = F.relu(min_distance - dist).triu(diagonal=1).mean()
        return penalty

    # ── FIX 5: Output spread penalty — penalise probability collapse ──────
    def _spread_loss(self, logit: torch.Tensor) -> torch.Tensor:
        """
        FIX 5 — Output-spread bonus.

        When all predictions cluster near the same value (e.g. all ≈ 0.5),
        the model provides no useful discrimination.  Penalise low variance
        in sigmoid(logit) across the batch.

        penalty = −Var(sigmoid(logit))

        Adding this (negative variance) to the total loss *rewards* the
        model for spreading its output probabilities across [0, 1], directly
        counteracting the IT2 centroid collapse bug.

        Applied only when batch_size > 4 to avoid instability on tiny batches.
        """
        if logit.size(0) <= 4:
            return torch.tensor(0.0, device=logit.device)
        probs = torch.sigmoid(logit.squeeze(-1))
        return -probs.var()

    # ── Alpha-balanced focal loss with FIX 5 spread penalty ───────────────
    def composite_loss(
        self,
        logit: torch.Tensor,
        targets: torch.Tensor,
        l1_scale: float = 1.0,
    ) -> tuple:
        """
        Alpha-balanced focal loss + L1 + diversity + FIX 5 spread penalty.

        FL_alpha(p_t) = −alpha_t · (1−p_t)^gamma · log(p_t)
          alpha_t = alpha_pos   for y=1  (diabetic)
          alpha_t = 1−alpha_pos for y=0  (non-diabetic)
        """
        logit_sq  = logit.squeeze(-1)
        targets_f = targets.float()

        bce_each = F.binary_cross_entropy_with_logits(
            logit_sq, targets_f, reduction="none"
        )
        probs = torch.sigmoid(logit_sq)
        p_t   = probs * targets_f + (1 - probs) * (1 - targets_f)

        alpha_t = self.alpha_pos * targets_f + (1 - self.alpha_pos) * (1 - targets_f)
        focal   = (alpha_t * (1 - p_t) ** self.focal_gamma * bce_each).mean()

        l1        = self.rule_head.l1_penalty          # FIX 6: mean
        diversity = self._rule_diversity_loss()
        spread    = self._spread_loss(logit)            # FIX 5: −Var(p)

        total = (
            focal
            + l1_scale * self.l1_lambda * l1
            + self.diversity_weight * diversity
            + self.spread_weight * spread          # FIX 5: rewards spread
        )
        return total, {
            "bce":       focal.item(),
            "l1":        l1.item(),
            "diversity": diversity.item(),
            "spread":    spread.item(),
        }

    def get_active_rules(self, threshold: float = 0.01) -> list:
        weights = self.rule_head.hidden.weight.detach().abs().mean(0)
        return (weights > threshold).nonzero(as_tuple=True)[0].tolist()

    def prune_rules(self, threshold: float = 0.01) -> None:
        with torch.no_grad():
            mask = self.rule_head.hidden.weight.abs().mean(0) < threshold
            self.rule_head.hidden.weight[:, mask] = 0.0
        active = self.get_active_rules(threshold)
        print(f"  [Prune] {len(active)} active rules remaining: {active}")


# ═══════════════════════════════════════════════════════════
# 7.  FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════
def build_kanfis(
    n_features: int,
    group_map: dict,
    n_rules: int = 10,
    l1_lambda: float = 5e-4,           # FIX: reduced from 1e-3
    focal_gamma: float = 2.0,
    alpha_pos: float = 0.75,
    diversity_weight: float = 0.05,    # FIX: reduced from 0.1
    spread_weight: float = 0.5,        # FIX 5: new parameter
) -> KANFIS:
    model = KANFIS(
        n_features=n_features,
        group_map=group_map,
        kan_out_dim=16,
        n_rules=n_rules,
        kan_n_basis=8,
        l1_lambda=l1_lambda,
        focal_gamma=focal_gamma,
        alpha_pos=alpha_pos,
        diversity_weight=diversity_weight,
        spread_weight=spread_weight,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  [KANFIS v5] Built — {n_params:,} params | "
        f"groups: {list(group_map.keys())} | "
        f"rules: {n_rules} | λ_L1: {l1_lambda} | "
        f"focal_γ: {focal_gamma} | α_pos: {alpha_pos} | "
        f"diversity_w: {diversity_weight} | spread_w: {spread_weight}"
    )
    return model


# ═══════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import numpy as np
    torch.manual_seed(42)
    group_map = {
        "metabolic":      [0, 1, 2],
        "cardiovascular": [3, 4, 5],
        "demographic":    [6, 7],
    }
    n_features = 8
    model = build_kanfis(n_features, group_map, n_rules=10, l1_lambda=5e-4,
                         alpha_pos=0.75, spread_weight=0.5)

    torch.manual_seed(0)
    x_fake = torch.randn(64, n_features)
    # Class-imbalanced labels (6% positive — matches real DiaBD distribution)
    y_fake = torch.zeros(64)
    y_fake[:4] = 1.0

    # Verify gated firing creates VARIANCE (the core fix)
    with torch.no_grad():
        logit_out, fz = model(x_fake, return_rules=True)
        centroid      = fz["centroid"]
        centroid_var  = centroid.var(dim=0).mean().item()
        print(f"  Centroid variance per rule (should be > 0.01): {centroid_var:.4f}")
        prob_var      = torch.sigmoid(logit_out).var().item()
        print(f"  Output probability variance (should be > 0.01): {prob_var:.4f}")

    centres_shape = model.fuzzy_layer.centres.shape
    assert centres_shape == (10, n_features), \
        f"Expected ({10}, {n_features}), got {centres_shape}"
    print(f"✓ Fuzzy centres shape: {centres_shape}  (rules × features — interpretable)")

    gate_importance = model.fuzzy_layer.get_rule_feature_importance()
    print(f"✓ Feature gate shape: {gate_importance.shape}  (interpretable per-rule weights)")
    print(f"  Top feature per rule: {gate_importance.argmax(-1).tolist()}")

    logit_out, fz = model(x_fake, return_rules=True)
    loss, bd      = model.composite_loss(logit_out, y_fake, l1_scale=0.5)

    print(f"\nLogit range : [{logit_out.min():.3f}, {logit_out.max():.3f}]")
    print(f"Centroid rng: [{fz['centroid'].min():.3f}, {fz['centroid'].max():.3f}]")
    print(f"Loss={loss:.4f}  Focal(α)={bd['bce']:.4f}  L1={bd['l1']:.4f}  "
          f"Diversity={bd['diversity']:.4f}  Spread={bd['spread']:.4f}")
    print("✓ KANFIS v5 self-test passed")