"""
kanfis_model.py  (v3 — performance improvements applied)
===============
Kolmogorov-Arnold Neuro-Fuzzy Inference System (KANFIS)

IMPROVEMENT CHANGELOG vs v2:
  IMP 1 — composite_loss: BCE → Focal Loss
           Focal Loss down-weights easy correct predictions and forces
           the model to focus on hard false negatives (missed diabetics).
           This directly fixes the 7:3 negative/positive rule imbalance
           caused by residual class imbalance surviving SMOTE-Tomek.
  IMP 2 — _rule_diversity_loss() added to KANFIS
           Penalises pairs of fuzzy rule centres that are too close together,
           forcing each rule to specialise on a distinct patient sub-population.
           Without this, multiple rules fire on identical profiles (evident in
           the original rule weights chart — all negative bars are ~equal height).
  IMP 3 — TemperatureScaling module added
           Post-training calibration wrapper. The original calibration curve
           showed systematic under-prediction (fraction positives << predicted
           probability). Temperature scaling learns a single scalar T on the
           validation set to correct this without retraining the full model.

Original bug-fix changelog (v2):
  FIX 1 — GaussianRBF: linear bypass  edge(x) = w*x + phi(x)
  FIX 2 — GroupKANLayer: LayerNorm → BatchNorm1d
  FIX 3 — IT2FuzzyLayer: dist.sum → per_feat.mean + smaller sigma init
  FIX 4 — SparseRuleHead: Kaiming → small init std=0.05
  FIX 5 — Bottleneck: Tanh → BatchNorm1d + GELU
  FIX 6 — composite_loss: L1 warm-up via l1_scale parameter
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

    The linear bypass is critical — without it, RBF≈0 at init,
    blocking all gradient signal from input features.
    """
    def __init__(self, n_basis: int = 8):
        super().__init__()
        self.n_basis = n_basis
        self.centres = nn.Parameter(
            torch.linspace(-3.0, 3.0, n_basis).unsqueeze(0)   # (1, n_basis)
        )
        self.log_widths     = nn.Parameter(torch.zeros(1, n_basis))
        self.weights        = nn.Parameter(torch.randn(n_basis) * 0.1)
        self.linear_weight  = nn.Parameter(torch.ones(1))

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
    BatchNorm1d (not LayerNorm) preserves inter-sample discriminative variance.
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
# 3.  IT2 FUZZY LAYER
# ═══════════════════════════════════════════════════════════
class IT2FuzzyLayer(nn.Module):
    """
    Interval Type-2 fuzzy antecedent layer.

    Additive aggregation (per_feature.mean) instead of product T-norm
    prevents firing strengths collapsing to ~0 in high-dimensional inputs.
    Smaller initial sigma (≈0.5) gives discriminative firing across the
    normalised Z-score input space.
    """
    def __init__(self, in_dim: int, n_rules: int = 8):
        super().__init__()
        self.in_dim  = in_dim
        self.n_rules = n_rules
        self.centres = nn.Parameter(torch.empty(n_rules, in_dim))
        nn.init.uniform_(self.centres, -1.0, 1.0)
        # sigma_upper ≈ 0.50, sigma_lower ≈ 0.30
        self.log_sigma_upper = nn.Parameter(torch.full((n_rules, in_dim), -0.7))
        self.log_sigma_lower = nn.Parameter(torch.full((n_rules, in_dim), -1.2))

    @property
    def sigma_upper(self):
        return torch.exp(self.log_sigma_upper) + 1e-6

    @property
    def sigma_lower(self):
        su = self.sigma_upper
        sl = torch.exp(self.log_sigma_lower) + 1e-6
        return torch.minimum(sl, su * 0.9)

    def _gaussian_mf(self, x, centres, sigmas):
        x_exp       = x.unsqueeze(1)
        c_exp       = centres.unsqueeze(0)
        s_exp       = sigmas.unsqueeze(0)
        dist        = ((x_exp - c_exp) / s_exp) ** 2
        per_feature = torch.exp(-0.5 * dist)
        return per_feature.mean(-1)                           # (batch, n_rules)

    def forward(self, x: torch.Tensor) -> dict:
        upper    = self._gaussian_mf(x, self.centres, self.sigma_upper)
        lower    = self._gaussian_mf(x, self.centres, self.sigma_lower)
        centroid = 0.5 * (upper + lower)
        return {"upper": upper, "lower": lower, "centroid": centroid}


# ═══════════════════════════════════════════════════════════
# 4.  SPARSE RULE HEAD
# ═══════════════════════════════════════════════════════════
class SparseRuleHead(nn.Module):
    """
    L1-regularised consequent layer.
    Small weight init (std=0.05) gives neutral start → prevents all-one-class
    prediction from epoch 1 which traps the L1 into zeroing everything.
    """
    def __init__(self, n_rules: int, n_classes: int = 1):
        super().__init__()
        self.rule_weights = nn.Linear(n_rules, n_classes, bias=True)
        nn.init.normal_(self.rule_weights.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.rule_weights.bias)

    def forward(self, firing_strengths: torch.Tensor) -> torch.Tensor:
        return self.rule_weights(firing_strengths)

    @property
    def l1_penalty(self) -> torch.Tensor:
        return self.rule_weights.weight.abs().sum()


# ═══════════════════════════════════════════════════════════
# 5.  TEMPERATURE SCALING  (IMP 3: new module)
# ═══════════════════════════════════════════════════════════
class TemperatureScaling(nn.Module):
    """
    IMP 3 — Post-training probability calibration via temperature scaling.

    The original calibration curve showed fraction_positives << predicted_prob
    across all bins — the model is systematically over-confident, predicting
    high probabilities for samples that are mostly non-diabetic.

    Temperature scaling divides all logits by a learned scalar T:
        calibrated_logit = logit / T
    T > 1 flattens the probability distribution (fixes over-confidence).
    T < 1 sharpens it. T=1 is identity (no change).

    T is learned by minimising NLL (cross-entropy) on the held-out
    validation set AFTER the main model is trained. Only T is optimised —
    the KANFIS weights are frozen. This is the standard post-hoc calibration
    approach (Guo et al., ICML 2017).

    Usage:
        ts = calibrate_temperature(model, X_val, y_val, device)
        probs = torch.sigmoid(ts(model(X)) )
    """
    def __init__(self):
        super().__init__()
        # Start at T=1.5: slight smoothing as a sensible prior
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.05)  # clamp prevents T→0


def calibrate_temperature(
    model: "KANFIS",
    X_val: "np.ndarray",
    y_val: "np.ndarray",
    device: torch.device,
) -> TemperatureScaling:
    """
    Fit TemperatureScaling on the validation set using L-BFGS.
    Returns the fitted TemperatureScaling module.
    Call this AFTER train_kanfis() completes.

    Example:
        model, history = train_kanfis(...)
        ts = calibrate_temperature(model, X_val, y_val, device)
        # In evaluate.py pass ts to full_evaluation:
        full_evaluation(model, ts, X_test, y_test, ...)
    """
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
        scaled_logits = ts(logits)
        loss = F.binary_cross_entropy_with_logits(scaled_logits, labels)
        loss.backward()
        return loss

    optimiser.step(_eval)

    T = ts.temperature.item()
    print(f"  [TemperatureScaling] Optimal T = {T:.4f}  "
          f"({'smoothing — was over-confident' if T > 1 else 'sharpening'})")
    return ts


# ═══════════════════════════════════════════════════════════
# 6.  FULL KANFIS MODEL
# ═══════════════════════════════════════════════════════════
class KANFIS(nn.Module):
    def __init__(self, n_features: int, group_map: dict,
                 kan_out_dim: int = 16, n_rules: int = 10,
                 kan_n_basis: int = 8, l1_lambda: float = 1e-3,
                 focal_gamma: float = 2.0,
                 diversity_weight: float = 0.1):
        super().__init__()
        self.l1_lambda       = l1_lambda
        self.n_features      = n_features
        self.group_map       = group_map
        self.focal_gamma     = focal_gamma       # IMP 1
        self.diversity_weight = diversity_weight  # IMP 2

        # Stage 1 — Group KAN
        self.group_kan = GroupKANLayer(
            n_features=n_features, group_map=group_map,
            out_dim=kan_out_dim, n_basis=kan_n_basis,
        )
        kan_total_out = len(group_map) * kan_out_dim

        # Bottleneck: KAN space → compact fuzzy input (BN + GELU, no Tanh)
        fuzzy_in_dim = max(2 * n_rules, 8)
        self.fuzzy_proj = nn.Sequential(
            nn.Linear(kan_total_out, fuzzy_in_dim),
            nn.BatchNorm1d(fuzzy_in_dim),
            nn.GELU(),
        )
        nn.init.xavier_uniform_(self.fuzzy_proj[0].weight)
        nn.init.zeros_(self.fuzzy_proj[0].bias)

        # Stage 2 — IT2 Fuzzy antecedent
        self.fuzzy_layer = IT2FuzzyLayer(in_dim=fuzzy_in_dim, n_rules=n_rules)

        # Stage 3 — Sparse rule consequent
        self.rule_head = SparseRuleHead(n_rules=n_rules, n_classes=1)

    def forward(self, x: torch.Tensor, return_rules: bool = False):
        kan_out   = self.group_kan(x)
        fuzzy_in  = self.fuzzy_proj(kan_out)
        fuzzy_out = self.fuzzy_layer(fuzzy_in)
        firing    = fuzzy_out["centroid"]
        logit     = self.rule_head(firing)
        if return_rules:
            return logit, fuzzy_out
        return logit

    # ── IMP 2: Rule diversity loss ───────────────────────────────────────
    def _rule_diversity_loss(self, min_distance: float = 0.5) -> torch.Tensor:
        """
        IMP 2 — Penalise pairs of fuzzy rule centres that are too similar.

        Why this matters:
          Without diversity regularisation, multiple rules learn to fire
          on nearly identical patient profiles (e.g. all "high BMI + high glucose").
          The rule weight chart showed 7 near-identical protective bars, which is
          symptomatic of this collapse. Diversity loss pushes centres apart so
          each rule specialises on a DIFFERENT clinical sub-population.

        Implementation:
          Computes pairwise L2 distances between rule centre vectors.
          Penalises any pair closer than min_distance (in Z-score units).
          Only the upper triangle is counted (avoids double-counting).
        """
        c    = self.fuzzy_layer.centres                         # (n_rules, in_dim)
        diff = c.unsqueeze(0) - c.unsqueeze(1)                  # (n,n,d)
        dist = diff.pow(2).sum(-1).sqrt()                       # (n,n)
        # relu(min_dist - dist): positive only when pair is too close
        penalty = F.relu(min_distance - dist).triu(diagonal=1).mean()
        return penalty

    # ── IMP 1: Focal Loss composite ──────────────────────────────────────
    def composite_loss(
        self,
        logit: torch.Tensor,
        targets: torch.Tensor,
        l1_scale: float = 1.0,
    ) -> tuple:
        """
        IMP 1 — Focal Loss replaces BCE as the primary classification objective.

        Why Focal Loss?
          DiaBD has 840 diabetic vs 225 non-diabetic cases (3.7:1 ratio).
          Even after SMOTE-Tomek, the class balance is imperfect.
          Standard BCE treats every sample equally; easy majority-class
          samples (correctly classified non-diabetics) dominate the gradient,
          pushing the model toward specificity at the cost of sensitivity.

          Focal Loss: FL(p_t) = -(1 - p_t)^γ · log(p_t)
            γ (focal_gamma) = 2.0 by default
            → Well-classified samples (p_t ≈ 1) get weight ≈ 0 (down-weighted)
            → Hard samples (p_t ≈ 0.5) get full weight

          This directly addresses the 7:3 negative/positive rule imbalance
          seen in the rule_weights.png output.

        Loss = FocalLoss + l1_scale * λ_L1 * L1 + diversity_weight * DiversityLoss

        l1_scale: linearly warmed up 0→1 over warmup_epochs (prevents L1 from
                  zeroing weights before any learning occurs — see train.py).
        """
        # Focal loss
        logit_sq  = logit.squeeze(-1)
        bce_each  = F.binary_cross_entropy_with_logits(
            logit_sq, targets.float(), reduction="none"
        )
        probs  = torch.sigmoid(logit_sq)
        p_t    = probs * targets + (1 - probs) * (1 - targets)
        focal  = ((1 - p_t) ** self.focal_gamma * bce_each).mean()

        # L1 sparsity on rule weights
        l1 = self.rule_head.l1_penalty

        # IMP 2: Rule diversity (weighted by diversity_weight)
        diversity = self._rule_diversity_loss()

        total = (
            focal
            + l1_scale * self.l1_lambda * l1
            + self.diversity_weight * diversity
        )
        return total, {
            "bce":       focal.item(),
            "l1":        l1.item(),
            "diversity": diversity.item(),
        }

    def get_active_rules(self, threshold: float = 0.01) -> list:
        weights = self.rule_head.rule_weights.weight.detach().abs().squeeze(0)
        return (weights > threshold).nonzero(as_tuple=True)[0].tolist()

    def prune_rules(self, threshold: float = 0.01) -> None:
        with torch.no_grad():
            mask = self.rule_head.rule_weights.weight.abs() < threshold
            self.rule_head.rule_weights.weight[mask] = 0.0
        active = self.get_active_rules(threshold)
        print(f"  [Prune] {len(active)} active rules remaining: {active}")


# ═══════════════════════════════════════════════════════════
# 7.  FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════
def build_kanfis(
    n_features: int,
    group_map: dict,
    n_rules: int = 10,
    l1_lambda: float = 1e-3,
    focal_gamma: float = 2.0,       # IMP 1
    diversity_weight: float = 0.1,  # IMP 2
) -> KANFIS:
    model = KANFIS(
        n_features=n_features,
        group_map=group_map,
        kan_out_dim=16,
        n_rules=n_rules,
        kan_n_basis=8,
        l1_lambda=l1_lambda,
        focal_gamma=focal_gamma,
        diversity_weight=diversity_weight,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"  [KANFIS] Built — {n_params:,} params | "
        f"groups: {list(group_map.keys())} | "
        f"rules: {n_rules} | λ_L1: {l1_lambda} | "
        f"focal_γ: {focal_gamma} | diversity_w: {diversity_weight}"
    )
    return model


if __name__ == "__main__":
    torch.manual_seed(42)
    group_map  = {"metabolic": [0,1,2,3], "cardiovascular": [4,5,6], "demographic": [7]}
    model      = build_kanfis(8, group_map, n_rules=10, l1_lambda=1e-3)
    x_fake     = torch.randn(32, 8)
    y_fake     = torch.randint(0, 2, (32,)).float()
    logit, fz  = model(x_fake, return_rules=True)
    loss, bd   = model.composite_loss(logit, y_fake, l1_scale=0.0)
    print(f"Logit range : [{logit.min():.3f}, {logit.max():.3f}]")
    print(f"Centroid rng: [{fz['centroid'].min():.3f}, {fz['centroid'].max():.3f}]")
    print(f"Loss={loss:.4f}  BCE(focal)={bd['bce']:.4f}  "
          f"L1={bd['l1']:.4f}  Diversity={bd['diversity']:.4f}")