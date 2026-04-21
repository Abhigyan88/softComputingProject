"""
kanfis_model.py  (v2 — all bugs fixed)
===============
Kolmogorov-Arnold Neuro-Fuzzy Inference System (KANFIS)

Bug-fix changelog vs original:
  FIX 1 — GaussianRBF: add linear bypass  edge(x) = w*x + phi(x)
           Without it RBF≈0 at init, blocking all gradient from inputs.
  FIX 2 — GroupKANLayer: LayerNorm → BatchNorm1d
           LayerNorm made every sample statistically identical; BN preserves
           inter-sample discriminative variance.
  FIX 3 — IT2FuzzyLayer: dist.sum → per_feat.mean (true additive aggregation)
           and sigma init ≈ 0.5 (not 1.0) for wider firing range.
  FIX 4 — SparseRuleHead: Kaiming init → small init std=0.05
           Large initial weights caused all-one-class prediction from epoch 1;
           L1 then zeroed everything; early-stop fired before any learning.
  FIX 5 — Bottleneck: Tanh → BatchNorm1d + GELU (no saturation).
  FIX 6 — composite_loss: l1_scale warm-up parameter (see train.py).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
# 1.  GAUSSIAN RBF KAN EDGE  (FIX 1: linear bypass added)
# ═══════════════════════════════════════════════════════════
class GaussianRBF(nn.Module):
    """
    Univariate KAN edge:  out(x) = linear_weight*x  +  Σ_k w_k·φ_k(x)

    The linear term is the critical missing piece from the original code.
    In pykan the edge is  w_res*SiLU(x) + w_spline*phi(x).  Without the
    residual, random w_k (mean≈0) cause RBF≈0 at init, so NO gradient
    signal reaches input features → entire model dead from epoch 1.
    """
    def __init__(self, n_basis: int = 8):
        super().__init__()
        self.n_basis = n_basis
        self.centres = nn.Parameter(
            torch.linspace(-3.0, 3.0, n_basis).unsqueeze(0)   # (1, n_basis)
        )
        self.log_widths = nn.Parameter(torch.zeros(1, n_basis))
        # Near-zero init: linear term dominates at start; RBF learns gradually
        self.weights = nn.Parameter(torch.randn(n_basis) * 0.1)
        # Linear bypass: start as near-identity passthrough
        self.linear_weight = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch,)  →  out: (batch,)"""
        linear_out = self.linear_weight * x                    # (batch,)
        x_exp  = x.unsqueeze(-1)                               # (batch, 1)
        sigma  = torch.exp(self.log_widths) + 1e-6             # (1, n_basis)
        rbf    = torch.exp(-0.5 * ((x_exp - self.centres) / sigma) ** 2)
        rbf_out = (rbf * self.weights).sum(-1)                 # (batch,)
        return linear_out + rbf_out


# ═══════════════════════════════════════════════════════════
# 2.  GROUP KAN LAYER  (FIX 2: BatchNorm1d replaces LayerNorm)
# ═══════════════════════════════════════════════════════════
class GroupKANLayer(nn.Module):
    """
    KAN layer with clinical domain grouping and shared RBF activations.

    FIX 2: LayerNorm normalises each SAMPLE across its 48 features, making
    every patient's representation statistically identical and destroying
    the between-patient discrimination needed for classification.
    BatchNorm1d normalises each FEATURE across the batch — correct behaviour.
    """
    def __init__(self, n_features: int, group_map: dict,
                 out_dim: int = 16, n_basis: int = 8):
        super().__init__()
        self.group_map = group_map
        self.out_dim   = out_dim
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
        """x: (batch, n_features)  →  (batch, n_groups * out_dim)"""
        group_outputs = []
        for name, indices in self.group_map.items():
            x_g = x[:, indices]                           # (batch, g_size)
            rbf_out = torch.stack(
                [self.group_rbf[name](x_g[:, i]) for i in range(x_g.size(1))],
                dim=1,
            )                                             # (batch, g_size)
            proj_out = self.group_proj[name](rbf_out)     # (batch, out_dim)
            group_outputs.append(proj_out)
        out = torch.cat(group_outputs, dim=-1)            # (batch, n_groups*out_dim)
        return self.batch_norm(out)


# ═══════════════════════════════════════════════════════════
# 3.  IT2 FUZZY LAYER  (FIX 3: mean aggregation + smaller sigma)
# ═══════════════════════════════════════════════════════════
class IT2FuzzyLayer(nn.Module):
    """
    Interval Type-2 fuzzy antecedent layer.

    FIX 3a — True additive aggregation:
      Old: exp(-0.5 * dist.sum(-1))  = product T-norm
           With in_dim=48: exp(-24) ≈ 3e-11 → ALL firing strengths ≈ 0
      New: per_feature.mean(-1)      = true additive aggregation
           Stays in (0,1] regardless of dimensionality.

    FIX 3b — Smaller initial sigma (≈0.5 not 1.0):
      sigma=1.0: two inputs 0.5 units apart → 88% similar firing (barely discriminative)
      sigma=0.5: same distance → 61% similarity; range [0.01, 1.0] across space
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
        """Returns (batch, n_rules) via mean over feature dims."""
        x_exp       = x.unsqueeze(1)                          # (batch,1,in_dim)
        c_exp       = centres.unsqueeze(0)                    # (1,n_rules,in_dim)
        s_exp       = sigmas.unsqueeze(0)
        dist        = ((x_exp - c_exp) / s_exp) ** 2         # (batch,n_rules,in_dim)
        per_feature = torch.exp(-0.5 * dist)                  # ∈ (0,1]
        return per_feature.mean(-1)                           # (batch, n_rules)

    def forward(self, x: torch.Tensor) -> dict:
        upper    = self._gaussian_mf(x, self.centres, self.sigma_upper)
        lower    = self._gaussian_mf(x, self.centres, self.sigma_lower)
        centroid = 0.5 * (upper + lower)
        return {"upper": upper, "lower": lower, "centroid": centroid}


# ═══════════════════════════════════════════════════════════
# 4.  SPARSE RULE HEAD  (FIX 4: small weight init)
# ═══════════════════════════════════════════════════════════
class SparseRuleHead(nn.Module):
    """
    L1-regularised consequent layer.

    FIX 4: Kaiming init (std≈1) with 10 rules and firing≈0.6 gives initial
    logit ≈ ±1.9 → model predicts all-one-class immediately → L1 zeroes
    weights → early stop fires at epoch 20 with useless model.
    std=0.05 gives initial logit ≈ ±0.09 → neutral start → real learning.
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
# 5.  FULL KANFIS MODEL
# ═══════════════════════════════════════════════════════════
class KANFIS(nn.Module):
    def __init__(self, n_features: int, group_map: dict,
                 kan_out_dim: int = 16, n_rules: int = 10,
                 kan_n_basis: int = 8, l1_lambda: float = 1e-3):
        super().__init__()
        self.l1_lambda  = l1_lambda
        self.n_features = n_features
        self.group_map  = group_map

        # Stage 1 — Group KAN
        self.group_kan = GroupKANLayer(
            n_features=n_features, group_map=group_map,
            out_dim=kan_out_dim, n_basis=kan_n_basis,
        )
        kan_total_out = len(group_map) * kan_out_dim

        # Bottleneck: KAN space → compact fuzzy input (FIX 5: BN+GELU, not Tanh)
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
        kan_out   = self.group_kan(x)           # (batch, kan_total_out)
        fuzzy_in  = self.fuzzy_proj(kan_out)    # (batch, fuzzy_in_dim)
        fuzzy_out = self.fuzzy_layer(fuzzy_in)  # dict of (batch, n_rules)
        firing    = fuzzy_out["centroid"]       # (batch, n_rules)
        logit     = self.rule_head(firing)      # (batch, 1)
        if return_rules:
            return logit, fuzzy_out
        return logit

    def composite_loss(self, logit: torch.Tensor, targets: torch.Tensor,
                       l1_scale: float = 1.0) -> tuple:
        """
        BCE + L1 sparsity.  l1_scale ramps 0→1 over warmup epochs
        (FIX 6) to prevent L1 from zeroing weights before any learning.
        """
        bce   = F.binary_cross_entropy_with_logits(logit.squeeze(-1), targets.float())
        l1    = self.rule_head.l1_penalty
        total = bce + l1_scale * self.l1_lambda * l1
        return total, {"bce": bce.item(), "l1": l1.item()}

    def get_active_rules(self, threshold: float = 0.01) -> list:
        weights = self.rule_head.rule_weights.weight.detach().abs().squeeze(0)
        return (weights > threshold).nonzero(as_tuple=True)[0].tolist()

    def prune_rules(self, threshold: float = 0.01) -> None:
        with torch.no_grad():
            mask = self.rule_head.rule_weights.weight.abs() < threshold
            self.rule_head.rule_weights.weight[mask] = 0.0
        active = self.get_active_rules(threshold)
        print(f"[Prune] {len(active)} active rules remaining: {active}")


# ═══════════════════════════════════════════════════════════
# 6.  FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════
def build_kanfis(n_features: int, group_map: dict,
                 n_rules: int = 10, l1_lambda: float = 1e-3) -> KANFIS:
    model = KANFIS(
        n_features=n_features, group_map=group_map,
        kan_out_dim=16, n_rules=n_rules, kan_n_basis=8, l1_lambda=l1_lambda,
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[KANFIS] Built — {n_params:,} params | groups: {list(group_map.keys())}"
          f" | rules: {n_rules} | λ_L1: {l1_lambda}")
    return model


if __name__ == "__main__":
    torch.manual_seed(42)
    group_map  = {"metabolic": [0,1,2,3], "cardiovascular": [4,5,6], "demographic": [7]}
    model      = build_kanfis(8, group_map, n_rules=10, l1_lambda=1e-3)
    x_fake     = torch.randn(32, 8)
    y_fake     = torch.randint(0, 2, (32,)).float()
    logit, fz  = model(x_fake, return_rules=True)
    loss, bd   = model.composite_loss(logit, y_fake, l1_scale=0.0)
    print(f"Logit range : [{logit.min():.3f}, {logit.max():.3f}]  (should vary, not constant)")
    print(f"Centroid rng: [{fz['centroid'].min():.3f}, {fz['centroid'].max():.3f}]")
    print(f"Loss={loss:.4f}  BCE={bd['bce']:.4f}  L1={bd['l1']:.4f}")