"""
kanfis_model.py  (v6 — unblocked rule head & diversity)
===============
Kolmogorov-Arnold Neuro-Fuzzy Inference System (KANFIS)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
# 1.  GAUSSIAN RBF KAN EDGE
# ═══════════════════════════════════════════════════════════
class GaussianRBF(nn.Module):
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
    def __init__(self, in_dim: int, n_rules: int = 8):
        super().__init__()
        self.in_dim  = in_dim
        self.n_rules = n_rules

        self.centres = nn.Parameter(torch.empty(n_rules, in_dim))
        nn.init.uniform_(self.centres, -2.0, 2.0)

        self.log_sigma_upper = nn.Parameter(torch.full((n_rules, in_dim), -0.2))
        self.log_sigma_lower = nn.Parameter(torch.full((n_rules, in_dim), -0.8))
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
        x_exp = x.unsqueeze(1)           
        c_exp = centres.unsqueeze(0)     
        s_exp = sigmas.unsqueeze(0)      
        dist  = ((x_exp - c_exp) / s_exp) ** 2
        rbf   = torch.exp(-0.5 * dist)  

        gate      = torch.sigmoid(self.feature_gate)               
        gate_norm = gate / gate.sum(-1, keepdim=True).clamp(min=0.1)  

        log_rbf = torch.log(rbf.clamp(min=1e-6))                   
        weighted_log = (gate_norm.unsqueeze(0) * log_rbf).sum(-1)  
        return torch.exp(weighted_log)                              

    def forward(self, x: torch.Tensor) -> dict:
        upper    = self._gated_gaussian_mf(x, self.centres, self.sigma_upper)
        lower    = self._gated_gaussian_mf(x, self.centres, self.sigma_lower)
        centroid = 0.5 * (upper + lower)
        return {"upper": upper, "lower": lower, "centroid": centroid}

    def get_rule_feature_importance(self) -> torch.Tensor:
        gate = torch.sigmoid(self.feature_gate).detach()
        return gate / gate.sum(-1, keepdim=True).clamp(min=0.1)


# ═══════════════════════════════════════════════════════════
# 4.  SPARSE RULE HEAD
# ═══════════════════════════════════════════════════════════
class SparseRuleHead(nn.Module):
    def __init__(self, n_rules: int, n_classes: int = 1, dropout: float = 0.10):
        super().__init__()
        self.hidden = nn.Linear(n_rules, n_rules)
        # RESTORED BIAS: Required for proper logit centering
        self.output = nn.Linear(n_rules, n_classes, bias=True)
        self.dropout = nn.Dropout(dropout)

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
        # L1 ON FIRST LAYER ONLY: Prevents strangling the network output
        return self.hidden.weight.abs().mean()


# ═══════════════════════════════════════════════════════════
# 5.  TEMPERATURE SCALING
# ═══════════════════════════════════════════════════════════
class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.05)

def calibrate_temperature(
    model: "KANFIS", X_val: "np.ndarray", y_val: "np.ndarray", device: torch.device
) -> "TemperatureScaling":
    import numpy as np
    ts = TemperatureScaling().to(device)
    model.eval()

    with torch.no_grad():
        logits = model(torch.FloatTensor(X_val).to(device)).squeeze(-1).detach()
    labels = torch.FloatTensor(y_val).to(device)

    optimiser = torch.optim.LBFGS([ts.temperature], lr=0.01, max_iter=100, line_search_fn="strong_wolfe")

    def _eval():
        optimiser.zero_grad()
        loss = F.binary_cross_entropy_with_logits(ts(logits), labels)
        loss.backward()
        return loss

    optimiser.step(_eval)
    T = ts.temperature.item()
    print(f"  [TemperatureScaling] T = {T:.4f}")
    return ts


# ═══════════════════════════════════════════════════════════
# 6.  FULL KANFIS MODEL
# ═══════════════════════════════════════════════════════════
class KANFIS(nn.Module):
    def __init__(
        self,
        n_features: int,
        group_map: dict,
        kan_out_dim: int = 16,
        n_rules: int = 10,
        kan_n_basis: int = 8,
        l1_lambda: float = 5e-5,        # REDUCED L1
        focal_gamma: float = 2.0,
        alpha_pos: float = 0.75,
        diversity_weight: float = 0.2,  # INCREASED DIVERSITY WEIGHT
        spread_weight: float = 0.0,     # DISABLED SPREAD WEIGHT
    ):
        super().__init__()
        self.l1_lambda        = l1_lambda
        self.n_features       = n_features
        self.group_map        = group_map
        self.focal_gamma      = focal_gamma
        self.alpha_pos        = alpha_pos
        self.diversity_weight = diversity_weight
        self.spread_weight    = spread_weight

        self.group_kan = GroupKANLayer(
            n_features=n_features, group_map=group_map, out_dim=kan_out_dim, n_basis=kan_n_basis
        )
        kan_total_out = len(group_map) * kan_out_dim

        self.kan_proj = nn.Linear(kan_total_out, n_rules)
        nn.init.xavier_uniform_(self.kan_proj.weight)
        nn.init.zeros_(self.kan_proj.bias)

        self.fuzzy_layer = IT2FuzzyLayer(in_dim=n_features, n_rules=n_rules)
        self.firing_bn = nn.BatchNorm1d(n_rules, momentum=0.1)
        self.rule_head = SparseRuleHead(n_rules=n_rules, n_classes=1, dropout=0.10)

    def forward(self, x: torch.Tensor, return_rules: bool = False):
        kan_out        = self.group_kan(x)
        kan_correction = 0.5 * torch.tanh(self.kan_proj(kan_out))

        fuzzy_out = self.fuzzy_layer(x)
        centroid  = fuzzy_out["centroid"]                  
        firing = centroid + kan_correction                 

        if firing.size(0) > 1:
            firing_normed = self.firing_bn(firing)
        else:
            firing_normed = firing  

        logit = self.rule_head(firing_normed)
        if return_rules:
            return logit, fuzzy_out
        return logit

    def _rule_diversity_loss(self, min_distance: float = 4.0) -> torch.Tensor:
        # INCREASED MIN DISTANCE: Forces rules apart
        c    = self.fuzzy_layer.centres                           
        diff = c.unsqueeze(0) - c.unsqueeze(1)                   
        dist = (diff.pow(2).sum(-1) + 1e-8).sqrt()               
        penalty = F.relu(min_distance - dist).triu(diagonal=1).mean()
        return penalty

    def _spread_loss(self, logit: torch.Tensor) -> torch.Tensor:
        if logit.size(0) <= 4:
            return torch.tensor(0.0, device=logit.device)
        probs = torch.sigmoid(logit.squeeze(-1))
        return -probs.var()

    def composite_loss(
        self, logit: torch.Tensor, targets: torch.Tensor, l1_scale: float = 1.0
    ) -> tuple:
        logit_sq  = logit.squeeze(-1)
        targets_f = targets.float()

        bce_each = F.binary_cross_entropy_with_logits(logit_sq, targets_f, reduction="none")
        probs = torch.sigmoid(logit_sq)
        p_t   = probs * targets_f + (1 - probs) * (1 - targets_f)

        alpha_t = self.alpha_pos * targets_f + (1 - self.alpha_pos) * (1 - targets_f)
        focal   = (alpha_t * (1 - p_t) ** self.focal_gamma * bce_each).mean()

        l1        = self.rule_head.l1_penalty          
        diversity = self._rule_diversity_loss()
        spread    = self._spread_loss(logit)            

        total = (
            focal
            + l1_scale * self.l1_lambda * l1
            + self.diversity_weight * diversity
            + self.spread_weight * spread          
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
    l1_lambda: float = 5e-5,           
    focal_gamma: float = 2.0,
    alpha_pos: float = 0.75,
    diversity_weight: float = 0.2,    
    spread_weight: float = 0.0,        
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
    return model