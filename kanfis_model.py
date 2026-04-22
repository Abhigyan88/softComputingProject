"""
kanfis_model.py  (v7 — The Precision & Logic Fix)
===============
Kolmogorov-Arnold Neuro-Fuzzy Inference System (KANFIS)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianRBF(nn.Module):
    def __init__(self, n_basis: int = 8):
        super().__init__()
        self.centres       = nn.Parameter(torch.linspace(-3.0, 3.0, n_basis).unsqueeze(0))
        self.log_widths    = nn.Parameter(torch.zeros(1, n_basis))
        self.weights       = nn.Parameter(torch.randn(n_basis) * 0.1)
        self.linear_weight = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_out = self.linear_weight * x
        sigma  = torch.exp(self.log_widths) + 1e-6
        rbf    = torch.exp(-0.5 * ((x.unsqueeze(-1) - self.centres) / sigma) ** 2)
        return linear_out + (rbf * self.weights).sum(-1)

class GroupKANLayer(nn.Module):
    def __init__(self, n_features: int, group_map: dict, out_dim: int = 16, n_basis: int = 8):
        super().__init__()
        self.group_map  = group_map
        self.group_rbf  = nn.ModuleDict()
        self.group_proj = nn.ModuleDict()

        for name, indices in group_map.items():
            self.group_rbf[name]  = GaussianRBF(n_basis)
            self.group_proj[name] = nn.Linear(len(indices), out_dim, bias=True)
            nn.init.kaiming_normal_(self.group_proj[name].weight, nonlinearity="relu")

        self.batch_norm = nn.BatchNorm1d(len(group_map) * out_dim, momentum=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        group_outputs = []
        for name, indices in self.group_map.items():
            x_g = x[:, indices]
            rbf_out = torch.stack([self.group_rbf[name](x_g[:, i]) for i in range(x_g.size(1))], dim=1)
            group_outputs.append(self.group_proj[name](rbf_out))
        return self.batch_norm(torch.cat(group_outputs, dim=-1))

class IT2FuzzyLayer(nn.Module):
    def __init__(self, in_dim: int, n_rules: int = 8):
        super().__init__()
        self.centres = nn.Parameter(torch.empty(n_rules, in_dim))
        nn.init.uniform_(self.centres, -2.0, 2.0)
        self.log_sigma_upper = nn.Parameter(torch.full((n_rules, in_dim), -0.2))
        self.log_sigma_lower = nn.Parameter(torch.full((n_rules, in_dim), -0.8))
        self.feature_gate = nn.Parameter(torch.zeros(n_rules, in_dim))

    @property
    def sigma_upper(self): return torch.exp(self.log_sigma_upper) + 1e-6
    @property
    def sigma_lower(self): return torch.minimum(torch.exp(self.log_sigma_lower) + 1e-6, self.sigma_upper * 0.9)

    def _gated_gaussian_mf(self, x, sigmas):
        dist = ((x.unsqueeze(1) - self.centres.unsqueeze(0)) / sigmas.unsqueeze(0)) ** 2
        gate = torch.sigmoid(self.feature_gate)
        gate_norm = gate / gate.sum(-1, keepdim=True).clamp(min=0.1)
        weighted_log = (gate_norm.unsqueeze(0) * torch.log(torch.exp(-0.5 * dist).clamp(min=1e-6))).sum(-1)
        return torch.exp(weighted_log)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        upper = self._gated_gaussian_mf(x, self.sigma_upper)
        lower = self._gated_gaussian_mf(x, self.sigma_lower)
        return 0.5 * (upper + lower) # Centroid firing strength

class SparseRuleHead(nn.Module):
    def __init__(self, n_rules: int):
        super().__init__()
        self.rule_weights = nn.Linear(n_rules, 1, bias=True)
        nn.init.normal_(self.rule_weights.weight, mean=0.0, std=0.1)
        nn.init.constant_(self.rule_weights.bias, -0.5) # Slight negative bias to curb false positives

    def forward(self, firing_strengths: torch.Tensor) -> torch.Tensor:
        return self.rule_weights(firing_strengths)

class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    def forward(self, logits): return logits / self.temperature.clamp(min=0.05)

def calibrate_temperature(model, X_val, y_val, device):
    ts = TemperatureScaling().to(device)
    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(X_val).to(device)).squeeze(-1).detach()
    labels = torch.FloatTensor(y_val).to(device)
    optimiser = torch.optim.LBFGS([ts.temperature], lr=0.01, max_iter=100)
    def _eval():
        optimiser.zero_grad()
        loss = F.binary_cross_entropy_with_logits(ts(logits), labels)
        loss.backward()
        return loss
    optimiser.step(_eval)
    return ts

class KANFIS(nn.Module):
    def __init__(self, n_features, group_map, kan_out_dim=16, n_rules=10, l1_lambda=1e-3, focal_gamma=2.0, diversity_weight=0.2):
        super().__init__()
        
        # RESTORED ATTRIBUTES: Required by train.py for save_model()
        self.n_features = n_features
        self.group_map = group_map
        
        self.l1_lambda = l1_lambda
        self.focal_gamma = focal_gamma
        self.diversity_weight = diversity_weight
        
        self.group_kan = GroupKANLayer(n_features, group_map, kan_out_dim)
        self.kan_proj = nn.Linear(len(group_map) * kan_out_dim, n_rules)
        self.fuzzy_layer = IT2FuzzyLayer(n_features, n_rules)
        self.rule_head = SparseRuleHead(n_rules)

    def forward(self, x: torch.Tensor, return_rules: bool = False):
        kan_out = self.group_kan(x)
        # KAN acts as a bounded micro-adjustment (+/- 0.15) to fuzzy logic
        kan_correction = 0.15 * torch.tanh(self.kan_proj(kan_out))
        
        centroid = self.fuzzy_layer(x)
        # Strictly bound firing strengths to [0, 1] to preserve fuzzy math
        firing = torch.clamp(centroid + kan_correction, 0.0, 1.0)
        
        logit = self.rule_head(firing)
        return (logit, centroid) if return_rules else logit

    def composite_loss(self, logit: torch.Tensor, targets: torch.Tensor, l1_scale: float = 1.0) -> tuple:
        targets_f = targets.float()
        bce = F.binary_cross_entropy_with_logits(logit.squeeze(-1), targets_f, reduction="none")
        probs = torch.sigmoid(logit.squeeze(-1))
        p_t = probs * targets_f + (1 - probs) * (1 - targets_f)
        
        # Alpha is fixed to 0.5 because SMOTE already perfectly balances the dataset
        focal = (0.5 * (1 - p_t) ** self.focal_gamma * bce).mean()
        
        l1 = self.rule_head.rule_weights.weight.abs().mean()
        
        c = self.fuzzy_layer.centres                           
        diff = c.unsqueeze(0) - c.unsqueeze(1)                   
        dist = (diff.pow(2).sum(-1) + 1e-8).sqrt()               
        diversity = F.relu(4.0 - dist).triu(diagonal=1).mean()

        total = focal + (l1_scale * self.l1_lambda * l1) + (self.diversity_weight * diversity)
        return total, {"bce": focal.item(), "l1": l1.item(), "diversity": diversity.item()}

    def get_active_rules(self, threshold: float = 0.01) -> list:
        weights = self.rule_head.rule_weights.weight.detach().abs().squeeze(0)
        return (weights > threshold).nonzero(as_tuple=True)[0].tolist()

    def prune_rules(self, threshold: float = 0.01) -> None:
        with torch.no_grad():
            mask = self.rule_head.rule_weights.weight.abs() < threshold
            self.rule_head.rule_weights.weight[mask] = 0.0

def build_kanfis(n_features, group_map, n_rules=10, l1_lambda=1e-3, focal_gamma=2.0, diversity_weight=0.2, **kwargs) -> KANFIS:
    return KANFIS(n_features, group_map, 16, n_rules, l1_lambda, focal_gamma, diversity_weight)