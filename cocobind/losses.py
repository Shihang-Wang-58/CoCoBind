"""
CoCoBind Loss Functions

1. L_int: Interaction binary classification loss
2. L_site: Binding site sequence labeling loss
3. L_cons: Interaction↔Site consistency constraint (key innovation)
4. L_ctr: Contrastive learning loss (optional)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class InteractionLoss(nn.Module):
    """Interaction binary classification loss (BCEWithLogitsLoss)."""
    
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(
        self,
        logits: torch.Tensor,      # [B]
        targets: torch.Tensor,     # [B]
    ) -> torch.Tensor:
        return self.criterion(logits, targets.float())


class SiteLoss(nn.Module):
    """
    Binding site sequence labeling loss.
    site_only_positive=True: Only compute for samples with interactions==1 and valid binding_site_index
    site_only_positive=False: Compute for all samples (negative samples use all-zero labels)
    """
    
    def __init__(self, site_only_positive: bool = True):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.site_only_positive = site_only_positive
    
    def forward(
        self,
        site_logits: torch.Tensor,     # [B, L]
        site_labels: torch.Tensor,     # [B, L]
        rna_mask: torch.Tensor,        # [B, L]
        has_site_labels: torch.Tensor, # [B] whether sample has valid site labels
        interactions: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute site loss, only for valid samples and positions."""
        loss_per_pos = self.criterion(site_logits, site_labels)
        loss_per_pos = loss_per_pos * rna_mask
        
        sample_lens = rna_mask.sum(dim=1).clamp(min=1)
        loss_per_sample = loss_per_pos.sum(dim=1) / sample_lens
        
        if self.site_only_positive:
            valid_mask = has_site_labels.float()
        else:
            valid_mask = torch.ones_like(has_site_labels).float()
        
        n_valid = valid_mask.sum().clamp(min=1)
        loss = (loss_per_sample * valid_mask).sum() / n_valid
        
        return loss


class ConsistencyLoss(nn.Module):
    """
    Consistency Constraint Loss (Key Innovation)
    
    Aggregates site probabilities into p_any_site, then aligns with interaction probability.
    
    Aggregation methods:
    - noisy_or: p_any = 1 - Π(1 - s_i), computed in log-space for numerical stability
    - max: p_any = max(s_i)
    - mean: p_any = mean(s_i)
    
    Loss: MSE(sigmoid(interaction_logit), p_any)
    """
    
    def __init__(self, aggregation: str = "noisy_or", eps: float = 1e-7):
        super().__init__()
        self.aggregation = aggregation
        self.eps = eps
    
    def _noisy_or(
        self,
        site_probs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Noisy-OR aggregation: p_any = 1 - Π(1 - s_i)
        Computed in log-space for numerical stability.
        """
        probs_clamped = site_probs.clamp(self.eps, 1 - self.eps)
        log_complement = torch.log(1 - probs_clamped)
        log_complement = log_complement * mask
        log_prod = log_complement.sum(dim=1)
        p_any = 1 - torch.exp(log_prod)
        return p_any.clamp(self.eps, 1 - self.eps)
    
    def _max_aggregation(
        self,
        site_probs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Max aggregation."""
        probs_masked = site_probs.clone()
        probs_masked[mask == 0] = -float('inf')
        p_any = probs_masked.max(dim=1)[0]
        return p_any.clamp(self.eps, 1 - self.eps)
    
    def _mean_aggregation(
        self,
        site_probs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean aggregation."""
        probs_sum = (site_probs * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1)
        p_any = probs_sum / count
        return p_any.clamp(self.eps, 1 - self.eps)
    
    def forward(
        self,
        interaction_logit: torch.Tensor,
        site_logits: torch.Tensor,
        rna_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute consistency loss."""
        p_int = torch.sigmoid(interaction_logit)
        site_probs = torch.sigmoid(site_logits)
        
        if self.aggregation == "noisy_or":
            p_any_site = self._noisy_or(site_probs, rna_mask)
        elif self.aggregation == "max":
            p_any_site = self._max_aggregation(site_probs, rna_mask)
        elif self.aggregation == "mean":
            p_any_site = self._mean_aggregation(site_probs, rna_mask)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        loss = F.mse_loss(p_int, p_any_site)
        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive Learning Loss (InfoNCE)
    Brings positive (rna_pool, mol_pool) pairs closer, uses in-batch negatives.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        rna_pool: torch.Tensor,
        mol_pool: torch.Tensor,
        interactions: torch.Tensor,
    ) -> torch.Tensor:
        """InfoNCE contrastive loss."""
        rna_norm = F.normalize(rna_pool, p=2, dim=1)
        mol_norm = F.normalize(mol_pool, p=2, dim=1)
        
        logits = torch.matmul(rna_norm, mol_norm.T) / self.temperature
        
        B = logits.shape[0]
        labels = torch.arange(B, device=logits.device)
        
        pos_mask = interactions > 0.5
        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        loss_rna2mol = F.cross_entropy(logits[pos_mask], labels[pos_mask], reduction='mean')
        loss_mol2rna = F.cross_entropy(logits.T[pos_mask], labels[pos_mask], reduction='mean')
        
        return (loss_rna2mol + loss_mol2rna) / 2


class MultiTaskLoss(nn.Module):
    """
    Multi-task Loss Combination
    L = L_int + λ_site * L_site + λ_cons * L_cons (+ λ_ctr * L_ctr)
    
    Ablation switches:
    - lambda_cons=0: Disable consistency constraint (no_consistency)
    - use_contrastive=False: Disable contrastive learning (no_contrastive)
    - site_only_positive=True: Site supervision only on positive samples
    """
    
    def __init__(
        self,
        lambda_site: float = 1.0,
        lambda_cons: float = 0.5,
        lambda_ctr: float = 0.1,
        use_contrastive: bool = False,
        cons_aggregation: str = "noisy_or",
        site_only_positive: bool = True,
    ):
        super().__init__()
        
        self.lambda_site = lambda_site
        self.lambda_cons = lambda_cons
        self.lambda_ctr = lambda_ctr
        self.use_contrastive = use_contrastive
        
        self.int_loss = InteractionLoss()
        self.site_loss = SiteLoss(site_only_positive=site_only_positive)
        self.cons_loss = ConsistencyLoss(aggregation=cons_aggregation) if lambda_cons > 0 else None
        self.ctr_loss = ContrastiveLoss() if use_contrastive else None
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses.
        
        Returns:
            dict with keys: loss, loss_int, loss_site, loss_cons, (loss_ctr)
        """
        interaction_logit = outputs["interaction_logit"]
        site_logits = outputs["site_logits"]
        rna_pool = outputs["rna_pool"]
        mol_pool = outputs["mol_pool"]
        
        interactions = batch["interactions"]
        site_labels = batch["site_labels"]
        rna_mask = batch["rna_mask"]
        has_site_labels = batch["has_site_labels"]
        
        loss_int = self.int_loss(interaction_logit, interactions)
        loss_site = self.site_loss(site_logits, site_labels, rna_mask, has_site_labels, interactions)
        
        if self.cons_loss is not None and self.lambda_cons > 0:
            loss_cons = self.cons_loss(interaction_logit, site_logits, rna_mask)
        else:
            loss_cons = torch.tensor(0.0, device=interaction_logit.device)
        
        total_loss = loss_int + self.lambda_site * loss_site + self.lambda_cons * loss_cons
        
        result = {
            "loss": total_loss,
            "loss_int": loss_int.detach(),
            "loss_site": loss_site.detach(),
            "loss_cons": loss_cons.detach() if isinstance(loss_cons, torch.Tensor) else loss_cons,
        }
        
        if self.use_contrastive and self.ctr_loss is not None:
            loss_ctr = self.ctr_loss(rna_pool, mol_pool, interactions)
            total_loss = total_loss + self.lambda_ctr * loss_ctr
            result["loss"] = total_loss
            result["loss_ctr"] = loss_ctr.detach()
        
        return result
