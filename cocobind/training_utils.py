"""
CoCoBind Training Utilities

- Learning rate schedulers (Warmup + Cosine Annealing)
- EMA (Exponential Moving Average)
- Focal Loss
- Gradient accumulation
"""
import math
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Warmup + Cosine Annealing learning rate scheduler.
    
    Learning rate schedule:
    1. Warmup phase: lr linearly increases from warmup_lr to peak_lr
    2. Cosine phase: lr decays from peak_lr to min_lr using cosine annealing
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        warmup_lr: float = 1e-6,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        
        self.peak_lrs = [group['lr'] for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [
                self.warmup_lr + alpha * (peak_lr - self.warmup_lr)
                for peak_lr in self.peak_lrs
            ]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + cosine_decay * (peak_lr - self.min_lr)
                for peak_lr in self.peak_lrs
            ]


class EMA:
    """
    Exponential Moving Average for model parameters.
    
    Used for training stabilization and improved generalization.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply shadow parameters (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters (after evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    When gamma=0, degenerates to standard BCE.
    """
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        focal_loss = alpha_t * focal_weight * bce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingBCE(nn.Module):
    """BCE Loss with label smoothing."""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_smooth = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, targets_smooth)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
) -> Optional[_LRScheduler]:
    """Create learning rate scheduler based on configuration."""
    scheduler_type = config.get("scheduler", "cosine_warmup")
    
    if scheduler_type == "cosine_warmup":
        return WarmupCosineScheduler(
            optimizer,
            warmup_epochs=config.get("warmup_epochs", 10),
            total_epochs=config.get("epochs", 300),
            warmup_lr=config.get("warmup_lr", 1e-6),
            min_lr=config.get("min_lr", 1e-6),
        )
    
    elif scheduler_type == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(
            optimizer,
            T_max=config.get("epochs", 300),
            eta_min=config.get("min_lr", 1e-6),
        )
    
    elif scheduler_type == "step":
        from torch.optim.lr_scheduler import StepLR
        return StepLR(
            optimizer,
            step_size=config.get("step_size", 50),
            gamma=config.get("gamma", 0.5),
        )
    
    elif scheduler_type == "plateau":
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=10,
            min_lr=config.get("min_lr", 1e-6),
        )
    
    else:
        return None


def get_optimizer(
    model: nn.Module,
    config: Dict[str, Any],
) -> torch.optim.Optimizer:
    """Create optimizer with optional layer-wise learning rates."""
    lr = config.get("lr", 1e-4)
    weight_decay = config.get("weight_decay", 1e-4)
    
    decay_params = []
    no_decay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "norm" in name or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    optimizer_type = config.get("optimizer", "adamw")
    
    if optimizer_type == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.999))
    elif optimizer_type == "adam":
        return torch.optim.Adam(param_groups, lr=lr)
    elif optimizer_type == "sgd":
        return torch.optim.SGD(param_groups, lr=lr, momentum=0.9)
    else:
        return torch.optim.AdamW(param_groups, lr=lr)
