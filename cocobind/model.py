"""
CoCoBind Model Architecture

Multi-task model for RNA-small molecule interaction prediction and binding site identification.
Uses Cross-Attention for RNA-molecule fusion with optional ablation modes.

Components:
- CrossAttention: RNA tokens query molecule tokens
- MolTokenProjector: Projects molecular fingerprints to K tokens
- RNAProjector: Projects RNA-FM embeddings to model dimension
- InteractionHead: Predicts binding interaction probability
- SiteHead: Predicts per-residue binding site probabilities
"""
import math
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """
    Cross-Attention layer.
    RNA tokens serve as Query, molecule tokens serve as Key/Value.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)
    
    def forward(
        self,
        query: torch.Tensor,      # [B, L_q, d]
        key: torch.Tensor,        # [B, L_k, d]
        value: torch.Tensor,      # [B, L_k, d]
        query_mask: torch.Tensor = None,  # [B, L_q]
        key_mask: torch.Tensor = None     # [B, L_k]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            output: [B, L_q, d]
            attn_weights: [B, n_heads, L_q, L_k]
        """
        B, L_q, _ = query.shape
        L_k = key.shape[1]
        
        # Project
        Q = self.q_proj(query).view(B, L_q, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(key).view(B, L_k, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(value).view(B, L_k, self.n_heads, self.d_head).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, L_q, L_k]
        
        # Mask
        if key_mask is not None:
            mask = key_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)  # [B, H, L_q, d_head]
        context = context.transpose(1, 2).contiguous().view(B, L_q, self.d_model)
        
        output = self.out_proj(context)
        
        return output, attn_weights


class MolTokenProjector(nn.Module):
    """
    Project molecular fingerprint and reshape into K tokens.
    ECFP4 [n_bits] -> [K, d_model]
    """
    
    def __init__(self, n_bits: int, d_model: int, n_tokens: int):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model
        
        self.proj = nn.Sequential(
            nn.Linear(n_bits, d_model * n_tokens),
            nn.LayerNorm(d_model * n_tokens),
            nn.GELU(),
            nn.Dropout(0.1),
        )
    
    def forward(self, mol_fp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mol_fp: [B, n_bits]
        Returns:
            mol_tokens: [B, K, d_model]
        """
        B = mol_fp.shape[0]
        x = self.proj(mol_fp)
        return x.view(B, self.n_tokens, self.d_model)


class RNAProjector(nn.Module):
    """RNA embedding projection layer."""
    
    def __init__(self, d_rna: int, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_rna, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.1),
        )
    
    def forward(self, rna_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rna_embed: [B, L, d_rna]
        Returns:
            rna_tokens: [B, L, d_model]
        """
        return self.proj(rna_embed)


class InteractionHead(nn.Module):
    """Interaction prediction head."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
    
    def forward(
        self,
        rna_fused: torch.Tensor,  # [B, L, d]
        mol_tokens: torch.Tensor,  # [B, K, d]
        rna_mask: torch.Tensor,   # [B, L]
    ) -> torch.Tensor:
        """
        Returns:
            logit: [B] interaction logit
        """
        # Mean pooling with mask
        rna_mask_expanded = rna_mask.unsqueeze(-1)
        rna_sum = (rna_fused * rna_mask_expanded).sum(dim=1)
        rna_count = rna_mask.sum(dim=1, keepdim=True).clamp(min=1)
        rna_pool = rna_sum / rna_count
        
        mol_pool = mol_tokens.mean(dim=1)
        
        # Concat and predict
        combined = torch.cat([rna_pool, mol_pool], dim=-1)
        logit = self.mlp(combined).squeeze(-1)
        
        return logit


class SiteHead(nn.Module):
    """Binding site prediction head (sequence labeling)."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )
    
    def forward(self, rna_fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rna_fused: [B, L, d]
        Returns:
            site_logits: [B, L]
        """
        return self.mlp(rna_fused).squeeze(-1)


class RNADTModel(nn.Module):
    """
    CoCoBind: RNA-Drug Multi-task Model
    
    Architecture:
    1. RNA Projector: Projects RNA-FM embeddings to d_model
    2. Mol Projector: Projects ECFP4 fingerprints to K tokens
    3. Cross-Attention: RNA queries molecule (can be disabled via use_cross_attn=False)
    4. Dual-head prediction:
       - InteractionHead: Predicts interaction probability
       - SiteHead: Predicts binding site probabilities
    
    Ablation mode (use_cross_attn=False):
       - Interaction: pooled_RNA + pooled_mol directly concatenated
       - Site: RNA projection only, no cross-attention fusion
    """
    
    def __init__(
        self,
        d_rna: int = 640,       # RNA-FM embedding dimension
        d_mol: int = 2048,      # ECFP4 fingerprint dimension
        d_model: int = 256,     # Hidden layer dimension
        n_mol_tokens: int = 4,  # Number of molecule tokens
        n_heads: int = 4,       # Attention heads
        dropout: float = 0.1,
        use_cross_attn: bool = True,  # Ablation switch
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_cross_attn = use_cross_attn
        
        # Projection layers
        self.rna_proj = RNAProjector(d_rna, d_model)
        self.mol_proj = MolTokenProjector(d_mol, d_model, n_mol_tokens)
        
        # Cross-Attention (only used when use_cross_attn=True)
        if use_cross_attn:
            self.cross_attn = CrossAttention(d_model, n_heads, dropout)
            self.norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
        else:
            self.cross_attn = None
            self.norm = None
            self.dropout = nn.Dropout(dropout)
        
        # Prediction heads
        self.interaction_head = InteractionHead(d_model, dropout)
        self.site_head = SiteHead(d_model, dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: dict with keys:
                - rna_embed: [B, L, d_rna]
                - mol_fp: [B, n_bits]
                - rna_mask: [B, L]
        
        Returns:
            dict with keys:
                - interaction_logit: [B]
                - site_logits: [B, L]
                - rna_pool: [B, d] (for contrastive learning)
                - mol_pool: [B, d] (for contrastive learning)
        """
        rna_embed = batch["rna_embed"]
        mol_fp = batch["mol_fp"]
        rna_mask = batch["rna_mask"]
        
        # Projection
        rna_tokens = self.rna_proj(rna_embed)
        mol_tokens = self.mol_proj(mol_fp)
        
        if self.use_cross_attn:
            # Cross-Attention: RNA queries molecule
            attn_out, _ = self.cross_attn(
                query=rna_tokens,
                key=mol_tokens,
                value=mol_tokens,
                query_mask=rna_mask,
                key_mask=None
            )
            # Residual + Norm
            rna_fused = self.norm(rna_tokens + self.dropout(attn_out))
        else:
            # Ablation mode: No Cross-Attention
            rna_fused = rna_tokens
        
        # Prediction
        interaction_logit = self.interaction_head(rna_fused, mol_tokens, rna_mask)
        site_logits = self.site_head(rna_fused)
        
        # Pooled representations (for contrastive learning)
        rna_mask_expanded = rna_mask.unsqueeze(-1)
        rna_sum = (rna_fused * rna_mask_expanded).sum(dim=1)
        rna_count = rna_mask.sum(dim=1, keepdim=True).clamp(min=1)
        rna_pool = rna_sum / rna_count
        mol_pool = mol_tokens.mean(dim=1)
        
        return {
            "interaction_logit": interaction_logit,
            "site_logits": site_logits,
            "rna_pool": rna_pool,
            "mol_pool": mol_pool,
        }
