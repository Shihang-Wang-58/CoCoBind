"""
CoCoBind Data Loading and Preprocessing

Supports DeepRNA-DTI Dataset format with:
- ecfp4: Morgan fingerprints (2048 bits)
- kpgt: KPGT precomputed features (2304 dims)
- ouroboros: Ouroboros precomputed features (2048 dims)
"""
import os
import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .featurizers import (
    ECFP4Featurizer, 
    RNAFMFeaturizer, 
    PrecomputedMolFeaturizer,
    get_mol_featurizer,
    get_mol_feature_dim,
)

logger = logging.getLogger(__name__)


def parse_binding_sites(value: Union[str, list, float]) -> Optional[List[float]]:
    """
    Parse binding_site_index column, supporting multiple formats:
    - "[0.0, 1.0, 0.0, ...]" (JSON/Python list string)
    - "1,2,3" (comma-separated indices)
    - Empty/NaN
    
    Returns: 0/1 list (same length as sequence) or None
    """
    if pd.isna(value) or value == "" or value == "[]":
        return None
    
    if isinstance(value, list):
        return [float(x) for x in value]
    
    value = str(value).strip()
    
    if value.startswith("["):
        try:
            parsed = ast.literal_eval(value)
            return [float(x) for x in parsed]
        except:
            pass
    
    if "," in value and not value.startswith("["):
        try:
            indices = [int(x.strip()) for x in value.split(",")]
            return {"type": "indices", "values": indices}
        except:
            pass
    
    return None


def indices_to_mask(indices: List[int], seq_len: int) -> List[float]:
    """Convert index list to 0/1 mask."""
    mask = [0.0] * seq_len
    for idx in indices:
        if 0 <= idx < seq_len:
            mask[idx] = 1.0
    return mask


class CombinedRNADTDataset(Dataset):
    """
    Combined dataset:
    - Loads interaction labels from dti_data (includes positive and negative samples)
    - Loads real binding site labels from bs_data (only positive samples have detailed annotations)
    """
    
    def __init__(
        self,
        dti_csv_path: str,
        bs_csv_path: Optional[str],
        mol_featurizer,
        rna_featurizer: RNAFMFeaturizer,
        max_len: int = 512,
    ):
        self.mol_featurizer = mol_featurizer
        self.rna_featurizer = rna_featurizer
        self.max_len = max_len
        
        logger.info(f"Loading DTI data from {dti_csv_path}")
        self.dti_df = pd.read_csv(dti_csv_path)
        
        self.bs_lookup = {}
        if bs_csv_path and os.path.exists(bs_csv_path):
            logger.info(f"Loading BS data from {bs_csv_path}")
            bs_df = pd.read_csv(bs_csv_path)
            for _, row in bs_df.iterrows():
                seq = str(row["sequence"]).upper().replace('T', 'U')[:max_len]
                smi = str(row["smiles"])
                key = (seq, smi)
                sites = self._parse_binding_sites(row.get("binding_site_index", None), len(seq))
                if sites is not None and sum(sites) > 0:
                    self.bs_lookup[key] = sites
            logger.info(f"BS lookup built: {len(self.bs_lookup)} samples with valid site labels")
        
        self.dti_df["sequence"] = self.dti_df["sequence"].apply(
            lambda x: str(x).upper().replace('T', 'U')[:max_len]
        )
        
        n_pos = (self.dti_df["interactions"] > 0.5).sum()
        n_neg = (self.dti_df["interactions"] <= 0.5).sum()
        logger.info(f"Dataset: {len(self.dti_df)} samples (pos: {n_pos}, neg: {n_neg})")
        
        self._cached_rna = {}
        self._cached_mol = {}
    
    def _parse_binding_sites(self, value, seq_len: int) -> Optional[List[float]]:
        if pd.isna(value) or value == "" or value == "[]":
            return None
        
        if isinstance(value, list):
            return [float(x) for x in value[:seq_len]]
        
        value = str(value).strip()
        
        if value.startswith("["):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list) and len(parsed) > 1:
                    return [float(x) for x in parsed[:seq_len]]
            except:
                pass
        
        return None
    
    def __len__(self) -> int:
        return len(self.dti_df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.dti_df.iloc[idx]
        
        seq = row["sequence"]
        smi = row["smiles"]
        interaction = float(row["interactions"])
        
        if seq in self._cached_rna:
            rna_embed = self._cached_rna[seq]
        else:
            rna_embed = self.rna_featurizer(seq, self.max_len)
            if len(self._cached_rna) < 10000:
                self._cached_rna[seq] = rna_embed
        
        if smi in self._cached_mol:
            mol_fp = self._cached_mol[smi]
        else:
            mol_fp = self.mol_featurizer(smi)
            if len(self._cached_mol) < 50000:
                self._cached_mol[smi] = mol_fp
        
        seq_len = rna_embed.shape[0]
        
        result = {
            "rna_embed": rna_embed.float(),
            "mol_fp": mol_fp.float(),
            "seq_len": torch.tensor(seq_len),
            "interaction": torch.tensor(interaction),
        }
        
        key = (seq, smi)
        if key in self.bs_lookup and interaction > 0.5:
            sites = self.bs_lookup[key]
            if len(sites) >= seq_len:
                site_labels = torch.tensor(sites[:seq_len], dtype=torch.float32)
            else:
                site_labels = torch.zeros(seq_len, dtype=torch.float32)
                site_labels[:len(sites)] = torch.tensor(sites, dtype=torch.float32)
            result["site_labels"] = site_labels
            result["has_site_labels"] = torch.tensor(1.0)
        else:
            result["site_labels"] = torch.zeros(seq_len, dtype=torch.float32)
            result["has_site_labels"] = torch.tensor(0.0)
        
        return result


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for variable-length sequences."""
    max_len = max(item["rna_embed"].shape[0] for item in batch)
    batch_size = len(batch)
    d_rna = batch[0]["rna_embed"].shape[1]
    
    rna_embed = torch.zeros(batch_size, max_len, d_rna)
    rna_mask = torch.zeros(batch_size, max_len)
    site_labels = torch.zeros(batch_size, max_len)
    mol_fp = torch.stack([item["mol_fp"] for item in batch])
    seq_lens = torch.stack([item["seq_len"] for item in batch])
    interactions = torch.stack([item["interaction"] for item in batch])
    has_site_labels = torch.stack([item["has_site_labels"] for item in batch])
    
    for i, item in enumerate(batch):
        L = item["rna_embed"].shape[0]
        rna_embed[i, :L] = item["rna_embed"]
        rna_mask[i, :L] = 1.0
        site_labels[i, :L] = item["site_labels"]
    
    return {
        "rna_embed": rna_embed,
        "rna_mask": rna_mask,
        "mol_fp": mol_fp,
        "seq_lens": seq_lens,
        "interactions": interactions,
        "site_labels": site_labels,
        "has_site_labels": has_site_labels,
    }


def get_data_paths(data_root: str, split: str, fold: int, use_bs_data: bool = True) -> Dict[str, Dict[str, str]]:
    """Get data file paths."""
    dti_base = Path(data_root) / split / "dti_data"
    bs_base = Path(data_root) / split / "bs_data"
    
    result = {
        "train": {
            "dti": str(dti_base / f"train_fold{fold}" / "raw" / "interactions.csv"),
        },
        "val": {
            "dti": str(dti_base / f"val_fold{fold}" / "raw" / "interactions.csv"),
        },
        "test": {
            "dti": str(dti_base / "test_fold" / "raw" / "interactions.csv"),
        },
    }
    
    if use_bs_data:
        result["train"]["bs"] = str(bs_base / f"train_fold{fold}" / "raw" / "interactions.csv")
        result["val"]["bs"] = str(bs_base / f"val_fold{fold}" / "raw" / "interactions.csv")
        result["test"]["bs"] = str(bs_base / "test_fold" / "raw" / "interactions.csv")
    
    return result


def create_dataloaders(
    data_root: str,
    split: str,
    fold: int,
    batch_size: int,
    max_len: int = 512,
    cache_dir: str = "cache",
    device: str = "cpu",
    num_workers: int = 0,
    use_bs_data: bool = True,
    mol_encoder: str = "ecfp4",
    mol_features_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, RNAFMFeaturizer]:
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_root: Dataset root directory
        split: Data split type (unseen_both, unseen_compound, etc.)
        fold: Cross-validation fold number
        batch_size: Batch size
        max_len: Maximum sequence length
        cache_dir: Cache directory
        device: Compute device
        num_workers: DataLoader worker processes
        use_bs_data: Whether to use bs_data for real site annotations
        mol_encoder: Molecular encoder type ("ecfp4", "kpgt", "ouroboros")
        mol_features_path: Precomputed molecular features path
    """
    paths = get_data_paths(data_root, split, fold, use_bs_data=use_bs_data)
    
    for name, path_dict in paths.items():
        dti_path = path_dict["dti"]
        if not os.path.exists(dti_path):
            raise FileNotFoundError(f"{name} DTI data not found: {dti_path}")
    
    mol_featurizer = get_mol_featurizer(
        mol_encoder=mol_encoder,
        mol_features_path=mol_features_path,
        n_bits=2048,
    )
    
    rna_featurizer = RNAFMFeaturizer(
        cache_dir=os.path.join(cache_dir, "rna_embeddings"),
        device=device
    )
    
    train_ds = CombinedRNADTDataset(
        paths["train"]["dti"],
        paths["train"].get("bs"),
        mol_featurizer, rna_featurizer, max_len
    )
    val_ds = CombinedRNADTDataset(
        paths["val"]["dti"],
        paths["val"].get("bs"),
        mol_featurizer, rna_featurizer, max_len
    )
    test_ds = CombinedRNADTDataset(
        paths["test"]["dti"],
        paths["test"].get("bs"),
        mol_featurizer, rna_featurizer, max_len
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, rna_featurizer
