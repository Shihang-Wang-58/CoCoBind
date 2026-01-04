"""
Feature Extractors for CoCoBind

Supported molecular encoders:
- ECFP4: Morgan fingerprints (default)
- KPGT: Pre-trained molecular representations
- Ouroboros: Pre-trained molecular representations

RNA encoder:
- RNA-FM: Pre-trained RNA foundation model (with disk caching)
"""
import os
import hashlib
import logging
import pickle
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ECFP4Featurizer:
    """Morgan fingerprint (ECFP4) molecular feature extractor."""
    
    def __init__(self, n_bits: int = 2048, radius: int = 2):
        self.n_bits = n_bits
        self.radius = radius
        self.feature_dim = n_bits
        
    def __call__(self, smiles: str) -> torch.Tensor:
        """
        Convert SMILES to ECFP4 fingerprint.
        
        Args:
            smiles: Molecule SMILES string
            
        Returns:
            torch.Tensor: [n_bits] float32 fingerprint vector
        """
        from rdkit import Chem
        from rdkit.Chem import AllChem
        
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}, returning zero vector")
            return torch.zeros(self.n_bits, dtype=torch.float32)
        
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        arr = np.zeros(self.n_bits, dtype=np.float32)
        Chem.DataStructs.ConvertToNumpyArray(fp, arr)
        return torch.from_numpy(arr)


class PrecomputedMolFeaturizer:
    """
    Precomputed molecular feature loader.
    
    Loads molecular features from pre-computed pickle file (KPGT or Ouroboros).
    Falls back to ECFP4 if SMILES not in precomputed dictionary.
    """
    
    def __init__(
        self,
        features_path: str,
        fallback_to_ecfp: bool = True,
        ecfp_bits: int = 2048
    ):
        """
        Args:
            features_path: Path to precomputed features (.pkl)
            fallback_to_ecfp: Whether to fall back to ECFP4 if SMILES not found
            ecfp_bits: ECFP4 fingerprint bits (for dimension matching in fallback)
        """
        self.features_path = features_path
        self.fallback_to_ecfp = fallback_to_ecfp
        
        logger.info(f"Loading precomputed mol features from {features_path}")
        with open(features_path, 'rb') as f:
            self._features_dict: Dict[str, np.ndarray] = pickle.load(f)
        
        # Get feature dimension
        sample = next(iter(self._features_dict.values()))
        self.feature_dim = sample.shape[0]
        logger.info(f"Loaded {len(self._features_dict)} precomputed features (dim={self.feature_dim})")
        
        # ECFP4 fallback
        if fallback_to_ecfp:
            self._ecfp = ECFP4Featurizer(n_bits=ecfp_bits)
        
        self.hits = 0
        self.misses = 0
        
    def __call__(self, smiles: str) -> torch.Tensor:
        if smiles in self._features_dict:
            self.hits += 1
            return torch.from_numpy(self._features_dict[smiles].astype(np.float32))
        
        self.misses += 1
        if self.fallback_to_ecfp:
            logger.debug(f"SMILES not in precomputed: {smiles[:50]}...")
            ecfp_feat = self._ecfp(smiles)
            if ecfp_feat.shape[0] == self.feature_dim:
                return ecfp_feat
            elif ecfp_feat.shape[0] < self.feature_dim:
                padded = torch.zeros(self.feature_dim, dtype=torch.float32)
                padded[:ecfp_feat.shape[0]] = ecfp_feat
                return padded
            else:
                return ecfp_feat[:self.feature_dim]
        else:
            raise KeyError(f"SMILES not in precomputed features: {smiles}")
    
    def get_stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "precomputed_hits": self.hits,
            "precomputed_misses": self.misses,
            "hit_rate": f"{self.hits/total:.2%}" if total > 0 else "N/A"
        }


class RNAFMFeaturizer:
    """
    RNA-FM embedding feature extractor (using transformers/huggingface).
    Supports disk caching to avoid redundant computation.
    
    Note: RNA-FM always runs on CPU to avoid CUDA fork issues in DataLoader multiprocessing.
    Embeddings are cached to disk for subsequent access.
    """
    
    def __init__(
        self,
        cache_dir: str = "cache/rna_embeddings",
        device: str = "cpu",
        model_name: str = "multimolecule/rnafm"
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cpu"  # Always use CPU for DataLoader compatibility
        self.model_name = model_name
        
        self._model = None
        self._tokenizer = None
        
        self.cache_hits = 0
        self.cache_misses = 0
        
    def _load_model(self):
        """Lazy model loading."""
        if self._model is None:
            from transformers import AutoTokenizer, AutoModel
            
            try:
                import multimolecule  # noqa: F401
                logger.info("multimolecule package loaded")
            except ImportError:
                logger.warning(
                    "multimolecule package not found. "
                    "Install with: pip install multimolecule"
                )
            
            logger.info(f"Loading RNA-FM model: {self.model_name}")
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, trust_remote_code=True
                )
                self._model = AutoModel.from_pretrained(
                    self.model_name, trust_remote_code=True
                )
            except ValueError as e:
                if "RnaTokenizer" in str(e):
                    raise RuntimeError(
                        f"Failed to load RNA-FM tokenizer. "
                        f"Please install multimolecule: pip install multimolecule\n"
                        f"Original error: {e}"
                    ) from e
                raise
            
            self._model.to(self.device)
            self._model.eval()
            logger.info("RNA-FM model loaded successfully")
            
    def _get_cache_path(self, sequence: str) -> Path:
        """Compute cache file path from sequence."""
        seq_hash = hashlib.md5(sequence.encode()).hexdigest()
        return self.cache_dir / f"{seq_hash}.npy"
        
    def __call__(self, sequence: str, max_len: int = 512) -> torch.Tensor:
        """
        Extract RNA sequence token embeddings.
        
        Args:
            sequence: RNA sequence (auto T->U, uppercase)
            max_len: Maximum sequence length
            
        Returns:
            torch.Tensor: [L, d_model] token embedding
        """
        sequence = sequence.upper().replace('T', 'U')
        
        if len(sequence) > max_len:
            sequence = sequence[:max_len]
        
        cache_path = self._get_cache_path(sequence)
        if cache_path.exists():
            self.cache_hits += 1
            embedding = np.load(cache_path)
            return torch.from_numpy(embedding)
        
        self.cache_misses += 1
        
        self._load_model()
        
        with torch.no_grad():
            inputs = self._tokenizer(
                sequence,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=max_len + 2
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            
            hidden = outputs.last_hidden_state[0]
            embedding = hidden[1:-1].cpu().numpy()
        
        np.save(cache_path, embedding)
        
        return torch.from_numpy(embedding)
    
    def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": f"{hit_rate:.2%}"
        }


def get_rna_embedding_dim(model_name: str = "multimolecule/rnafm") -> int:
    """Get RNA-FM model embedding dimension."""
    model_dims = {
        "multimolecule/rnafm": 640,
        "multimolecule/rnabert": 120,
    }
    return model_dims.get(model_name, 640)


def get_mol_featurizer(
    mol_encoder: str = "ecfp4",
    mol_features_path: Optional[str] = None,
    n_bits: int = 2048,
):
    """
    Get molecular feature extractor.
    
    Args:
        mol_encoder: Encoder type ("ecfp4", "kpgt", "ouroboros")
        mol_features_path: Precomputed features path (required for KPGT/Ouroboros)
        n_bits: ECFP4 bits
        
    Returns:
        Molecular feature extractor instance
    """
    mol_encoder = mol_encoder.lower()
    
    if mol_encoder == "ecfp4":
        logger.info(f"Using ECFP4 featurizer (n_bits={n_bits})")
        return ECFP4Featurizer(n_bits=n_bits)
    
    elif mol_encoder in ["kpgt", "ouroboros"]:
        if mol_features_path is None:
            raise ValueError(
                f"mol_features_path is required for {mol_encoder}. "
                f"Please run precompute_mol_features.py first."
            )
        if not os.path.exists(mol_features_path):
            raise FileNotFoundError(
                f"Precomputed features not found: {mol_features_path}. "
                f"Please run: python -m cocobind.precompute_mol_features --model {mol_encoder} ..."
            )
        logger.info(f"Using precomputed {mol_encoder.upper()} features from {mol_features_path}")
        return PrecomputedMolFeaturizer(mol_features_path, fallback_to_ecfp=True)
    
    else:
        raise ValueError(f"Unknown mol_encoder: {mol_encoder}. Use 'ecfp4', 'kpgt', or 'ouroboros'")


def get_mol_feature_dim(mol_encoder: str = "ecfp4", mol_features_path: Optional[str] = None) -> int:
    """
    Get molecular feature dimension.
    
    Args:
        mol_encoder: Encoder type
        mol_features_path: Precomputed features path
        
    Returns:
        Feature dimension
    """
    mol_encoder = mol_encoder.lower()
    
    if mol_encoder == "ecfp4":
        return 2048
    elif mol_encoder == "kpgt":
        return 2304
    elif mol_encoder == "ouroboros":
        return 2048
    else:
        raise ValueError(f"Unknown mol_encoder: {mol_encoder}")
