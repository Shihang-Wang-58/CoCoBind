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
import sys
from pathlib import Path
from typing import Optional, Dict, List

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
        from rdkit import RDLogger
        
        # Suppress RDKit warnings (including MorganGenerator deprecation)
        RDLogger.DisableLog('rdApp.warning')
        
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


class OuroborosFeaturizer:
    """
    On-demand Ouroboros molecular feature extractor with disk caching.

    This avoids a separate precompute step for small prediction jobs. For
    training and large virtual-screening libraries, precomputed .pkl features
    are still faster and easier to audit.
    """

    def __init__(
        self,
        model_path: str,
        cache_dir: str = "cache/ouroboros_features",
        batch_size: int = 256,
        device: str = "cuda",
    ):
        self.model_path = str(model_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.device = device
        self.feature_dim = 2048
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "Ouroboros on-demand features require a CUDA-enabled PyTorch/DGL stack. "
                "Use precomputed features or run on a GPU server."
            )

        ouroboros_root = Path(self.model_path).resolve().parents[1]
        if str(ouroboros_root) not in sys.path:
            sys.path.insert(0, str(ouroboros_root))

        try:
            from ouroboros.model.GeminiMol import GeminiMol
        except ImportError as exc:
            raise ImportError(
                f"Cannot import Ouroboros from {ouroboros_root}. "
                "Install requirements-ouroboros.txt and check mol_encoder.model_path."
            ) from exc

        logger.info(f"Loading Ouroboros model from {self.model_path}")
        self._model = GeminiMol(
            model_path=self.model_path,
            batch_size=self.batch_size,
            cache=True,
        )
        self._model.eval()

    def _get_cache_path(self, smiles: str) -> Path:
        smi_hash = hashlib.md5(smiles.encode()).hexdigest()
        return self.cache_dir / f"{smi_hash}.npy"

    def encode_many(self, smiles_list: List[str]) -> torch.Tensor:
        features = [None] * len(smiles_list)
        missing_indices = []
        missing_smiles = []

        for i, smi in enumerate(smiles_list):
            cache_path = self._get_cache_path(smi)
            if cache_path.exists():
                features[i] = np.load(cache_path).astype(np.float32)
            else:
                missing_indices.append(i)
                missing_smiles.append(smi)

        if missing_smiles:
            self._load_model()
            with torch.no_grad():
                encoded = self._model.encode(missing_smiles).detach().cpu().numpy().astype(np.float32)
            for i, smi, feat in zip(missing_indices, missing_smiles, encoded):
                np.save(self._get_cache_path(smi), feat)
                features[i] = feat

        return torch.from_numpy(np.stack(features).astype(np.float32))

    def __call__(self, smiles: str) -> torch.Tensor:
        return self.encode_many([smiles])[0]


class HybridMolFeaturizer:
    """Try a primary featurizer first, then fall back to an on-demand featurizer."""

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        self.feature_dim = getattr(primary, "feature_dim", getattr(fallback, "feature_dim", None))

    def __call__(self, smiles: str) -> torch.Tensor:
        try:
            return self.primary(smiles)
        except KeyError:
            return self.fallback(smiles)

    def get_stats(self) -> dict:
        stats = {}
        if hasattr(self.primary, "get_stats"):
            stats.update(self.primary.get_stats())
        stats["fallback"] = type(self.fallback).__name__
        return stats


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
    mol_model_path: Optional[str] = None,
    mol_cache_dir: Optional[str] = None,
    device: str = "cuda",
    n_bits: int = 2048,
    fallback_to_ecfp: bool = False,
):
    """
    Get molecular feature extractor.
    
    Args:
        mol_encoder: Encoder type ("ecfp4", "kpgt", "ouroboros")
        mol_features_path: Optional precomputed features path for KPGT/Ouroboros
        mol_model_path: Optional model path for on-demand Ouroboros features
        n_bits: ECFP4 bits
        
    Returns:
        Molecular feature extractor instance
    """
    mol_encoder = mol_encoder.lower()
    
    if mol_encoder == "ecfp4":
        logger.info(f"Using ECFP4 featurizer (n_bits={n_bits})")
        return ECFP4Featurizer(n_bits=n_bits)
    
    elif mol_encoder == "ouroboros":
        if mol_features_path is not None and os.path.exists(mol_features_path):
            logger.info(f"Using precomputed OUROBOROS features from {mol_features_path}")
            primary = PrecomputedMolFeaturizer(mol_features_path, fallback_to_ecfp=fallback_to_ecfp)
            if mol_model_path is not None and os.path.exists(mol_model_path):
                cache_dir = mol_cache_dir or "cache/ouroboros_features"
                fallback = OuroborosFeaturizer(
                    model_path=mol_model_path,
                    cache_dir=cache_dir,
                    device=device,
                )
                return HybridMolFeaturizer(primary, fallback)
            return primary

        if mol_features_path is not None and not os.path.exists(mol_features_path):
            logger.warning(f"Precomputed Ouroboros features not found: {mol_features_path}")

        if mol_model_path is None:
            raise ValueError(
                "Either mol_features_path or mol_model_path is required for Ouroboros. "
                "Use a precomputed .pkl for speed, or provide mol_encoder.model_path "
                "for on-demand feature extraction."
            )
        if not os.path.exists(mol_model_path):
            raise FileNotFoundError(f"Ouroboros model_path not found: {mol_model_path}")
        cache_dir = mol_cache_dir or "cache/ouroboros_features"
        logger.info(f"Using on-demand Ouroboros features from {mol_model_path}; cache={cache_dir}")
        return OuroborosFeaturizer(
            model_path=mol_model_path,
            cache_dir=cache_dir,
            device=device,
        )

    elif mol_encoder == "kpgt":
        if mol_features_path is None:
            raise ValueError(
                "mol_features_path is required for KPGT. "
                "Please run precompute_mol_features.py first."
            )
        if not os.path.exists(mol_features_path):
            raise FileNotFoundError(
                f"Precomputed features not found: {mol_features_path}. "
                f"Please run: python -m cocobind.precompute_mol_features --model {mol_encoder} ..."
            )
        logger.info(f"Using precomputed {mol_encoder.upper()} features from {mol_features_path}")
        return PrecomputedMolFeaturizer(mol_features_path, fallback_to_ecfp=fallback_to_ecfp)
    
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
