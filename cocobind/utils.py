"""
Utility functions for CoCoBind
"""
import os
import random
import logging
from pathlib import Path
from typing import Dict, Any

import yaml
import numpy as np
import torch


def set_seed(seed: int):
    """Fix random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file: str = None, level: int = logging.INFO):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], path: str):
    """Save configuration to YAML."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def save_json(data: dict, path: str):
    """Save JSON file."""
    import json
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str) -> dict:
    """Load JSON file."""
    import json
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_device(device_str: str) -> torch.device:
    """Get PyTorch device."""
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif device_str.startswith("cuda:") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable model parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
