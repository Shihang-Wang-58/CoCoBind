# CoCoBind: RNA-Drug Interaction & Binding Site Prediction
# Multi-task model with consistency constraint
"""
CoCoBind - Cooperative Consistency-constrained Binding Prediction

A multi-task deep learning framework for RNA-small molecule interaction 
prediction and binding site identification with novel consistency constraints.
"""
__version__ = "1.0.0"
__author__ = "CoCoBind Team"

from .model import RNADTModel
from .featurizers import ECFP4Featurizer, RNAFMFeaturizer, PrecomputedMolFeaturizer

__all__ = [
    "RNADTModel",
    "ECFP4Featurizer",
    "RNAFMFeaturizer",
    "PrecomputedMolFeaturizer",
]
