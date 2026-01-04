"""
CoCoBind Evaluation Metrics

Comprehensive metrics for:
- Interaction: AUROC, AUPR, F1, ACC, Precision, Recall, Specificity, MCC, Balanced ACC
- Binding-site: Micro/Macro AUROC/AUPR, TopK Recall/Precision, IoU, Dice
"""
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    confusion_matrix,
)


def compute_interaction_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute interaction prediction metrics (binary classification).
    
    Args:
        y_true: Ground truth labels [N]
        y_prob: Predicted probabilities [N]
        threshold: Classification threshold
    
    Returns:
        dict with comprehensive metrics
    """
    y_pred = (y_prob >= threshold).astype(int)
    y_true = y_true.astype(int)
    
    n_classes = len(np.unique(y_true))
    if n_classes < 2:
        auroc = 0.5
        aupr = float(np.mean(y_true))
    else:
        auroc = roc_auc_score(y_true, y_prob)
        aupr = average_precision_score(y_true, y_prob)
    
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred) if n_classes >= 2 else 0.0
    bal_acc = balanced_accuracy_score(y_true, y_pred) if n_classes >= 2 else 0.5
    
    if n_classes >= 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        specificity = 0.0
        tn, fp, fn, tp = 0, 0, 0, 0
    
    return {
        "int_auroc": float(auroc),
        "int_aupr": float(aupr),
        "int_f1": float(f1),
        "int_acc": float(acc),
        "int_precision": float(precision),
        "int_recall": float(recall),
        "int_specificity": float(specificity),
        "int_mcc": float(mcc),
        "int_balanced_acc": float(bal_acc),
        "int_tp": int(tp),
        "int_tn": int(tn),
        "int_fp": int(fp),
        "int_fn": int(fn),
    }


def compute_site_metrics(
    site_labels_list: List[np.ndarray],
    site_probs_list: List[np.ndarray],
    interactions: np.ndarray,
    topk_values: List[int] = [5, 10, 20],
    topk_ratio: float = 0.05,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute binding site prediction metrics (only for positive interaction samples).
    
    Args:
        site_labels_list: Site labels per sample [N x [L_i]]
        site_probs_list: Site prediction probabilities per sample [N x [L_i]]
        interactions: Interaction labels [N]
        topk_values: K values for TopK Recall/Precision
        topk_ratio: Ratio for TopK Recall
        threshold: Binarization threshold
    
    Returns:
        dict with comprehensive site metrics
    """
    all_labels = []
    all_probs = []
    
    per_sample_auroc = []
    per_sample_aupr = []
    per_sample_f1 = []
    per_sample_iou = []
    per_sample_dice = []
    
    topk_recalls = {k: [] for k in topk_values}
    topk_precisions = {k: [] for k in topk_values}
    topk_ratio_recalls = []
    topk_ratio_precisions = []
    
    n_samples_with_sites = 0
    
    for i, (labels, probs, y) in enumerate(zip(
        site_labels_list, site_probs_list, interactions
    )):
        if y < 0.5:
            continue
        
        labels = np.array(labels)
        probs = np.array(probs)
        
        seq_len = len(labels)
        if seq_len == 0:
            continue
        
        n_positives = int(labels.sum())
        if n_positives == 0:
            continue
        
        n_samples_with_sites += 1
        
        all_labels.extend(labels.tolist())
        all_probs.extend(probs.tolist())
        
        if len(np.unique(labels)) >= 2:
            per_sample_auroc.append(roc_auc_score(labels, probs))
            per_sample_aupr.append(average_precision_score(labels, probs))
        
        preds = (probs >= threshold).astype(int)
        per_sample_f1.append(f1_score(labels, preds, zero_division=0))
        
        tp = np.sum((labels == 1) & (preds == 1))
        fp = np.sum((labels == 0) & (preds == 1))
        fn = np.sum((labels == 1) & (preds == 0))
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        per_sample_iou.append(iou)
        
        dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        per_sample_dice.append(dice)
        
        sorted_indices = np.argsort(probs)[::-1]
        
        for k in topk_values:
            top_k = min(k, seq_len)
            top_k_indices = sorted_indices[:top_k]
            tp_at_k = labels[top_k_indices].sum()
            
            recall_at_k = tp_at_k / n_positives
            precision_at_k = tp_at_k / top_k
            
            topk_recalls[k].append(recall_at_k)
            topk_precisions[k].append(precision_at_k)
        
        k_ratio = max(1, int(math.ceil(topk_ratio * seq_len)))
        top_k_indices = sorted_indices[:k_ratio]
        tp_at_k = labels[top_k_indices].sum()
        
        topk_ratio_recalls.append(tp_at_k / n_positives)
        topk_ratio_precisions.append(tp_at_k / k_ratio)
    
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    if len(all_labels) == 0 or len(np.unique(all_labels)) < 2:
        micro_auroc = 0.5
        micro_aupr = 0.0
        micro_f1 = 0.0
        micro_mcc = 0.0
    else:
        micro_auroc = roc_auc_score(all_labels, all_probs)
        micro_aupr = average_precision_score(all_labels, all_probs)
        all_preds = (all_probs >= threshold).astype(int)
        micro_f1 = f1_score(all_labels, all_preds, zero_division=0)
        micro_mcc = matthews_corrcoef(all_labels, all_preds)
    
    result = {
        "site_micro_auroc": float(micro_auroc),
        "site_micro_aupr": float(micro_aupr),
        "site_micro_f1": float(micro_f1),
        "site_micro_mcc": float(micro_mcc),
        "site_n_samples": n_samples_with_sites,
    }
    
    if len(per_sample_auroc) > 0:
        result["site_macro_auroc"] = float(np.mean(per_sample_auroc))
        result["site_macro_aupr"] = float(np.mean(per_sample_aupr))
    else:
        result["site_macro_auroc"] = 0.5
        result["site_macro_aupr"] = 0.0
    
    if len(per_sample_f1) > 0:
        result["site_macro_f1"] = float(np.mean(per_sample_f1))
        result["site_macro_iou"] = float(np.mean(per_sample_iou))
        result["site_macro_dice"] = float(np.mean(per_sample_dice))
    else:
        result["site_macro_f1"] = 0.0
        result["site_macro_iou"] = 0.0
        result["site_macro_dice"] = 0.0
    
    for k in topk_values:
        if len(topk_recalls[k]) > 0:
            result[f"site_topk{k}_recall"] = float(np.mean(topk_recalls[k]))
            result[f"site_topk{k}_precision"] = float(np.mean(topk_precisions[k]))
        else:
            result[f"site_topk{k}_recall"] = 0.0
            result[f"site_topk{k}_precision"] = 0.0
    
    if len(topk_ratio_recalls) > 0:
        result["site_topk_ratio_recall"] = float(np.mean(topk_ratio_recalls))
        result["site_topk_ratio_precision"] = float(np.mean(topk_ratio_precisions))
    else:
        result["site_topk_ratio_recall"] = 0.0
        result["site_topk_ratio_precision"] = 0.0
    
    return result


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    site_labels_list: List[np.ndarray],
    site_probs_list: List[np.ndarray],
    topk_values: List[int] = [5, 10, 20],
    topk_ratio: float = 0.05,
) -> Dict[str, float]:
    """Compute all metrics."""
    int_metrics = compute_interaction_metrics(y_true, y_prob)
    site_metrics = compute_site_metrics(
        site_labels_list, site_probs_list, y_true,
        topk_values, topk_ratio
    )
    return {**int_metrics, **site_metrics}


def format_metrics_table(metrics: Dict[str, float], prefix: str = "") -> str:
    """Format metrics as readable table."""
    lines = []
    
    lines.append(f"{prefix}=== Interaction Metrics ===")
    lines.append(f"{prefix}  AUROC: {metrics.get('int_auroc', 0):.4f}")
    lines.append(f"{prefix}  AUPR:  {metrics.get('int_aupr', 0):.4f}")
    lines.append(f"{prefix}  F1:    {metrics.get('int_f1', 0):.4f}")
    lines.append(f"{prefix}  MCC:   {metrics.get('int_mcc', 0):.4f}")
    lines.append(f"{prefix}  Precision: {metrics.get('int_precision', 0):.4f}")
    lines.append(f"{prefix}  Recall:    {metrics.get('int_recall', 0):.4f}")
    
    lines.append(f"{prefix}=== Binding Site Metrics ===")
    lines.append(f"{prefix}  Micro-AUROC: {metrics.get('site_micro_auroc', 0):.4f}")
    lines.append(f"{prefix}  Micro-AUPR:  {metrics.get('site_micro_aupr', 0):.4f}")
    lines.append(f"{prefix}  Micro-F1:    {metrics.get('site_micro_f1', 0):.4f}")
    lines.append(f"{prefix}  Macro-IoU:   {metrics.get('site_macro_iou', 0):.4f}")
    lines.append(f"{prefix}  Macro-Dice:  {metrics.get('site_macro_dice', 0):.4f}")
    
    for k in [5, 10, 20]:
        if f"site_topk{k}_recall" in metrics:
            lines.append(f"{prefix}  Top{k}-Recall: {metrics.get(f'site_topk{k}_recall', 0):.4f}")
    
    return "\n".join(lines)
