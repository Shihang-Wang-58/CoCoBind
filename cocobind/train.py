"""
CoCoBind Training Script
Usage: python -m cocobind.train --split unseen_pair --fold 0
"""
import os
import sys
import argparse
import logging
import copy
from pathlib import Path
from typing import Dict, Optional

import yaml
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .data import create_dataloaders
from .model import RNADTModel
from .losses import MultiTaskLoss
from .metrics import compute_all_metrics, format_metrics_table
from .utils import (
    set_seed, setup_logging, load_config, save_json, get_device, count_parameters
)
from .featurizers import get_rna_embedding_dim, get_mol_feature_dim
from .training_utils import get_optimizer, create_scheduler, EMA

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="CoCoBind Training")
    parser.add_argument("--split", type=str, required=True,
                        choices=["unseen_pair", "unseen_rna", "unseen_compound", "unseen_both"],
                        help="Data split type")
    parser.add_argument("--fold", type=int, required=True,
                        choices=[0, 1, 2, 3, 4], help="Fold number")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name (default: from config)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override data root directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs from config")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch_size from config")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate from config")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU device ID")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device from config")
    # Ablation switches
    parser.add_argument("--use_cross_attn", type=lambda x: x.lower() == 'true', default=None,
                        help="Override model.use_cross_attn (true/false)")
    parser.add_argument("--lambda_cons", type=float, default=None,
                        help="Override loss.lambda_cons (0 to disable consistency)")
    parser.add_argument("--use_contrastive", type=lambda x: x.lower() == 'true', default=None,
                        help="Override contrastive.enabled (true/false)")
    # Molecular encoder
    parser.add_argument("--mol_encoder", type=str, default=None,
                        choices=["ecfp4", "kpgt", "ouroboros"],
                        help="Molecular encoder type")
    parser.add_argument("--mol_features_path", type=str, default=None,
                        help="Path to precomputed molecular features")
    return parser.parse_args()


def train_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train one epoch."""
    model.train()
    
    total_loss = 0.0
    loss_components = {"loss_int": 0.0, "loss_site": 0.0, "loss_cons": 0.0, "loss_ctr": 0.0}
    n_batches = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        optimizer.zero_grad()
        outputs = model(batch)
        losses = criterion(outputs, batch)
        losses["loss"].backward()
        
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        
        total_loss += losses["loss"].item()
        for k in loss_components:
            if k in losses:
                val = losses[k]
                loss_components[k] += val.item() if hasattr(val, 'item') else float(val)
        n_batches += 1
        
        pbar.set_postfix(loss=f"{losses['loss'].item():.4f}")
    
    avg_loss = total_loss / max(n_batches, 1)
    avg_components = {k: v / max(n_batches, 1) for k, v in loss_components.items()}
    
    return {"loss": avg_loss, **avg_components}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
    topk_values: list = [10],
    topk_ratio: float = 0.05,
) -> Dict:
    """Evaluate model."""
    model.eval()
    
    all_int_logits = []
    all_int_labels = []
    all_site_probs = []
    all_site_labels = []
    
    total_loss = 0.0
    n_batches = 0
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                 for k, v in batch.items()}
        
        outputs = model(batch)
        losses = criterion(outputs, batch)
        
        total_loss += losses["loss"].item()
        n_batches += 1
        
        int_logits = outputs["interaction_logit"].cpu().numpy()
        site_logits = outputs["site_logits"].cpu().numpy()
        
        all_int_logits.extend(int_logits.tolist())
        all_int_labels.extend(batch["interactions"].cpu().numpy().tolist())
        
        seq_lens = batch["seq_lens"].cpu().numpy()
        site_labels = batch["site_labels"].cpu().numpy()
        
        for i in range(len(seq_lens)):
            L = int(seq_lens[i])
            all_site_probs.append(
                (1 / (1 + np.exp(-site_logits[i, :L]))).tolist()
            )
            all_site_labels.append(site_labels[i, :L].tolist())
    
    y_true = np.array(all_int_labels)
    y_prob = 1 / (1 + np.exp(-np.array(all_int_logits)))
    
    metrics = compute_all_metrics(
        y_true, y_prob,
        all_site_labels, all_site_probs,
        topk_values, topk_ratio
    )
    metrics["loss"] = total_loss / max(n_batches, 1)
    
    preds = {
        "y_true": y_true.tolist(),
        "y_prob": y_prob.tolist(),
        "site_labels": all_site_labels,
        "site_probs": all_site_probs,
    }
    
    return metrics, preds


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epochs: int,
    patience: int,
    grad_clip: float,
    output_dir: str,
    topk_values: list,
    topk_ratio: float,
    ema=None,
    save_every: int = 50,
    best_metric: str = "int_aupr",
):
    """Full training loop."""
    best_val_score = 0.0
    patience_counter = 0
    history = []
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch}/{epochs} (lr={current_lr:.2e})")
        
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, grad_clip
        )
        
        if ema is not None:
            ema.update()
        
        logger.info(
            f"  Train - Loss: {train_metrics['loss']:.4f} | "
            f"L_int: {train_metrics['loss_int']:.4f} | "
            f"L_site: {train_metrics['loss_site']:.4f} | "
            f"L_cons: {train_metrics['loss_cons']:.4f}"
        )
        
        if ema is not None:
            ema.apply_shadow()
        
        val_metrics, _ = evaluate(
            model, val_loader, criterion, device, topk_values, topk_ratio
        )
        
        if ema is not None:
            ema.restore()
        
        logger.info(
            f"  Val   - Loss: {val_metrics['loss']:.4f} | "
            f"Int-AUROC: {val_metrics['int_auroc']:.4f}, Int-AUPR: {val_metrics['int_aupr']:.4f} | "
            f"Site-AUROC: {val_metrics['site_micro_auroc']:.4f}"
        )
        
        if scheduler:
            scheduler.step()
        
        history.append({
            "epoch": epoch,
            "lr": current_lr,
            "train": train_metrics,
            "val": val_metrics,
        })
        
        current_score = val_metrics.get(best_metric, val_metrics.get("int_aupr", 0))
        
        if current_score > best_val_score:
            best_val_score = current_score
            patience_counter = 0
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
            }, output_path / "best.pt")
            logger.info(f"  ★ New best model saved ({best_metric}: {best_val_score:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        if save_every > 0 and epoch % save_every == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, output_path / f"checkpoint_epoch{epoch}.pt")
    
    save_json(history, str(output_path / "history.json"))
    
    return best_val_score


def main():
    args = parse_args()
    
    # Load config
    if args.config:
        config_path = args.config
    else:
        config_path = Path(__file__).parent / "config.yaml"
    
    config = load_config(str(config_path))
    
    # Override config with args
    if args.epochs:
        config["train"]["epochs"] = args.epochs
    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size
    if args.lr:
        config["train"]["lr"] = args.lr
    if args.gpu is not None:
        config["device"] = f"cuda:{args.gpu}"
    elif args.device:
        config["device"] = args.device
    if args.data_root:
        config["data"]["root"] = args.data_root
    if args.exp_name:
        config["exp_name"] = args.exp_name
    
    # Ablation switches
    if args.use_cross_attn is not None:
        config["model"]["use_cross_attn"] = args.use_cross_attn
    if args.lambda_cons is not None:
        config["loss"]["lambda_cons"] = args.lambda_cons
    if args.use_contrastive is not None:
        config["contrastive"]["enabled"] = args.use_contrastive
    
    exp_name = config.get("exp_name", "base")
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            config["output"]["dir"], exp_name, args.split, f"fold{args.fold}"
        )
    
    setup_logging(os.path.join(output_dir, "train.log"))
    logger.info(f"Training: exp_name={exp_name}, split={args.split}, fold={args.fold}")
    
    set_seed(config["seed"])
    
    device = get_device(config["device"])
    logger.info(f"Using device: {device}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "config_resolved.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # Data
    mol_encoder_cfg = config.get("mol_encoder", {})
    mol_encoder_type = args.mol_encoder if args.mol_encoder else mol_encoder_cfg.get("type", "ecfp4")
    mol_features_path = args.mol_features_path if args.mol_features_path else mol_encoder_cfg.get("features_path", None)
    
    train_loader, val_loader, test_loader, rna_featurizer = create_dataloaders(
        data_root=config["data"]["root"],
        split=args.split,
        fold=args.fold,
        batch_size=config["train"]["batch_size"],
        max_len=config["data"]["max_len"],
        cache_dir=config["data"]["cache_dir"],
        device=str(device),
        num_workers=config.get("num_workers", 0),
        mol_encoder=mol_encoder_type,
        mol_features_path=mol_features_path,
    )
    logger.info(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Model
    d_rna = get_rna_embedding_dim()
    d_mol = get_mol_feature_dim(mol_encoder_type, mol_features_path)
    
    model = RNADTModel(
        d_rna=d_rna,
        d_mol=d_mol,
        d_model=config["model"]["d_model"],
        n_mol_tokens=config["model"]["n_mol_tokens"],
        n_heads=config["model"]["n_heads"],
        dropout=config["model"]["dropout"],
        use_cross_attn=config["model"].get("use_cross_attn", True),
    ).to(device)
    logger.info(f"Model parameters: {count_parameters(model):,}")
    
    # Loss
    criterion = MultiTaskLoss(
        lambda_site=config["loss"]["lambda_site"],
        lambda_cons=config["loss"]["lambda_cons"],
        lambda_ctr=config["loss"].get("lambda_ctr", 0.1),
        use_contrastive=config.get("contrastive", {}).get("enabled", False),
        cons_aggregation=config["loss"]["cons_aggregation"],
        site_only_positive=config["loss"].get("site_only_positive", True),
    )
    
    optimizer = get_optimizer(model, config["train"])
    scheduler = create_scheduler(optimizer, config["train"])
    
    ema = None
    if config["train"].get("use_ema", False):
        ema = EMA(model, decay=config["train"].get("ema_decay", 0.999))
    
    # Train
    best_val_score = train(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, config["train"]["epochs"], config["train"]["patience"],
        config["train"]["grad_clip"], output_dir,
        config["eval"]["topk_recall"], config["eval"]["topk_ratio"],
        ema=ema,
        save_every=config["output"].get("save_every", 50),
        best_metric=config["eval"].get("best_metric", "int_aupr"),
    )
    
    # Test
    logger.info("Evaluating on test set...")
    ckpt = torch.load(os.path.join(output_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    
    test_metrics, test_preds = evaluate(
        model, test_loader, criterion, device,
        config["eval"]["topk_recall"], config["eval"]["topk_ratio"],
    )
    
    logger.info(f"Test - Int-AUROC: {test_metrics['int_auroc']:.4f}, Int-AUPR: {test_metrics['int_aupr']:.4f}")
    logger.info(f"Test - Site-AUROC: {test_metrics['site_micro_auroc']:.4f}")
    
    save_json(test_metrics, os.path.join(output_dir, "metrics.json"))
    
    if config["output"]["save_preds"]:
        preds_df = pd.DataFrame({
            "y_true": test_preds["y_true"],
            "y_prob": test_preds["y_prob"],
        })
        preds_df.to_csv(os.path.join(output_dir, "preds.csv"), index=False)
    
    logger.info("Training completed!")
    return test_metrics


if __name__ == "__main__":
    main()
