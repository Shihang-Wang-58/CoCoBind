"""
CoCoBind Evaluation Script
Usage: python -m cocobind.eval --split unseen_pair --fold 0 --checkpoint outputs/...
"""
import os
import argparse
import logging
from pathlib import Path
from typing import Dict

import yaml
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .data import create_dataloaders
from .model import RNADTModel
from .losses import MultiTaskLoss
from .metrics import compute_all_metrics, format_metrics_table
from .utils import set_seed, setup_logging, load_config, save_json, get_device
from .featurizers import get_rna_embedding_dim, get_mol_feature_dim

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="CoCoBind Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--split", type=str, required=True,
                        choices=["unseen_pair", "unseen_rna", "unseen_compound", "unseen_both"],
                        help="Data split type")
    parser.add_argument("--fold", type=int, required=True,
                        choices=[0, 1, 2, 3, 4], help="Fold number")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config file")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Override data root directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for predictions")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch_size from config")
    parser.add_argument("--gpu", type=int, default=None,
                        help="GPU device ID")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cpu/cuda/cuda:0)")
    parser.add_argument("--save_preds", action="store_true",
                        help="Save predictions to CSV")
    parser.add_argument("--save_site_preds", action="store_true",
                        help="Save binding site predictions")
    # Molecular encoder
    parser.add_argument("--mol_encoder", type=str, default=None,
                        choices=["ecfp4", "kpgt", "ouroboros"],
                        help="Molecular encoder type")
    parser.add_argument("--mol_features_path", type=str, default=None,
                        help="Path to precomputed molecular features")
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
    topk_values: list = [10],
    topk_ratio: float = 0.05,
) -> Dict:
    """Evaluate model on a dataset."""
    model.eval()
    
    all_int_logits = []
    all_int_labels = []
    all_site_probs = []
    all_site_labels = []
    
    total_loss = 0.0
    n_batches = 0
    
    for batch in tqdm(loader, desc="Evaluating"):
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


def main():
    args = parse_args()
    
    # Load config
    if args.config:
        config_path = args.config
    else:
        # Try to find config next to checkpoint
        ckpt_dir = Path(args.checkpoint).parent
        if (ckpt_dir / "config_resolved.yaml").exists():
            config_path = str(ckpt_dir / "config_resolved.yaml")
        else:
            config_path = str(Path(__file__).parent / "config.yaml")
    
    config = load_config(config_path)
    
    # Override config with args
    if args.batch_size:
        config["train"]["batch_size"] = args.batch_size
    if args.gpu is not None:
        config["device"] = f"cuda:{args.gpu}"
    elif args.device:
        config["device"] = args.device
    if args.data_root:
        config["data"]["root"] = args.data_root
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = str(Path(args.checkpoint).parent / "eval_results")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    setup_logging(os.path.join(output_dir, "eval.log"))
    logger.info(f"Evaluating: split={args.split}, fold={args.fold}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    
    set_seed(config["seed"])
    
    device = get_device(config["device"])
    logger.info(f"Using device: {device}")
    
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
    logger.info(f"Test set size: {len(test_loader.dataset)}")
    
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
    
    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(ckpt)
        logger.info("Loaded checkpoint (raw state dict)")
    
    # Loss (for computing loss values)
    criterion = MultiTaskLoss(
        lambda_site=config["loss"]["lambda_site"],
        lambda_cons=config["loss"]["lambda_cons"],
        lambda_ctr=config["loss"].get("lambda_ctr", 0.1),
        use_contrastive=config.get("contrastive", {}).get("enabled", False),
        cons_aggregation=config["loss"]["cons_aggregation"],
        site_only_positive=config["loss"].get("site_only_positive", True),
    )
    
    # Evaluate
    test_metrics, test_preds = evaluate(
        model, test_loader, criterion, device,
        config["eval"]["topk_recall"], config["eval"]["topk_ratio"],
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(format_metrics_table(test_metrics))
    print("=" * 60)
    
    # Save metrics
    save_json(test_metrics, os.path.join(output_dir, "metrics.json"))
    logger.info(f"Metrics saved to {output_dir}/metrics.json")
    
    # Save predictions
    if args.save_preds:
        preds_df = pd.DataFrame({
            "y_true": test_preds["y_true"],
            "y_prob": test_preds["y_prob"],
        })
        preds_df.to_csv(os.path.join(output_dir, "preds.csv"), index=False)
        logger.info(f"Predictions saved to {output_dir}/preds.csv")
    
    if args.save_site_preds:
        import pickle
        with open(os.path.join(output_dir, "site_preds.pkl"), "wb") as f:
            pickle.dump({
                "site_labels": test_preds["site_labels"],
                "site_probs": test_preds["site_probs"],
            }, f)
        logger.info(f"Site predictions saved to {output_dir}/site_preds.pkl")
    
    logger.info("Evaluation completed!")
    return test_metrics


if __name__ == "__main__":
    main()
