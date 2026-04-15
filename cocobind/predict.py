#!/usr/bin/env python
"""
predict.py — 新样本预测脚本
============================
给定 RNA 序列和化合物 SMILES，加载训练好的模型，输出：
  1. 相互作用概率 (interaction probability)
  2. 每个核苷酸的结合位点概率 (per-position binding site probability)
  3. 基于 NoisyOR 的一致性置信度指标

支持三种输入方式：
  - 命令行直接传入序列和 SMILES
  - 从 CSV 文件批量预测
  - 交互式逐条输入

Usage
-----
# 单条预测
python -m cocobind.predict \\
    --checkpoint models/ECFP4/best_model.pt \\
    --sequence "GGCUAGCUAUAGC" \\
    --smiles "CC1=NC2=CC=CC=C2N1"

# CSV 批量预测
python -m cocobind.predict \\
    --checkpoint models/ECFP4/best_model.pt \\
    --input_csv samples.csv \\
    --output_csv predictions.csv

# 交互模式
python -m cocobind.predict \\
    --checkpoint models/ECFP4/best_model.pt \\
    --interactive
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import yaml

# ── Path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR  = Path(__file__).parent.resolve()
_PROJECT_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

try:
    from .model import RNADTModel
    from .featurizers import RNAFMFeaturizer, get_mol_featurizer, get_mol_feature_dim
except ImportError:
    from cocobind.model import RNADTModel
    from cocobind.featurizers import RNAFMFeaturizer, get_mol_featurizer, get_mol_feature_dim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# Model loading
# ============================================================================

def load_model(
    checkpoint_path: str,
    config_path: Optional[str],
    device: torch.device,
    mol_encoder_override: Optional[str] = None,
) -> Tuple[RNADTModel, dict]:
    """加载模型和配置"""
    checkpoint_dir = Path(checkpoint_path).parent

    if config_path is None:
        config_path = checkpoint_dir / "config_resolved.yaml"
        if not Path(config_path).exists():
            config_path = checkpoint_dir / "config.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Config: {config_path}")

    if mol_encoder_override:
        config.setdefault("mol_encoder", {})
        config["mol_encoder"]["type"] = mol_encoder_override

    model_cfg = config.get("model", {})
    mol_cfg = config.get("mol_encoder", {}) or {}
    mol_encoder = mol_cfg.get("type", "ecfp4")
    mol_features_path = mol_cfg.get("features_path", None)
    d_mol = get_mol_feature_dim(mol_encoder, mol_features_path)

    model = RNADTModel(
        d_rna=640,
        d_mol=model_cfg.get("d_mol", d_mol),
        d_model=model_cfg.get("d_model", 256),
        n_mol_tokens=model_cfg.get("n_mol_tokens", 4),
        n_heads=model_cfg.get("n_heads", 4),
        dropout=model_cfg.get("dropout", 0.1),
        use_cross_attn=model_cfg.get("use_cross_attn", True),
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model = model.to(device).eval()

    logger.info(f"Model loaded: {checkpoint_path}  "
                f"(d_model={model_cfg.get('d_model')}, "
                f"mol_encoder={mol_encoder}, "
                f"n_heads={model_cfg.get('n_heads')}, "
                f"cross_attn={model_cfg.get('use_cross_attn', True)})")
    return model, config


# ============================================================================
# Single-sample prediction
# ============================================================================

def predict_single(
    model: RNADTModel,
    rna_featurizer: RNAFMFeaturizer,
    mol_featurizer,
    sequence: str,
    smiles: str,
    device: torch.device,
    max_len: int = 512,
) -> Dict:
    """
    对单个 RNA-化合物对做完整预测。

    Returns
    -------
    dict with keys:
        sequence          : str   — 输入序列（预处理后）
        smiles            : str   — 输入 SMILES
        seq_len           : int   — 有效序列长度
        interaction_prob  : float — 相互作用概率 ∈ [0, 1]
        noisy_or_prob     : float — NoisyOR(p_site): 1 - ∏(1-p_i)
        consistency_gap   : float — |p_int − NoisyOR(p_site)|
        confidence        : str   — 综合置信度等级 (high / medium / low)
        site_probs        : list  — 逐位点结合概率 [L]
        top_sites         : list  — top-K 高概率结合位点 [{pos, nt, prob}]
    """
    seq_clean = sequence.upper().replace("T", "U").strip()
    L = min(len(seq_clean), max_len)

    with torch.no_grad():
        rna_embed = rna_featurizer(seq_clean, max_len=max_len)
        if isinstance(rna_embed, torch.Tensor):
            rna_embed = rna_embed.numpy()
        rna_embed = rna_embed[:L]

        mol_fp = mol_featurizer(smiles)
        if isinstance(mol_fp, torch.Tensor):
            mol_fp = mol_fp.numpy()

        rna_t = torch.tensor(rna_embed, dtype=torch.float32).unsqueeze(0).to(device)
        mol_t = torch.tensor(mol_fp,    dtype=torch.float32).unsqueeze(0).to(device)
        mask  = torch.ones(1, L, dtype=torch.float32).to(device)

        outputs = model({"rna_embed": rna_t, "mol_fp": mol_t, "rna_mask": mask})

        p_int  = float(torch.sigmoid(outputs["interaction_logit"]).cpu().item())
        p_site = torch.sigmoid(outputs["site_logits"]).cpu().numpy()[0, :L]

    # NoisyOR: p_any = 1 − ∏(1 − p_site_i)
    noisy_or = float(1.0 - np.prod(1.0 - np.clip(p_site, 0, 1 - 1e-7)))
    gap = abs(p_int - noisy_or)
    confidence = _assess_confidence(p_int, noisy_or, gap)

    # Top binding sites
    sorted_idx = np.argsort(p_site)[::-1]
    top_sites = []
    for i in sorted_idx[:10]:
        prob = float(p_site[i])
        if prob < 0.05 and len(top_sites) >= 3:
            break
        top_sites.append({
            "pos": int(i) + 1,
            "nt":  seq_clean[i] if i < len(seq_clean) else "?",
            "prob": round(prob, 4),
        })

    return {
        "sequence":         seq_clean[:L],
        "smiles":           smiles,
        "seq_len":          L,
        "interaction_prob": round(p_int, 4),
        "noisy_or_prob":    round(noisy_or, 4),
        "consistency_gap":  round(gap, 4),
        "confidence":       confidence,
        "site_probs":       [round(float(v), 4) for v in p_site],
        "top_sites":        top_sites,
    }


def _assess_confidence(p_int: float, noisy_or: float, gap: float) -> str:
    """
    综合置信度评估。

    high   : p_int 和 NoisyOR 一致 (gap<0.15) 且远离 0.5 边界
    medium : 基本一致 (gap<0.25) 或中等距离
    low    : 两头矛盾或不确定
    """
    boundary_dist = min(abs(p_int - 0.5), abs(noisy_or - 0.5))
    if gap < 0.15 and boundary_dist > 0.3:
        return "high"
    elif gap < 0.25 and boundary_dist > 0.15:
        return "medium"
    else:
        return "low"


# ============================================================================
# Batch prediction
# ============================================================================

def predict_batch(
    model: RNADTModel,
    rna_featurizer: RNAFMFeaturizer,
    mol_featurizer,
    pairs: List[Tuple[str, str]],
    device: torch.device,
    max_len: int = 512,
) -> List[Dict]:
    """批量预测，返回结果列表。"""
    results = []
    for i, (seq, smi) in enumerate(pairs):
        if (i + 1) % 50 == 0 or i == 0:
            logger.info(f"  [{i+1}/{len(pairs)}]")
        try:
            res = predict_single(model, rna_featurizer, mol_featurizer,
                                 seq, smi, device, max_len)
            results.append(res)
        except Exception as e:
            logger.error(f"  [{i+1}] Failed: {e}")
            results.append({"sequence": seq[:50], "smiles": smi[:50], "error": str(e)})
    return results


# ============================================================================
# Output formatting
# ============================================================================

def format_result(res: Dict, verbose: bool = True) -> str:
    """格式化单条预测结果为人类可读字符串。"""
    if "error" in res:
        return f"ERROR: {res['error']}"

    p = res["interaction_prob"]
    verdict = "INTERACTS" if p >= 0.5 else "No interaction"
    conf = res["confidence"].upper()

    lines = [
        "─" * 70,
        f"  Sequence : {res['sequence'][:80]}{'...' if len(res['sequence']) > 80 else ''}",
        f"  SMILES   : {res['smiles'][:80]}{'...' if len(res['smiles']) > 80 else ''}",
        f"  Seq len  : {res['seq_len']}",
        "",
        f"  Interaction prob   : {p:.4f}  → {verdict}",
        f"  NoisyOR(p_site)   : {res['noisy_or_prob']:.4f}",
        f"  Consistency gap   : {res['consistency_gap']:.4f}",
        f"  Confidence        : {conf}",
    ]

    if verbose and res.get("top_sites"):
        lines.append("")
        lines.append("  Top predicted binding sites:")
        lines.append(f"    {'Pos':>5}  {'Nt':>3}  {'p_site':>8}")
        for s in res["top_sites"]:
            marker = " ◀" if s["prob"] >= 0.5 else ""
            lines.append(f"    {s['pos']:>5}  {s['nt']:>3}  {s['prob']:>8.4f}{marker}")

    lines.append("─" * 70)
    return "\n".join(lines)


def results_to_csv(results: List[Dict], path: str):
    """将预测结果保存为 CSV。"""
    fieldnames = [
        "sequence", "smiles", "seq_len",
        "interaction_prob", "noisy_or_prob", "consistency_gap", "confidence",
        "n_sites_above_0.5", "top_site_pos", "top_site_prob",
        "site_probs",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            if "error" in r:
                writer.writerow({"sequence": r.get("sequence", ""),
                                 "smiles": r.get("smiles", ""),
                                 "interaction_prob": "ERROR",
                                 "confidence": r["error"]})
                continue
            sp = r.get("site_probs", [])
            top = r.get("top_sites", [{}])[0] if r.get("top_sites") else {}
            writer.writerow({
                "sequence":           r["sequence"],
                "smiles":             r["smiles"],
                "seq_len":            r["seq_len"],
                "interaction_prob":   r["interaction_prob"],
                "noisy_or_prob":      r["noisy_or_prob"],
                "consistency_gap":    r["consistency_gap"],
                "confidence":         r["confidence"],
                "n_sites_above_0.5":  sum(1 for v in sp if v >= 0.5),
                "top_site_pos":       top.get("pos", ""),
                "top_site_prob":      top.get("prob", ""),
                "site_probs":         json.dumps(sp),
            })
    logger.info(f"Saved {len(results)} predictions → {path}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="CoCoBind Prediction — 新 RNA-化合物对的互作 & 结合位点预测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 单条
  python -m cocobind.predict \\
      --checkpoint models/ECFP4/best_model.pt \\
      --sequence "GGCUAGCUAUAGCUAGC" \\
      --smiles "CC1=NC2=CC=CC=C2N1"

  # CSV 批量 (需含 sequence, smiles 列)
  python -m cocobind.predict \\
      --checkpoint models/ECFP4/best_model.pt \\
      --input_csv samples.csv --output_csv predictions.csv

  # 交互模式
  python -m cocobind.predict \\
      --checkpoint models/ECFP4/best_model.pt \\
      --interactive
""",
    )
    # Model
    p.add_argument("--checkpoint", required=True,
                   help="Path to model checkpoint (.pt)")
    p.add_argument("--config", default=None,
                   help="Config file; auto-detected from checkpoint dir if omitted")
    p.add_argument("--mol_encoder", default=None,
                   choices=["ecfp4", "kpgt", "ouroboros"],
                   help="Molecular encoder (default: read from config). "
                        "ecfp4 can encode any SMILES on-the-fly; "
                        "kpgt uses precomputed features; ouroboros can use --mol_features_path "
                        "or --mol_model_path")
    p.add_argument("--mol_features_path", default=None,
                   help="Precomputed molecular features (.pkl). "
                        "Recommended for kpgt/ouroboros; generate with: "
                        "python -m cocobind.precompute_mol_features --model ouroboros ...")
    p.add_argument("--mol_model_path", default=None,
                   help="Molecular encoder model path for on-demand Ouroboros features")
    p.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    p.add_argument("--max_len", type=int, default=512,
                   help="Maximum RNA sequence length")

    # Input: single pair
    p.add_argument("--sequence", default=None,
                   help="RNA sequence (A/U/G/C)")
    p.add_argument("--smiles", default=None,
                   help="Compound SMILES string")

    # Input: batch
    p.add_argument("--input_csv", default=None,
                   help="CSV with 'sequence' and 'smiles' columns")

    # Input: interactive
    p.add_argument("--interactive", action="store_true",
                   help="Enter pairs interactively")

    # Output
    p.add_argument("--output_csv", default=None,
                   help="Save predictions to CSV")
    p.add_argument("--output_json", default=None,
                   help="Save predictions to JSON (includes full site_probs)")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-sample console output")

    return p.parse_args()


def _init_featurizers(config: dict, args) -> Tuple:
    """初始化 RNA 和分子 featurizer。"""
    cache_dir = config.get("data", {}).get("cache_dir", "cache")
    if not Path(cache_dir).is_absolute():
        cache_dir = str(_PROJECT_DIR / cache_dir)
    rna_featurizer = RNAFMFeaturizer(cache_dir=str(cache_dir))

    mol_encoder = args.mol_encoder or config.get("mol_encoder", {}).get("type", "ecfp4")
    mol_features_path = args.mol_features_path
    mol_model_path = args.mol_model_path or config.get("mol_encoder", {}).get("model_path")
    mol_cache_dir = config.get("mol_encoder", {}).get("cache_dir")
    if mol_features_path and not Path(mol_features_path).is_absolute():
        mol_features_path = str(_PROJECT_DIR / mol_features_path)
    if mol_features_path and not Path(mol_features_path).exists():
        logger.warning(f"Precomputed molecular features not found: {mol_features_path}")
        mol_features_path = None
    if mol_model_path and not Path(mol_model_path).is_absolute():
        mol_model_path = str(_PROJECT_DIR / mol_model_path)
    if mol_encoder == "ouroboros" and mol_model_path is None:
        default_model_path = _PROJECT_DIR / "Ouroboros" / "models" / "Ouroboros_M1c"
        if default_model_path.exists():
            mol_model_path = str(default_model_path)
    if mol_cache_dir and not Path(mol_cache_dir).is_absolute():
        mol_cache_dir = str(_PROJECT_DIR / mol_cache_dir)

    if mol_features_path is None and mol_model_path is None and mol_encoder != "ecfp4":
        cfg_path = config.get("mol_encoder", {}).get("features_path", "")
        possible = [
            _PROJECT_DIR / f"cache/mol_features/{mol_encoder}_features.pkl",
            _SCRIPT_DIR / f"cache/mol_features/{mol_encoder}_features.pkl",
        ]
        if cfg_path:  # avoid Path("").exists() edge case
            cfg_feature_path = Path(cfg_path)
            possible.append(cfg_feature_path if cfg_feature_path.is_absolute() else _PROJECT_DIR / cfg_feature_path)
        for pp in possible:
            if pp.exists():
                mol_features_path = str(pp)
                logger.info(f"Auto-detected features: {mol_features_path}")
                break
        if mol_features_path is None:
            raise FileNotFoundError(
                f"Precomputed features for {mol_encoder} not found at any auto-detected path. "
                "Pass --mol_features_path or set mol_encoder.features_path in the config. "
                f"To create them, run: python -m cocobind.precompute_mol_features --model {mol_encoder} ..."
            )

    mol_featurizer = get_mol_featurizer(
        mol_encoder=mol_encoder,
        mol_features_path=mol_features_path,
        mol_model_path=mol_model_path,
        mol_cache_dir=mol_cache_dir,
        device=str(torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")),
    )
    logger.info(f"Mol encoder: {mol_encoder}")
    return rna_featurizer, mol_featurizer


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────
    model, config = load_model(args.checkpoint, args.config, device, args.mol_encoder)
    rna_feat, mol_feat = _init_featurizers(config, args)

    # ── Collect input pairs ───────────────────────────────────────────────
    pairs: List[Tuple[str, str]] = []

    if args.sequence and args.smiles:
        pairs.append((args.sequence, args.smiles))

    if args.input_csv:
        import pandas as pd
        df = pd.read_csv(args.input_csv)
        if "sequence" not in df.columns or "smiles" not in df.columns:
            logger.error("CSV must contain 'sequence' and 'smiles' columns.")
            sys.exit(1)
        for _, row in df.iterrows():
            pairs.append((str(row["sequence"]), str(row["smiles"])))
        logger.info(f"Loaded {len(df)} pairs from {args.input_csv}")

    # ── Run predictions ───────────────────────────────────────────────────
    all_results: List[Dict] = []

    if pairs:
        results = predict_batch(model, rna_feat, mol_feat, pairs, device, args.max_len)
        all_results.extend(results)
        if not args.quiet:
            for r in results:
                print(format_result(r))

    # ── Interactive mode ──────────────────────────────────────────────────
    if args.interactive:
        print("\n  CoCoBind 交互预测 (输入 'q' 退出)\n")
        while True:
            try:
                seq = input("  RNA sequence: ").strip()
                if seq.lower() in ("q", "quit", "exit", ""):
                    break
                smi = input("  SMILES      : ").strip()
                if smi.lower() in ("q", "quit", "exit", ""):
                    break
            except (EOFError, KeyboardInterrupt):
                break

            res = predict_single(model, rna_feat, mol_feat, seq, smi, device, args.max_len)
            all_results.append(res)
            print(format_result(res))

    # ── Save outputs ──────────────────────────────────────────────────────
    if not all_results:
        if not args.interactive:
            logger.error("No input provided. Use --sequence/--smiles, --input_csv, or --interactive.")
            sys.exit(1)
        return

    if args.output_csv:
        results_to_csv(all_results, args.output_csv)

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved JSON → {args.output_json}")

    # ── Summary ───────────────────────────────────────────────────────────
    n = len(all_results)
    n_int = sum(1 for r in all_results if r.get("interaction_prob", 0) >= 0.5)
    n_high = sum(1 for r in all_results if r.get("confidence") == "high")
    logger.info(f"Done. {n} predictions: {n_int} interacting, {n_high} high-confidence.")


if __name__ == "__main__":
    main()
