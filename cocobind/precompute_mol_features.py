#!/usr/bin/env python
"""
分子特征预计算脚本

支持 KPGT 和 Ouroboros 模型预计算分子表征
预计算后的特征可直接用于训练，避免重复计算

Usage:
    # Ouroboros (推荐，更简洁)
    python -m cocobind.precompute_mol_features --model ouroboros \
        --model_path /home/wangshihang/software/Ouroboros/models/Ouroboros_M1c \
        --data_root /home/wangshihang/project/RNA_CRIBS/DeepRNA-DTI/Dataset \
        --output cache/ouroboros/unseen_pair_fold0_ouroboros.pkl \
        --csv_files examples/NPSL2.csv examples/MCE.csv

    # KPGT
    python -m cocobind.precompute_mol_features --model kpgt \
        --model_path /path/to/KPGT/models/pretrained/base/base.pth \
        --data_root /path/to/DeepRNA-DTI/Dataset \
        --output cache/kpgt/train_features.pkl
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def collect_all_smiles(data_root: str) -> Set[str]:
    """收集数据集中所有唯一的 SMILES"""
    all_smiles = set()
    data_root = Path(data_root)
    
    splits = ["unseen_pair", "unseen_rna", "unseen_compound", "unseen_both"]
    
    for split in splits:
        for data_type in ["dti_data", "bs_data"]:
            base_dir = data_root / split / data_type
            if not base_dir.exists():
                continue
            
            # 遍历所有 fold
            for subdir in base_dir.iterdir():
                csv_path = subdir / "raw" / "interactions.csv"
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        if "smiles" in df.columns:
                            all_smiles.update(df["smiles"].dropna().unique())
                    except Exception as e:
                        logger.warning(f"Failed to read {csv_path}: {e}")
    
    logger.info(f"Collected {len(all_smiles)} unique SMILES")
    return all_smiles


def collect_smiles_from_csvs(csv_files: List[str]) -> Set[str]:
    """从额外的 CSV 文件收集 SMILES"""
    smiles = set()
    for path in csv_files:
        p = Path(path)
        if not p.exists():
            logger.warning(f"CSV not found: {p}")
            continue
        try:
            df = pd.read_csv(p)
            cols = [c for c in df.columns if c.lower() == "smiles"]
            if not cols:
                logger.warning(f"No 'smiles' column in {p}")
                continue
            smiles.update(df[cols[0]].dropna().unique())
            logger.info(f"Collected {len(df[cols[0]].dropna().unique())} SMILES from {p}")
        except Exception as e:
            logger.warning(f"Failed to read {p}: {e}")
    return smiles


def filter_valid_smiles(smiles_list: List[str]) -> Tuple[List[str], List[str]]:
    """过滤无效 SMILES，返回 (valid, invalid)"""
    try:
        from rdkit import Chem
    except ImportError:
        logger.warning("rdkit not installed; skipping SMILES validation")
        return smiles_list, []
    valid, invalid = [], []
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                invalid.append(smi)
            else:
                valid.append(smi)
        except Exception:
            invalid.append(smi)
    return valid, invalid


def resolve_output_paths(output: str, output_name: str, model: str) -> Tuple[Path, Path]:
    """Resolve output directory and pickle path.

    --output can be either a directory or a concrete .pkl file path. If it is a
    directory, --output_name controls the file name.
    """
    output_path = Path(output)
    if output_path.suffix.lower() == ".pkl":
        if output_name:
            raise ValueError("--output_name cannot be used when --output is a .pkl file path")
        output_dir = output_path.parent
        feature_path = output_path
    else:
        output_dir = output_path
        feature_name = output_name or f"{model}_features.pkl"
        if not feature_name.endswith(".pkl"):
            feature_name = f"{feature_name}.pkl"
        feature_path = output_dir / feature_name
    return output_dir, feature_path


def extract_ouroboros_features(
    smiles_list: List[str],
    model_path: str,
    batch_size: int = 256,
    device: str = "cuda"
) -> Dict[str, np.ndarray]:
    """使用 Ouroboros 提取分子特征"""
    # 动态导入 Ouroboros
    ouroboros_dir = Path(model_path).parent.parent
    if str(ouroboros_dir) not in sys.path:
        sys.path.insert(0, str(ouroboros_dir))
    
    try:
        from ouroboros.model.GeminiMol import GeminiMol
    except ImportError:
        raise ImportError(
            f"Cannot import Ouroboros. Please ensure the path is correct: {ouroboros_dir}"
        )
    
    # 加载模型
    logger.info(f"Loading Ouroboros from {model_path}")
    model = GeminiMol(
        model_path=model_path,
        batch_size=batch_size,
        cache=True
    )
    model.eval()
    
    # 提取特征
    features_dict = {}
    failed_smiles = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="Extracting Ouroboros features"):
            batch = smiles_list[i:i+batch_size]
            try:
                features = model.encode(batch)
                features_np = features.cpu().numpy()
                for j, smi in enumerate(batch):
                    features_dict[smi] = features_np[j]
            except Exception as e:
                logger.warning(f"Failed batch {i}: {e}")
                # 逐个处理
                for smi in batch:
                    try:
                        feat = model.encode([smi])
                        features_dict[smi] = feat.cpu().numpy()[0]
                    except:
                        failed_smiles.append(smi)
    
    logger.info(f"Extracted {len(features_dict)} features, {len(failed_smiles)} failed")
    return features_dict


def extract_kpgt_features(
    smiles_list: List[str],
    model_path: str,
    kpgt_dir: str,
    batch_size: int = 64,
    device: str = "cuda"
) -> Dict[str, np.ndarray]:
    """使用 KPGT 提取分子特征"""
    # 动态导入 KPGT
    kpgt_src = Path(kpgt_dir) / "src"
    kpgt_scripts = Path(kpgt_dir) / "scripts"
    
    if str(kpgt_src) not in sys.path:
        sys.path.insert(0, str(kpgt_src))
    if str(kpgt_scripts) not in sys.path:
        sys.path.insert(0, str(kpgt_scripts))
    
    try:
        from model.light import LiGhTPredictor as LiGhT
        from model_config import config_dict
        from data.featurizer import Vocab, N_ATOM_TYPES, N_BOND_TYPES, smiles_to_graph_tune
        from data.collator import preprocess_batch_light
        from data.descriptors.rdNormalizedDescriptors import RDKit2DNormalized
        from rdkit import Chem
        import dgl
    except ImportError as e:
        raise ImportError(
            f"Cannot import KPGT modules. Please check the path: {kpgt_dir}\nError: {e}"
        )
    
    config = config_dict['base']
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    
    # 加载模型
    logger.info(f"Loading KPGT from {model_path}")
    model = LiGhT(
        d_node_feats=config['d_node_feats'],
        d_edge_feats=config['d_edge_feats'],
        d_g_feats=config['d_g_feats'],
        d_hpath_ratio=config['d_hpath_ratio'],
        n_mol_layers=config['n_mol_layers'],
        path_length=config['path_length'],
        n_heads=config['n_heads'],
        n_ffn_dense_layers=config['n_ffn_dense_layers'],
        input_drop=0,
        attn_drop=0,
        feat_drop=0,
        n_node_types=vocab.vocab_size
    ).to(device)
    
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(model_path).items()})
    model.eval()
    
    # 描述符生成器
    generator = RDKit2DNormalized()
    
    features_dict = {}
    failed_smiles = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(smiles_list), batch_size), desc="Extracting KPGT features"):
            batch = smiles_list[i:i+batch_size]
            
            try:
                # 构建图
                graphs = []
                fps = []
                mds = []
                valid_smiles = []
                
                for smi in batch:
                    g = smiles_to_graph_tune(smi, max_length=config['path_length'], n_virtual_nodes=2)
                    if g is None:
                        failed_smiles.append(smi)
                        continue
                    
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        failed_smiles.append(smi)
                        continue
                    
                    # RDK 指纹
                    fp = list(Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=512))
                    # 分子描述符
                    md = generator.process(smi)
                    if md is None or md[0] == False:
                        md = [0.0] * 201
                    else:
                        md = list(md[1:])
                    
                    graphs.append(g)
                    fps.append(torch.tensor(fp, dtype=torch.float32))
                    mds.append(torch.tensor(md, dtype=torch.float32))
                    valid_smiles.append(smi)
                
                if not graphs:
                    continue
                
                batched_graph = dgl.batch(graphs).to(device)
                fp_tensor = torch.stack(fps).to(device)
                md_tensor = torch.stack(mds).to(device)
                
                # 预处理路径
                batched_graph.edata['path'][:, :] = preprocess_batch_light(
                    batched_graph.batch_num_nodes(),
                    batched_graph.batch_num_edges(),
                    batched_graph.edata['path'][:, :]
                )
                
                # 提取特征
                feats = model.generate_fps(batched_graph, fp_tensor, md_tensor)
                feats_np = feats.cpu().numpy()
                
                for j, smi in enumerate(valid_smiles):
                    features_dict[smi] = feats_np[j]
                    
            except Exception as e:
                logger.warning(f"Failed batch {i}: {e}")
                for smi in batch:
                    if smi not in features_dict:
                        failed_smiles.append(smi)
    
    logger.info(f"Extracted {len(features_dict)} features, {len(failed_smiles)} failed")
    return features_dict


def main():
    parser = argparse.ArgumentParser(description="Precompute molecular features")
    parser.add_argument("--model", type=str, required=True, choices=["kpgt", "ouroboros"],
                        help="Model to use for feature extraction")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the pretrained model")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Path to DeepRNA-DTI Dataset directory")
    parser.add_argument("--output", type=str, default="cache/mol_features",
                        help="Output directory or a concrete .pkl feature file path")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Feature file name when --output is a directory")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for feature extraction")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--kpgt_dir", type=str, default=None,
                        help="KPGT project directory (required if model=kpgt)")
    parser.add_argument("--csv_files", type=str, nargs='*', default=None,
                        help="Additional CSV files containing a 'smiles' column")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    output_dir, output_path = resolve_output_paths(args.output, args.output_name, args.model)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有 SMILES
    if not args.data_root and not args.csv_files:
        raise ValueError("Provide --data_root and/or --csv_files to collect SMILES.")

    all_smiles = collect_all_smiles(args.data_root) if args.data_root else set()
    if args.csv_files:
        extra = collect_smiles_from_csvs(args.csv_files)
        before = len(all_smiles)
        all_smiles.update(extra)
        logger.info(f"Added {len(extra)} SMILES from extra CSVs (total {len(all_smiles)}, +{len(all_smiles)-before})")

    # 过滤无效 SMILES，避免下游 Ouroboros 报错
    all_smiles = list(all_smiles)
    valid_smiles, invalid_smiles = filter_valid_smiles(all_smiles)
    if invalid_smiles:
        logger.warning(f"Filtered out {len(invalid_smiles)} invalid SMILES before feature extraction")
        invalid_path = output_dir / "invalid_smiles.txt"
        with open(invalid_path, "w", encoding="utf-8") as f:
            for smi in invalid_smiles:
                f.write(smi + "\n")
        logger.warning(f"Invalid SMILES list saved to {invalid_path}")
    all_smiles = sorted(valid_smiles)
    
    # 提取特征
    if args.model == "ouroboros":
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "Ouroboros feature extraction requested CUDA, but torch.cuda.is_available() is false. "
                "Install a CUDA-enabled PyTorch/DGL stack or run on a GPU server."
            )
        if not args.device.startswith("cuda"):
            logger.warning("The bundled Ouroboros GeminiMol code uses CUDA internally; --device cpu may fail.")
        features_dict = extract_ouroboros_features(
            all_smiles,
            args.model_path,
            args.batch_size,
            args.device
        )
        feature_dim = next(iter(features_dict.values())).shape[0] if features_dict else 0
    else:
        if args.kpgt_dir is None:
            raise ValueError("--kpgt_dir is required when using KPGT")
        features_dict = extract_kpgt_features(
            all_smiles,
            args.model_path,
            args.kpgt_dir,
            args.batch_size,
            args.device
        )
        feature_dim = next(iter(features_dict.values())).shape[0] if features_dict else 0
    
    # 保存
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(features_dict, f)
    
    logger.info(f"Saved {len(features_dict)} features (dim={feature_dim}) to {output_path}")
    
    # 保存元信息
    meta = {
        "model": args.model,
        "model_path": args.model_path,
        "data_root": args.data_root,
        "csv_files": args.csv_files,
        "output_path": str(output_path),
        "num_features": len(features_dict),
        "feature_dim": feature_dim,
    }
    with open(output_dir / f"{args.model}_meta.json", 'w') as f:
        import json
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
