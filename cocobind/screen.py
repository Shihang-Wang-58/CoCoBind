#!/usr/bin/env python
"""
CoCoBind virtual screening CLI.

Rank a compound library against a single RNA sequence using the same model
and featurization stack as prediction/evaluation.
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .predict import load_model
from .featurizers import RNAFMFeaturizer, get_mol_featurizer


_SCRIPT_DIR = Path(__file__).parent.resolve()
_PROJECT_DIR = _SCRIPT_DIR.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _resolve_model_artifacts(model_id: str) -> Tuple[str, str]:
    models_root = _PROJECT_DIR / "models"
    if not models_root.exists():
        raise FileNotFoundError(f"Models directory not found: {models_root}")

    model_dir = models_root / model_id
    if not model_dir.exists():
        for candidate in models_root.iterdir():
            if candidate.is_dir() and candidate.name.lower() == model_id.lower():
                model_dir = candidate
                break

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found for model_id={model_id!r}")

    checkpoint_candidates = [
        model_dir / "best_model.pt",
        model_dir / "best.pt",
        model_dir / "checkpoint.pt",
    ]
    checkpoint_candidates.extend(sorted(model_dir.glob("*.pt")))
    checkpoint_path = next((path for path in checkpoint_candidates if path.exists()), None)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found under {model_dir}")

    config_candidates = [
        model_dir / "config_resolved.yaml",
        model_dir / "config.yaml",
    ]
    config_path = next((path for path in config_candidates if path.exists()), None)
    if config_path is None:
        raise FileNotFoundError(f"No config found under {model_dir}")

    return str(checkpoint_path), str(config_path)


def _resolve_device(args) -> torch.device:
    if args.device:
        return torch.device(args.device)
    if torch.cuda.is_available():
        gpu = 0 if args.gpu is None else args.gpu
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def _init_featurizers(config: dict, args, device: torch.device):
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
        if cfg_path:
            cfg_feature_path = Path(cfg_path)
            possible.append(cfg_feature_path if cfg_feature_path.is_absolute() else _PROJECT_DIR / cfg_feature_path)
        for path in possible:
            if path.exists():
                mol_features_path = str(path)
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
        device=str(device),
        n_bits=2048,
    )
    logger.info(f"Mol encoder: {mol_encoder}")
    return rna_featurizer, mol_featurizer


def _detect_columns(fieldnames: List[str], smiles_column: Optional[str], title_column: Optional[str]) -> Tuple[str, Optional[str]]:
    normalized = {name.lower(): name for name in fieldnames}

    if smiles_column:
        smiles_key = normalized.get(smiles_column.lower(), smiles_column)
    else:
        smiles_key = next((name for lower, name in normalized.items() if lower == "smiles"), None)
    if smiles_key is None:
        raise ValueError("Library CSV must contain a SMILES column or you must pass --smiles_column")

    if title_column:
        title_key = normalized.get(title_column.lower(), title_column)
    else:
        title_key = next(
            (name for lower, name in normalized.items() if lower in {"title", "name", "id", "compound_id"}),
            None,
        )

    return smiles_key, title_key


def load_library_compounds(
    library_path: str,
    max_candidates: Optional[int] = None,
    smiles_column: Optional[str] = None,
    title_column: Optional[str] = None,
) -> Dict[str, Any]:
    path = Path(library_path)
    if not path.exists():
        raise FileNotFoundError(f"Library CSV not found: {library_path}")

    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if not fieldnames:
            raise ValueError(f"Library CSV has no header row: {library_path}")
        smiles_key, title_key = _detect_columns(fieldnames, smiles_column, title_column)

        compounds = []
        skipped = 0
        for row_index, row in enumerate(reader, start=2):
            smiles = (row.get(smiles_key) or "").strip()
            if not smiles:
                skipped += 1
                continue

            compounds.append(
                {
                    "row_index": row_index,
                    "smiles": smiles,
                    "title": (row.get(title_key) or "").strip() if title_key else "",
                }
            )

            if max_candidates is not None and len(compounds) >= max_candidates:
                break

    return {
        "path": str(path),
        "compounds": compounds,
        "skipped": skipped,
        "smiles_column": smiles_key,
        "title_column": title_key,
    }


def screen_compounds(
    model: torch.nn.Module,
    rna_featurizer: RNAFMFeaturizer,
    mol_featurizer,
    sequence: str,
    compounds: List[Dict[str, Any]],
    device: torch.device,
    max_len: int = 512,
    batch_size: int = 64,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    seq_clean = sequence.upper().replace("T", "U").strip()
    if not seq_clean:
        raise ValueError("RNA sequence cannot be empty")

    with torch.no_grad():
        rna_embed = rna_featurizer(seq_clean, max_len=max_len)
        if isinstance(rna_embed, torch.Tensor):
            rna_embed = rna_embed.cpu().numpy()
        L = min(len(seq_clean), max_len, rna_embed.shape[0])
        rna_embed = rna_embed[:L]

        rna_base = torch.tensor(rna_embed, dtype=torch.float32).unsqueeze(0).to(device)
        model.eval()

        ranked_results: List[Dict[str, Any]] = []
        skipped_results: List[Dict[str, Any]] = []

        for start in range(0, len(compounds), batch_size):
            chunk = compounds[start:start + batch_size]
            mol_features = []
            valid_rows = []

            for row in chunk:
                try:
                    feature = mol_featurizer(row["smiles"])
                    if isinstance(feature, torch.Tensor):
                        feature = feature.float()
                    else:
                        feature = torch.tensor(np.asarray(feature), dtype=torch.float32)
                    mol_features.append(feature)
                    valid_rows.append(row)
                except (KeyError, ValueError) as exc:
                    skipped_results.append(
                        {
                            "title": row.get("title", ""),
                            "smiles": row["smiles"],
                            "error": str(exc),
                        }
                    )

            if not valid_rows:
                continue

            batch_n = len(valid_rows)
            rna_batch = rna_base.repeat(batch_n, 1, 1)
            mask = torch.ones(batch_n, L, dtype=torch.float32, device=device)
            mol_batch = torch.stack(mol_features).to(device)

            outputs = model({"rna_embed": rna_batch, "mol_fp": mol_batch, "rna_mask": mask})
            probs = torch.sigmoid(outputs["interaction_logit"]).detach().cpu().numpy()

            for row, prob in zip(valid_rows, probs):
                ranked_results.append(
                    {
                        "title": row.get("title", ""),
                        "smiles": row["smiles"],
                        "interaction_prob": round(float(prob), 6),
                        "row_index": row.get("row_index"),
                    }
                )

    ranked_results.sort(key=lambda item: item["interaction_prob"], reverse=True)
    for index, row in enumerate(ranked_results, start=1):
        row["rank"] = index
        row["sequence"] = seq_clean

    return ranked_results, skipped_results


def results_to_csv(results: List[Dict[str, Any]], path: str):
    fieldnames = ["rank", "title", "smiles", "interaction_prob", "sequence", "row_index"]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    logger.info(f"Saved {len(results)} ranked compounds -> {path}")


def format_screen_table(results: List[Dict[str, Any]], top_k: int) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append(f"Top {min(top_k, len(results))} compounds")
    lines.append(f"{'Rank':>4}  {'Score':>8}  {'Title':<24}  SMILES")
    for row in results[:top_k]:
        title = (row.get("title") or "").replace("\n", " ")[:24]
        smiles = row.get("smiles", "")
        lines.append(f"{row['rank']:>4}  {row['interaction_prob']:>8.4f}  {title:<24}  {smiles[:48]}")
    lines.append("=" * 78)
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="CoCoBind virtual screening CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Screen with a packaged model directory
  python -m cocobind screen \
      --model_id ECFP4 \
      --sequence "UGGGACACCCCUCCCCAACGAGGGGCGAAUAUCUGGAAGGAUA" \
      --library data/compound_library/MCE.csv \
      --top_k 20 \
      --max_candidates 48504 \
      --output outputs/screening/MCE_ECFP4.csv

  # Screen with an explicit checkpoint path
  python -m cocobind screen \
      --checkpoint models/ECFP4/best_model.pt \
      --sequence "UGGGACACCCCUCCCCAACGAGGGGCGAAUAUCUGGAAGGAUA" \
      --library data/compound_library/MCE.csv
""",
    )

    model_group = parser.add_mutually_exclusive_group(required=False)
    model_group.add_argument("--model_id", default=None, help="Model directory under models/ (for example ECFP4 or Ouroboros)")
    model_group.add_argument("--checkpoint", default=None, help="Path to model checkpoint (.pt file)")

    parser.add_argument("--config", default=None, help="Config file; auto-detected from checkpoint/model dir if omitted")
    parser.add_argument("--mol_encoder", default=None, choices=["ecfp4", "kpgt", "ouroboros"], help="Molecular encoder override")
    parser.add_argument("--mol_features_path", default=None, help="Precomputed molecular features (.pkl)")
    parser.add_argument("--mol_model_path", default=None, help="Molecular encoder model path for on-demand Ouroboros features")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--device", default=None, help="Explicit torch device string, for example cpu or cuda:0")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum RNA sequence length")

    parser.add_argument("--sequence", required=True, help="RNA sequence (A/U/G/C)")
    parser.add_argument("--library", required=True, help="Compound library CSV containing a SMILES column")
    parser.add_argument("--smiles_column", default=None, help="Explicit SMILES column name for the library CSV")
    parser.add_argument("--title_column", default=None, help="Explicit title/name column name for the library CSV")
    parser.add_argument("--top_k", type=int, default=20, help="Number of top compounds to print")
    parser.add_argument("--max_candidates", type=int, default=None, help="Limit the number of library rows read from the CSV")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for scoring")
    parser.add_argument("--output", default=None, help="Write the ranked results to CSV")
    parser.add_argument("--quiet", action="store_true", help="Suppress console table output")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.top_k < 1:
        raise ValueError("--top_k must be at least 1")
    if args.max_candidates is not None and args.max_candidates < 1:
        raise ValueError("--max_candidates must be at least 1 when provided")

    device = _resolve_device(args)
    logger.info(f"Device: {device}")

    if args.checkpoint:
        checkpoint_path = args.checkpoint
        config_path = args.config
        if config_path is None:
            checkpoint_dir = Path(checkpoint_path).parent
            if (checkpoint_dir / "config_resolved.yaml").exists():
                config_path = str(checkpoint_dir / "config_resolved.yaml")
            elif (checkpoint_dir / "config.yaml").exists():
                config_path = str(checkpoint_dir / "config.yaml")
    else:
        checkpoint_path, resolved_config_path = _resolve_model_artifacts(args.model_id or "ECFP4")
        config_path = args.config or resolved_config_path

    model, config = load_model(checkpoint_path, config_path, device, args.mol_encoder)
    rna_featurizer, mol_featurizer = _init_featurizers(config, args, device)

    library_payload = load_library_compounds(
        args.library,
        max_candidates=args.max_candidates,
        smiles_column=args.smiles_column,
        title_column=args.title_column,
    )
    compounds = library_payload["compounds"]
    if not compounds:
        raise ValueError(f"Library '{args.library}' has no usable SMILES rows")

    logger.info(
        f"Loaded {len(compounds)} candidates from {library_payload['path']} "
        f"(skipped empty rows: {library_payload['skipped']})"
    )

    results, skipped = screen_compounds(
        model,
        rna_featurizer,
        mol_featurizer,
        args.sequence,
        compounds,
        device,
        max_len=args.max_len,
        batch_size=args.batch_size,
    )

    if not args.quiet:
        print()
        print(format_screen_table(results, args.top_k))

    if args.output:
        results_to_csv(results, args.output)

    logger.info(
        f"Done. Ranked {len(results)} compounds. "
        f"Skipped {len(skipped)} compounds with feature/SMILES issues."
    )


if __name__ == "__main__":
    main()