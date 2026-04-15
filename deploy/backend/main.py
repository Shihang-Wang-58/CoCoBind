"""
CoCoBind API Server
====================
FastAPI backend integrating cocobind/predict.py for real inference.
Supports:
 - Single prediction (POST /predict)
 - Batch CSV prediction (POST /predict_batch)
 - Job tracking (GET /jobs, GET /jobs/{job_id})
 - Input validation
"""

import io
import csv
import os
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import yaml
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

# ── Path setup so we can import from cocobind ──
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).parent.resolve()
_PROJECT_DIR = _BACKEND_DIR.parent.parent  # deploy/backend/../../ => project root
sys.path.insert(0, str(_PROJECT_DIR))

from cocobind.model import RNADTModel
from cocobind.featurizers import RNAFMFeaturizer, get_mol_featurizer
from cocobind.predict import predict_single, predict_batch, load_model

# ============================================================================
# App setup
# ============================================================================
app = FastAPI(title="CoCoBind API", version="1.0.0")


@app.middleware("http")
async def normalize_path_slashes(request: Request, call_next):
    path = request.scope.get("path", "")
    if "//" in path:
        normalized = re.sub(r"/+", "/", path)
        request.scope["path"] = normalized if normalized else "/"
    return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Model registry and library discovery
# ============================================================================
DEVICE = torch.device(os.getenv("COCOBIND_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
MODELS_DIR = Path(os.getenv("COCOBIND_MODELS_DIR", _PROJECT_DIR / "models"))
COMPOUND_LIBRARY_DIR = Path(os.getenv("COCOBIND_LIBRARY_DIR", _PROJECT_DIR / "data" / "compound_library"))
DEFAULT_MODEL_ID = os.getenv("COCOBIND_DEFAULT_MODEL", "ECFP4")
DEFAULT_OUROBOROS_MODEL_PATH = _PROJECT_DIR / "Ouroboros" / "models" / "Ouroboros_M1c"


def resolve_project_path(value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else _PROJECT_DIR / path


def _resolve_optional_path(value: Optional[str]) -> Optional[str]:
    if value in (None, "", "none", "None", "null", "Null"):
        return None
    path = Path(str(value))
    return str(path if path.is_absolute() else _PROJECT_DIR / path)


def _find_checkpoint(model_dir: Path) -> Optional[Path]:
    for name in ("best_model.pt", "best.pt", "checkpoint.pt"):
        candidate = model_dir / name
        if candidate.exists():
            return candidate
    pts = sorted(model_dir.glob("*.pt"))
    return pts[0] if pts else None


def discover_models() -> Dict[str, Dict[str, Any]]:
    registry: Dict[str, Dict[str, Any]] = {}
    if MODELS_DIR.exists():
        for model_dir in sorted(p for p in MODELS_DIR.iterdir() if p.is_dir()):
            config_path = model_dir / "config_resolved.yaml"
            checkpoint_path = _find_checkpoint(model_dir)
            if not config_path.exists() or checkpoint_path is None:
                continue
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            mol_cfg = cfg.get("mol_encoder", {}) or {}
            model_id = model_dir.name
            registry[model_id] = {
                "id": model_id,
                "label": model_id,
                "config_path": config_path,
                "checkpoint_path": checkpoint_path,
                "mol_encoder": mol_cfg.get("type", "ecfp4"),
                "ready": True,
            }

    env_checkpoint = os.getenv("COCOBIND_CHECKPOINT")
    env_config = os.getenv("COCOBIND_CONFIG")
    if env_checkpoint and env_config:
        registry.setdefault(DEFAULT_MODEL_ID, {
            "id": DEFAULT_MODEL_ID,
            "label": DEFAULT_MODEL_ID,
            "config_path": resolve_project_path(env_config),
            "checkpoint_path": resolve_project_path(env_checkpoint),
            "mol_encoder": os.getenv("COCOBIND_MOL_ENCODER", "ecfp4"),
            "ready": True,
        })
    return registry


MODEL_REGISTRY = discover_models()
MODEL_CACHE: Dict[str, Dict[str, Any]] = {}
DEFAULT_MAX_LEN = 512


def normalize_model_id(model_id: Optional[str]) -> str:
    if not MODEL_REGISTRY:
        raise HTTPException(status_code=500, detail=f"No model directories found in {MODELS_DIR}")
    requested = model_id or DEFAULT_MODEL_ID
    if requested in MODEL_REGISTRY:
        return requested
    lowered = requested.lower()
    for key in MODEL_REGISTRY:
        if key.lower() == lowered:
            return key
    raise HTTPException(status_code=404, detail=f"Unknown model '{requested}'. Available: {list(MODEL_REGISTRY)}")


def get_model_bundle(model_id: Optional[str] = None) -> Dict[str, Any]:
    key = normalize_model_id(model_id)
    if key in MODEL_CACHE:
        return MODEL_CACHE[key]

    info = MODEL_REGISTRY[key]
    config_path = Path(info["config_path"])
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    mol_cfg = cfg.get("mol_encoder", {}) or {}
    mol_encoder = mol_cfg.get("type", info.get("mol_encoder", "ecfp4"))
    features_path = _resolve_optional_path(mol_cfg.get("features_path"))
    if features_path and not Path(features_path).exists():
        features_path = None

    env_prefix = f"COCOBIND_{key.upper()}_"
    features_path = _resolve_optional_path(os.getenv(env_prefix + "MOL_FEATURES_PATH")) or features_path
    model_path = _resolve_optional_path(
        os.getenv(env_prefix + "MOL_MODEL_PATH")
        or os.getenv("COCOBIND_OUROBOROS_MODEL_PATH")
        or mol_cfg.get("model_path")
    )
    if mol_encoder == "ouroboros" and model_path is None and DEFAULT_OUROBOROS_MODEL_PATH.exists():
        model_path = str(DEFAULT_OUROBOROS_MODEL_PATH)

    cache_dir = cfg.get("data", {}).get("cache_dir", "cache/rna_embeddings")
    rna_cache_dir = str(resolve_project_path(cache_dir))
    mol_cache_dir = _resolve_optional_path(mol_cfg.get("cache_dir")) or str(_PROJECT_DIR / "cache" / "mol_features" / key)

    model, loaded_config = load_model(
        str(info["checkpoint_path"]),
        str(config_path),
        DEVICE,
        mol_encoder_override=mol_encoder,
    )
    bundle = {
        "id": key,
        "config": loaded_config,
        "model": model,
        "rna_featurizer": RNAFMFeaturizer(cache_dir=rna_cache_dir, device=str(DEVICE)),
        "mol_featurizer": get_mol_featurizer(
            mol_encoder=mol_encoder,
            mol_features_path=features_path,
            mol_model_path=model_path,
            mol_cache_dir=mol_cache_dir,
            device=str(DEVICE),
            n_bits=2048,
        ),
        "mol_encoder": mol_encoder,
        "max_len": loaded_config.get("data", {}).get("max_len", 512),
        "checkpoint_path": str(info["checkpoint_path"]),
        "config_path": str(config_path),
    }
    MODEL_CACHE[key] = bundle
    print(f"Model loaded: {key} ({mol_encoder}) from {info['checkpoint_path']}")
    return bundle


def list_compound_libraries() -> List[Dict[str, Any]]:
    libraries = []
    if not COMPOUND_LIBRARY_DIR.exists():
        return libraries
    for path in sorted(COMPOUND_LIBRARY_DIR.glob("*.csv")):
        try:
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                columns = reader.fieldnames or []
            smiles_col = next((c for c in columns if c.lower() == "smiles"), None)
            title_col = next((c for c in columns if c.lower() in {"title", "name", "id", "compound_id"}), None)
            with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
                n_rows = max(sum(1 for _ in f) - 1, 0)
            libraries.append({
                "id": path.stem,
                "name": path.stem,
                "filename": path.name,
                "n_rows": n_rows,
                "smiles_col": smiles_col,
                "title_col": title_col,
                "ready": smiles_col is not None,
            })
        except Exception as exc:
            libraries.append({
                "id": path.stem,
                "name": path.stem,
                "filename": path.name,
                "ready": False,
                "error": str(exc),
            })
    return libraries


def get_library_path(library_id: str) -> Path:
    for item in list_compound_libraries():
        if item["id"] == library_id:
            return COMPOUND_LIBRARY_DIR / item["filename"]
    raise HTTPException(status_code=404, detail=f"Unknown compound library '{library_id}'")

# ============================================================================
# Job store (in-memory)
# ============================================================================
jobs_store: dict = {}  # job_id -> {id, status, input, result, created_at, completed_at}


def create_job(input_summary: dict) -> str:
    """Create a new job entry and return job_id."""
    job_id = str(uuid.uuid4())[:8]
    jobs_store[job_id] = {
        "id": job_id,
        "status": "running",
        "type": input_summary.get("type", "single"),
        "input_summary": input_summary,
        "result": None,
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
    }
    return job_id


def complete_job(job_id: str, result, status: str = "completed"):
    """Mark a job as completed."""
    if job_id in jobs_store:
        jobs_store[job_id]["status"] = status
        jobs_store[job_id]["result"] = result
        jobs_store[job_id]["completed_at"] = datetime.now().isoformat()


# ============================================================================
# Validation
# ============================================================================
RNA_PATTERN = re.compile(r'^[AUGCaugcTt]+$')
# Basic SMILES pattern: common valid characters
SMILES_PATTERN = re.compile(r'^[A-Za-z0-9@+\-\[\]\(\)\\/=#$%.:~*{}!]+$')


def validate_rna(seq: str) -> str:
    """Validate and normalize RNA sequence."""
    seq = seq.strip().upper().replace("T", "U")
    if not seq:
        raise ValueError("RNA sequence cannot be empty")
    if not RNA_PATTERN.match(seq):
        raise ValueError(
            "Invalid RNA sequence. Only nucleotides A, U, G, C are allowed. "
            "DNA sequences (with T) are auto-converted to RNA (U)."
        )
    if len(seq) > DEFAULT_MAX_LEN:
        raise ValueError(f"RNA sequence too long ({len(seq)} nt). Maximum is {DEFAULT_MAX_LEN}.")
    return seq


def validate_smiles(smi: str) -> str:
    """Basic SMILES validation."""
    smi = smi.strip()
    if not smi:
        raise ValueError("SMILES string cannot be empty")
    if not SMILES_PATTERN.match(smi):
        raise ValueError("Invalid SMILES format. Please check for illegal characters.")
    return smi


# ============================================================================
# Request / Response Models
# ============================================================================
class PredictionRequest(BaseModel):
    rna_sequence: str
    smiles: str
    model_id: str = DEFAULT_MODEL_ID

    @field_validator("rna_sequence")
    @classmethod
    def check_rna(cls, v):
        return validate_rna(v)

    @field_validator("smiles")
    @classmethod
    def check_smiles(cls, v):
        return validate_smiles(v)


class ScreeningRequest(BaseModel):
    rna_sequence: str
    library_id: str
    model_id: str = DEFAULT_MODEL_ID
    top_k: int = 20
    max_candidates: int = 20000

    @field_validator("rna_sequence")
    @classmethod
    def check_rna(cls, v):
        return validate_rna(v)

    @field_validator("top_k")
    @classmethod
    def check_top_k(cls, v):
        if v < 1 or v > 200:
            raise ValueError("top_k must be between 1 and 200")
        return v

    @field_validator("max_candidates")
    @classmethod
    def check_max_candidates(cls, v):
        if v < 1 or v > 200000:
            raise ValueError("max_candidates must be between 1 and 200000")
        return v


def score_compounds(
    bundle: Dict[str, Any],
    sequence: str,
    compounds: List[Dict[str, str]],
    top_k: int,
    batch_size: int = 64,
) -> List[Dict[str, Any]]:
    seq_clean = sequence.upper().replace("T", "U").strip()
    max_len = bundle["max_len"]
    rna_embed = bundle["rna_featurizer"](seq_clean, max_len=max_len).float()
    L = min(rna_embed.shape[0], max_len)
    rna_embed = rna_embed[:L]

    model = bundle["model"]
    mol_featurizer = bundle["mol_featurizer"]
    results = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(compounds), batch_size):
            chunk = compounds[start:start + batch_size]
            mol_features = []
            valid_rows = []
            for row in chunk:
                try:
                    mol_features.append(mol_featurizer(row["smiles"]).float())
                    valid_rows.append(row)
                except (KeyError, ValueError) as exc:
                    row["error"] = str(exc)
                except Exception:
                    raise
            if not valid_rows:
                continue

            batch_n = len(valid_rows)
            rna_t = rna_embed.unsqueeze(0).repeat(batch_n, 1, 1).to(DEVICE)
            mask = torch.ones(batch_n, L, dtype=torch.float32, device=DEVICE)
            mol_t = torch.stack(mol_features).to(DEVICE)
            outputs = model({"rna_embed": rna_t, "mol_fp": mol_t, "rna_mask": mask})
            probs = torch.sigmoid(outputs["interaction_logit"]).detach().cpu().numpy()

            for row, prob in zip(valid_rows, probs):
                results.append({
                    "title": row.get("title", ""),
                    "smiles": row["smiles"],
                    "interaction_prob": round(float(prob), 6),
                })

    results.sort(key=lambda item: item["interaction_prob"], reverse=True)
    return results[:top_k]


def load_library_compounds(library_id: str, max_candidates: int) -> Dict[str, Any]:
    path = get_library_path(library_id)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        smiles_col = next((c for c in columns if c.lower() == "smiles"), None)
        title_col = next((c for c in columns if c.lower() in {"title", "name", "id", "compound_id"}), None)
        if smiles_col is None:
            raise HTTPException(status_code=400, detail=f"Library '{library_id}' has no SMILES column")

        compounds = []
        skipped = 0
        for row in reader:
            smi = (row.get(smiles_col) or "").strip()
            if not smi:
                skipped += 1
                continue
            compounds.append({
                "smiles": smi,
                "title": (row.get(title_col) or "") if title_col else "",
            })
            if len(compounds) >= max_candidates:
                break

    return {
        "path": str(path),
        "compounds": compounds,
        "skipped": skipped,
    }


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
@app.get("/api/")
async def root():
    return {
        "message": "CoCoBind API is running",
        "device": str(DEVICE),
        "models": list(MODEL_REGISTRY.keys()),
    }


@app.get("/models")
@app.get("/api/models")
async def list_models():
    return [
        {
            "id": info["id"],
            "label": info["label"],
            "mol_encoder": info["mol_encoder"],
            "checkpoint": str(info["checkpoint_path"]),
            "config": str(info["config_path"]),
            "default": info["id"] == normalize_model_id(DEFAULT_MODEL_ID) if MODEL_REGISTRY else False,
        }
        for info in MODEL_REGISTRY.values()
    ]


@app.get("/libraries")
@app.get("/api/libraries")
async def list_libraries():
    return list_compound_libraries()


@app.post("/predict")
@app.post("/api/predict")
async def predict(request: PredictionRequest):
    """Single RNA-compound pair prediction with full NoisyOR analysis."""
    bundle = get_model_bundle(request.model_id)
    job_id = create_job({
        "type": "single",
        "model_id": bundle["id"],
        "sequence": request.rna_sequence[:50] + ("..." if len(request.rna_sequence) > 50 else ""),
        "smiles": request.smiles[:50] + ("..." if len(request.smiles) > 50 else ""),
    })

    try:
        result = predict_single(
            model=bundle["model"],
            rna_featurizer=bundle["rna_featurizer"],
            mol_featurizer=bundle["mol_featurizer"],
            sequence=request.rna_sequence,
            smiles=request.smiles,
            device=DEVICE,
            max_len=bundle["max_len"],
        )
        result["job_id"] = job_id
        result["model_id"] = bundle["id"]
        result["mol_encoder"] = bundle["mol_encoder"]
        complete_job(job_id, result)
        return result

    except Exception as e:
        complete_job(job_id, {"error": str(e)}, status="failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
@app.post("/api/predict_batch")
async def predict_batch_endpoint(file: UploadFile = File(...), model_id: str = Form(DEFAULT_MODEL_ID)):
    """Batch prediction from CSV file upload."""
    bundle = get_model_bundle(model_id)
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")

    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        text = content.decode("latin-1")

    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames or []

    if "sequence" not in fieldnames or "smiles" not in fieldnames:
        raise HTTPException(
            status_code=400,
            detail=f"CSV must contain 'sequence' and 'smiles' columns. Found: {fieldnames}"
        )

    pairs = []
    errors = []
    for i, row in enumerate(reader):
        seq_raw = row.get("sequence", "").strip()
        smi_raw = row.get("smiles", "").strip()
        try:
            seq = validate_rna(seq_raw)
            smi = validate_smiles(smi_raw)
            pairs.append((seq, smi))
        except ValueError as e:
            errors.append({
                "sequence": seq_raw[:50],
                "smiles": smi_raw[:50],
                "error": f"Row {i+2}: {str(e)}"
            })

    if not pairs and errors:
        raise HTTPException(
            status_code=400,
            detail=f"All {len(errors)} rows failed validation. First error: {errors[0]['error']}"
        )

    job_id = create_job({
        "type": "batch",
        "model_id": bundle["id"],
        "n_pairs": len(pairs),
        "n_validation_errors": len(errors),
        "filename": file.filename,
    })

    try:
        results = predict_batch(
            model=bundle["model"],
            rna_featurizer=bundle["rna_featurizer"],
            mol_featurizer=bundle["mol_featurizer"],
            pairs=pairs,
            device=DEVICE,
            max_len=bundle["max_len"],
        )
        # Append any validation errors
        all_results = results + errors
        for r in all_results:
            r["job_id"] = job_id
            r["model_id"] = bundle["id"]
            r["mol_encoder"] = bundle["mol_encoder"]

        complete_job(job_id, {"n_predictions": len(results), "n_errors": len(errors)})
        return all_results

    except Exception as e:
        complete_job(job_id, {"error": str(e)}, status="failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/screen")
@app.post("/api/screen")
async def screen_library(request: ScreeningRequest):
    """Rank a compound library against one RNA sequence."""
    bundle = get_model_bundle(request.model_id)
    library_payload = load_library_compounds(request.library_id, request.max_candidates)
    compounds = library_payload["compounds"]
    if not compounds:
        raise HTTPException(status_code=400, detail=f"Library '{request.library_id}' has no usable SMILES")

    job_id = create_job({
        "type": "screening",
        "model_id": bundle["id"],
        "library_id": request.library_id,
        "n_candidates": len(compounds),
        "top_k": request.top_k,
    })

    try:
        ranked = score_compounds(
            bundle=bundle,
            sequence=request.rna_sequence,
            compounds=compounds,
            top_k=request.top_k,
            batch_size=64,
        )
        result = {
            "job_id": job_id,
            "model_id": bundle["id"],
            "mol_encoder": bundle["mol_encoder"],
            "library_id": request.library_id,
            "n_screened": len(compounds),
            "n_skipped": library_payload["skipped"],
            "top_k": request.top_k,
            "candidates": ranked,
        }
        complete_job(job_id, result)
        return result
    except Exception as e:
        complete_job(job_id, {"error": str(e)}, status="failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs")
@app.get("/api/jobs")
async def list_jobs(limit: int = 50):
    """List recent jobs (most recent first)."""
    sorted_jobs = sorted(
        jobs_store.values(),
        key=lambda j: j["created_at"],
        reverse=True,
    )[:limit]
    # Return summary without heavy result data
    return [
        {
            "id": j["id"],
            "type": j["type"],
            "status": j["status"],
            "input_summary": j["input_summary"],
            "created_at": j["created_at"],
            "completed_at": j["completed_at"],
        }
        for j in sorted_jobs
    ]


@app.get("/jobs/{job_id}")
@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    """Get full job details including results."""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return jobs_store[job_id]


@app.post("/validate")
@app.post("/api/validate")
async def validate_input(request: PredictionRequest):
    """Validate RNA sequence and SMILES without running prediction."""
    return {"valid": True, "rna_length": len(request.rna_sequence), "smiles": request.smiles}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

