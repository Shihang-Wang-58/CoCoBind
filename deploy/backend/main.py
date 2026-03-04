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
import re
import uuid
from datetime import datetime
from typing import Optional

import torch
import yaml
from fastapi import FastAPI, HTTPException, UploadFile, File
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Model & Featurizer initialization
# ============================================================================
CONFIG_PATH = _PROJECT_DIR / "cocobind" / "config.yaml"
CHECKPOINT_PATH = _PROJECT_DIR / "outputs" / "base" / "unseen_pair" / "fold0" / "best_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Featurizers
cache_dir = config.get("data", {}).get("cache_dir", "cache")
if not Path(cache_dir).is_absolute():
    cache_dir = str(_PROJECT_DIR / cache_dir)

rna_featurizer = RNAFMFeaturizer(cache_dir=cache_dir, device=str(DEVICE))
mol_featurizer = get_mol_featurizer(mol_encoder="ecfp4", n_bits=2048)

# Model
model: Optional[RNADTModel] = None
if CHECKPOINT_PATH.exists():
    model, _ = load_model(
        str(CHECKPOINT_PATH),
        str(CONFIG_PATH),
        DEVICE,
    )
    print(f"✓ Model loaded from {CHECKPOINT_PATH}")
else:
    # Fallback: create model with random weights for development
    model_cfg = config.get("model", {})
    model = RNADTModel(
        d_rna=640,
        d_mol=2048,
        d_model=model_cfg.get("d_model", 256),
        n_mol_tokens=model_cfg.get("n_mol_tokens", 8),
        n_heads=model_cfg.get("n_heads", 8),
        dropout=model_cfg.get("dropout", 0.1),
        use_cross_attn=model_cfg.get("use_cross_attn", True),
    ).to(DEVICE).eval()
    print(f"⚠ Checkpoint not found at {CHECKPOINT_PATH}. Using random weights.")

MAX_LEN = config.get("data", {}).get("max_len", 512)

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
    if len(seq) > MAX_LEN:
        raise ValueError(f"RNA sequence too long ({len(seq)} nt). Maximum is {MAX_LEN}.")
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

    @field_validator("rna_sequence")
    @classmethod
    def check_rna(cls, v):
        return validate_rna(v)

    @field_validator("smiles")
    @classmethod
    def check_smiles(cls, v):
        return validate_smiles(v)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def root():
    return {"message": "CoCoBind API is running", "device": str(DEVICE)}


@app.post("/predict")
async def predict(request: PredictionRequest):
    """Single RNA-compound pair prediction with full NoisyOR analysis."""
    job_id = create_job({
        "type": "single",
        "sequence": request.rna_sequence[:50] + ("..." if len(request.rna_sequence) > 50 else ""),
        "smiles": request.smiles[:50] + ("..." if len(request.smiles) > 50 else ""),
    })

    try:
        result = predict_single(
            model=model,
            rna_featurizer=rna_featurizer,
            mol_featurizer=mol_featurizer,
            sequence=request.rna_sequence,
            smiles=request.smiles,
            device=DEVICE,
            max_len=MAX_LEN,
        )
        result["job_id"] = job_id
        complete_job(job_id, result)
        return result

    except Exception as e:
        complete_job(job_id, {"error": str(e)}, status="failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
async def predict_batch_endpoint(file: UploadFile = File(...)):
    """Batch prediction from CSV file upload."""
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
        "n_pairs": len(pairs),
        "n_validation_errors": len(errors),
        "filename": file.filename,
    })

    try:
        results = predict_batch(
            model=model,
            rna_featurizer=rna_featurizer,
            mol_featurizer=mol_featurizer,
            pairs=pairs,
            device=DEVICE,
            max_len=MAX_LEN,
        )
        # Append any validation errors
        all_results = results + errors
        for r in all_results:
            r["job_id"] = job_id

        complete_job(job_id, {"n_predictions": len(results), "n_errors": len(errors)})
        return all_results

    except Exception as e:
        complete_job(job_id, {"error": str(e)}, status="failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs")
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
async def get_job(job_id: str):
    """Get full job details including results."""
    if job_id not in jobs_store:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return jobs_store[job_id]


@app.post("/validate")
async def validate_input(request: PredictionRequest):
    """Validate RNA sequence and SMILES without running prediction."""
    return {"valid": True, "rna_length": len(request.rna_sequence), "smiles": request.smiles}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

