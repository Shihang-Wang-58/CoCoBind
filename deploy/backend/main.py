from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from cocobind.model import RNADTModel
from cocobind.featurizers import RNAFMFeaturizer, ECFP4Featurizer
import yaml
import os

app = FastAPI()

# CORS Support
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
CONFIG_PATH = "cocobind/config.yaml"
# TODO: Set path to your best checkpoint
CHECKPOINT_PATH = "outputs/base/unseen_pair/fold0/best_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load config
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Initialize Featurizers
rna_featurizer = RNAFMFeaturizer(cache_dir="cache_deploy/rna_embeddings", device=DEVICE)
mol_featurizer = ECFP4Featurizer(n_bits=2048)

# Initialize Model
model = RNADTModel(
    d_rna=640, # RNA-FM dim
    d_mol=2048, # ECFP4 dim
    d_model=config["model"]["d_model"],
    n_mol_tokens=config["model"]["n_mol_tokens"],
    n_heads=config["model"]["n_heads"],
    dropout=config["model"]["dropout"],
    use_cross_attn=config["model"].get("use_cross_attn", True),
)

# Load checkpoint
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")
else:
    print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}. Using random weights.")

model.to(DEVICE)
model.eval()

class PredictionRequest(BaseModel):
    rna_sequence: str
    smiles: str

class PredictionResponse(BaseModel):
    interaction_probability: float
    binding_sites: list[float]

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    try:
        # Featurize
        rna_embed = rna_featurizer(request.rna_sequence, config["data"]["max_len"]).to(DEVICE)
        mol_fp = mol_featurizer(request.smiles).to(DEVICE)
        
        # Batch dimension
        rna_embed = rna_embed.unsqueeze(0) # [1, L, d_rna]
        mol_fp = mol_fp.unsqueeze(0) # [1, d_mol]
        
        # Masks
        seq_len = request.rna_sequence.count("") # Rough length check, better logic needed for actual seq len handling if tokenized differently, 
        # but here we use embedding size directly.
        # Actually rna_featurizer returns [L, d], and we pad usually in dataloaders. 
        # But for single inference, we can just use it as is if model handles variable length.
        # But wait, model expects batch and mask.
        
        # Let's see model forward
        # needs: rna_embed, mol_fp, rna_mask
        
        L = rna_embed.shape[1]
        rna_mask = torch.ones(1, L).to(DEVICE)
        
        batch = {
            "rna_embed": rna_embed,
            "mol_fp": mol_fp,
            "rna_mask": rna_mask
        }
        
        with torch.no_grad():
            outputs = model(batch)
            
            # Interaction
            int_logit = outputs["interaction_logit"]
            int_prob = torch.sigmoid(int_logit).item()
            
            # Binding Sites
            site_logits = outputs["site_logits"] # [1, L]
            site_probs = torch.sigmoid(site_logits).squeeze().tolist()
            if isinstance(site_probs, float):
                site_probs = [site_probs]
                
        return PredictionResponse(
            interaction_probability=int_prob,
            binding_sites=site_probs
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
