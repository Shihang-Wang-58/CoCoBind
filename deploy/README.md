# CoCoBind Deployment Guide

## Prerequisites
- Python 3.9+ with CUDA support is recommended
- Node.js for Frontend
- Pre-trained model checkpoint (place in `outputs/best_model.pt` or update `deploy/backend/main.py`)

## 1. Backend Setup

1. **Install Python dependencies:**
```bash
pip install fastapi uvicorn pydantic torch numpy transformers multimolecule rdkit
```
*(Make sure CoCoBind dependencies are also installed)*

2. **Run the FastAPI server:**
Navigate to the root project directory:
```bash
uvicorn deploy.backend.main:app --host 0.0.0.0 --port 8000 --reload
```
The backend API will be available at `http://localhost:8000`.

## 2. Frontend Setup

1. **Install Node.js dependencies:**
Navigate to `deploy/frontend`:
```bash
cd deploy/frontend
npm install
```

2. **Start the React application:**
```bash
npm start
```
The web interface will open at `http://localhost:3000`.

## Usage
1. Open the web interface.
2. Enter an RNA sequence (e.g., standard ACGU string).
3. Enter a Small Molecule SMILES string.
4. Click "Run Prediction" to see interaction probability and per-residue binding sites.
