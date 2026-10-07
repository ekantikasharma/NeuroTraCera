# XAI Brain Tumor Backend

FastAPI backend for an academic XAI brain-tumor MRI detection and localization prototype.

## Structure

- `server.py` — FastAPI API and frontend connectivity
- `analysis.py` — request/response schemas
- `model.py` — PyTorch ResNet50 inference
- `gradcam.py` — Grad-CAM heatmap and activation-region localization
- `db.py` — optional MongoDB persistence
- `sampledata.py` — synthetic demo MRI samples
- `evaluation.py` — classification metrics and uncertainty helpers
- `dashboard.py` — dashboard aggregation
- `seed.py` — seed synthetic sample metadata
- `models/brain_tumor_model.pth` — put your trained compatible model here

## Run

1. Create a virtual environment.
2. Install `requirements.txt`.
3. Copy `.env.example` to `.env`.
4. Put a compatible trained model at `models/brain_tumor_model.pth`.
5. Start:

`python -m uvicorn server:app --reload --host 127.0.0.1 --port 8000`

Swagger docs: `http://127.0.0.1:8000/docs`

React frontend should call:
`http://127.0.0.1:8000/api/analyze`

## Important

The default model architecture is a standard torchvision ResNet50 with a 4-class output. If your existing `.pth` was trained with a custom architecture, its architecture must match `model.py` before loading the checkpoint.

This is an academic/research prototype and is not a medical diagnostic system.
