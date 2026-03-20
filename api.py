"""
FastAPI backend for the ABSA pipeline.

Endpoints:
    POST /analyze  — analyze a review text, returns aspects + summary
    GET  /health   — health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from absa.pipeline import ABSAPipeline

app = FastAPI(title="ABSA API")

# Allow React dev server (port 5173) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pipeline once at startup
pipeline = ABSAPipeline("models/best_model.pt", "config.yaml")


class AnalyzeRequest(BaseModel):
    review: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not req.review.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty")
    result = pipeline.run(req.review)
    return result
