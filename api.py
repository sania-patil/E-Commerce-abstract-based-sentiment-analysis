"""
FastAPI backend for the ABSA pipeline.

Endpoints:
    POST /analyze  — analyze a review text, returns aspects + summary
    GET  /health   — health check
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from absa.pipeline import ABSAPipeline
from absa.opinion_summarizer import summarize
from absa.models import AspectSentimentPair

app = FastAPI(title="ABSA API")

# Allow React dev server and production Vercel deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        os.getenv("FRONTEND_URL", ""),
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pipeline once at startup
pipeline = ABSAPipeline("models/best_model.pt", "config.yaml")


class AnalyzeRequest(BaseModel):
    review: str


class BatchAnalyzeRequest(BaseModel):
    reviews: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    if not req.review.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty")
    result = pipeline.run(req.review)
    return result


@app.post("/analyze/batch")
def analyze_batch(req: BatchAnalyzeRequest):
    reviews = [r for r in req.reviews if r.strip()]
    if not reviews:
        raise HTTPException(status_code=400, detail="No valid reviews provided")

    all_pairs = []
    per_review = []
    for review in reviews:
        result = pipeline.run(review)
        per_review.append(result)
        for a in result["aspects"]:
            all_pairs.append(AspectSentimentPair(
                aspect=a["term"],
                polarity=a["polarity"],
                confidence=a["confidence"],
                span=[0, 0],
                low_confidence=a["low_confidence"],
            ))

    agg_summary = summarize(all_pairs)

    return {
        "per_review": per_review,
        "aggregated_summary": {
            "total_reviews": len(reviews),
            "strengths": [{"aspect": s.aspect, "count": s.count} for s in agg_summary.strengths],
            "weaknesses": [{"aspect": w.aspect, "count": w.count} for w in agg_summary.weaknesses],
        },
    }
