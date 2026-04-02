from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
from datetime import datetime
import os, sys, joblib, traceback

# ── path setup ────────────────────────────────────────────
current_file  = os.path.abspath(__file__)
api_dir       = os.path.dirname(current_file)
src_dir       = os.path.dirname(api_dir)
project_root  = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.features.feature_engineer import FeatureEngineer
from src.models.ai_recommender import AIRecommender

# ── globals ───────────────────────────────────────────────
model: Optional[AIRecommender] = None
feature_engineer: Optional[FeatureEngineer] = None


# ── lifespan ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, feature_engineer

    model_path = os.path.join(project_root, "data", "models",  "ai_hybrid_recommender_latest.pkl")
    fe_path    = os.path.join(project_root, "data", "processed", "feature_engineer.pkl")

    print(f"Loading model  : {model_path}  [exists={os.path.exists(model_path)}]")
    print(f"Loading FE     : {fe_path}     [exists={os.path.exists(fe_path)}]")

    try:
        model = AIRecommender.load(model_path)
    except Exception:
        print("Could not load ML model — service will use popularity fallback.")
        traceback.print_exc()

    try:
        feature_engineer = FeatureEngineer.load(fe_path)
    except Exception:
        print("Could not load feature engineer.")
        traceback.print_exc()

    print(f"\nStatus — model={model is not None}  FE={feature_engineer is not None}\n")
    yield
    print("Shutting down Fundora ML service…")


# ── app ───────────────────────────────────────────────────
app = FastAPI(
    title="Fundora ML + AI Service",
    description="Hybrid ML recommendations powered by Random Forest + Gemini AI explanations",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── request / response schemas ────────────────────────────
class RecommendationRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    user_id: int
    n_recommendations: int = 10
    exclude_viewed: bool = True


class StartupCard(BaseModel):
    startup_id: int
    score: float
    predicted_engagement: int
    company_name: str
    industry: str
    revenue_growth: Optional[float] = None
    expected_return: Optional[float] = None
    ai_explanation: Optional[str] = None


class RecommendationResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    user_id: int
    recommendations: List[StartupCard]
    model_version: str
    timestamp: str


class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    status: str
    model_loaded: bool
    feature_engineer_loaded: bool
    gemini_enabled: bool
    timestamp: str


class ExplainRequest(BaseModel):
    user_id: int
    startup_id: int


# ── helpers ───────────────────────────────────────────────
def _popular_startups(fe: FeatureEngineer, n: int = 10) -> List[StartupCard]:
    df = fe.startup_features.copy()
    df["popularity_score"] = (
        df["view_count"].fillna(0) * 0.4
        + df["unique_viewers"].fillna(0) * 0.3
        + df["avg_engagement"].fillna(0) * 0.3
    )
    df = df.sort_values("popularity_score", ascending=False)
    cards = []
    for _, row in df.head(n).iterrows():
        cards.append(StartupCard(
            startup_id=int(row["id"]),
            score=float(row["popularity_score"]),
            predicted_engagement=1,
            company_name=str(row["company_name"]),
            industry=str(row.get("industry", "Unknown")),
            revenue_growth=float(row.get("revenue_growth", 0) or 0) * 100,
            expected_return=float(row.get("expected_return", 0) or 0),
            ai_explanation=None,
        ))
    return cards


# ── routes ────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "service": "Fundora ML + AI Service v2",
        "endpoints": {
            "POST /api/recommendations": "Personalized recommendations",
            "POST /api/explain":         "AI explanation for one startup",
            "GET  /api/popular":         "Popular startups (cold start)",
            "GET  /health":              "Service health check",
        },
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    gemini_ok = False
    return HealthResponse(
        status="healthy" if (model and feature_engineer) else "degraded",
        model_loaded=model is not None,
        feature_engineer_loaded=feature_engineer is not None,
        gemini_enabled=False,
        timestamp=datetime.now().isoformat(),
    )


@app.post("/api/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    if not feature_engineer:
        raise HTTPException(503, "Feature engineer not loaded.")

    try:
        user_known = request.user_id in feature_engineer.user_preferences["user_id"].values

        if not model or not user_known:
            cards = _popular_startups(feature_engineer, n=request.n_recommendations)
        else:
            df = model.recommend(
                user_id=request.user_id,
                feature_engineer=feature_engineer,
                n_recommendations=request.n_recommendations,
                exclude_viewed=request.exclude_viewed,
            )

            cards = []
            for _, row in df.iterrows():
                startup = feature_engineer.startup_features[
                    feature_engineer.startup_features["id"] == row["startup_id"]
                ].iloc[0]

                cards.append(StartupCard(
                    startup_id=int(row["startup_id"]),
                    score=float(row["score"]),
                    predicted_engagement=int(row["predicted_engagement"]),
                    company_name=str(startup["company_name"]),
                    industry=str(startup.get("industry", "Unknown")),
                    revenue_growth=round(float(startup.get("revenue_growth", 0) or 0) * 100, 1),
                    expected_return=round(float(startup.get("expected_return", 0) or 0), 1),
                    ai_explanation=row.get("ai_explanation"),
                ))

        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=cards,
            model_version="v2.0-gemini",
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"Recommendation error: {e}")


@app.post("/api/explain")
async def explain_startup(request: ExplainRequest):
    """Get a Gemini AI explanation for a specific user–startup pair."""
    if not model or not feature_engineer:
        raise HTTPException(503, "Model not loaded.")

    try:
        result = model.explain_recommendation(
            request.user_id, request.startup_id, feature_engineer
        )
        return {"success": True, "data": result, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(500, f"Explanation error: {e}")


@app.get("/api/popular")
async def popular_startups(n: int = Query(10, ge=1, le=50)):
    if not feature_engineer:
        raise HTTPException(503, "Feature engineer not loaded.")
    try:
        cards = _popular_startups(feature_engineer, n=n)
        return {"popular_startups": [c.dict() for c in cards], "count": len(cards),
                "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(500, f"Error: {e}")


@app.post("/api/feedback")
async def record_feedback(user_id: int, startup_id: int, action: str):
    """Record user feedback for future retraining. Actions: view | compare | watchlist"""
    level_map = {"view": 1, "compare": 2, "watchlist": 3}
    return {
        "message": "Feedback recorded",
        "user_id": user_id,
        "startup_id": startup_id,
        "action": action,
        "engagement_level": level_map.get(action.lower(), 1),
        "timestamp": datetime.now().isoformat(),
    }


# ── entry point ───────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", 8001))
    print(f"\n{'='*60}\nFundora ML Service v2  →  http://localhost:{port}\nDocs  →  http://localhost:{port}/docs\n{'='*60}\n")
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)