from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
import joblib
import os
from datetime import datetime
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.features.feature_engineer import FeatureEngineer
from src.models.recommender import StartupRecommender

# Global variables for model and feature engineer
model = None
feature_engineer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    global model, feature_engineer
    
    # Startup
    try:
        print("Loading ML model...")
        model = StartupRecommender.load('data/models/recommender_latest.pkl')
        print("Model loaded successfully")
        
        print("Loading feature engineer...")
        feature_engineer = FeatureEngineer.load('data/processed/feature_engineer.pkl')
        print("Feature engineer loaded successfully")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("API will run in degraded mode")
    
    yield  # Server runs here
    
    # Shutdown (cleanup if needed)
    print("Shutting down ML service...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Fundora ML Recommendation Service",
    description="AI-powered startup recommendations for Fundora",
    version="1.0.0",
    lifespan=lifespan
)

# Models for API requests/responses
class RecommendationRequest(BaseModel):
    user_id: int
    n_recommendations: int = 10
    exclude_viewed: bool = True

class RecommendationResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    user_id: int
    recommendations: List[Dict]
    model_version: str
    timestamp: str

class HealthResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    
    status: str
    model_loaded: bool
    feature_engineer_loaded: bool
    timestamp: str

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Fundora ML Recommendation Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "recommendations": "/api/recommendations (POST)",
            "popular": "/api/popular (GET)"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if (model and feature_engineer) else "degraded",
        model_loaded=model is not None,
        feature_engineer_loaded=feature_engineer is not None,
        timestamp=datetime.now().isoformat()
    )

@app.post("/api/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Generate personalized startup recommendations for a user
    
    - **user_id**: The ID of the user requesting recommendations
    - **n_recommendations**: Number of recommendations to return (default: 10)
    - **exclude_viewed**: Whether to exclude already viewed startups (default: true)
    """
    global model, feature_engineer
    
    if not model or not feature_engineer:
        raise HTTPException(
            status_code=503,
            detail="ML model not loaded. Service unavailable."
        )
    
    try:
        # Check if user exists
        user_exists = request.user_id in feature_engineer.user_preferences['user_id'].values
        
        if not user_exists:
            # Cold start - return popular startups
            print(f"User {request.user_id} not found in training data. Using popularity-based recommendations.")
            recommendations = get_popular_startups(
                feature_engineer, 
                n=request.n_recommendations
            )
        else:
            # Use ML model
            recommendations_df = model.recommend(
                user_id=request.user_id,
                feature_engineer=feature_engineer,
                n_recommendations=request.n_recommendations,
                exclude_viewed=request.exclude_viewed
            )
            
            # Convert to list of dicts
            recommendations = []
            for _, row in recommendations_df.iterrows():
                startup = feature_engineer.startup_features[
                    feature_engineer.startup_features['id'] == row['startup_id']
                ].iloc[0]
                
                recommendations.append({
                    'startup_id': int(row['startup_id']),
                    'score': float(row['score']),
                    'predicted_engagement': int(row['predicted_engagement']),
                    'company_name': startup['company_name'],
                    'industry': startup['industry']
                })
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            model_version="v1.0",
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating recommendations: {str(e)}"
        )

@app.get("/api/popular")
async def get_popular_startups_endpoint(n: int = 10):
    """
    Get popular startups based on view count and engagement
    Used as fallback for cold start users
    """
    global feature_engineer
    
    if not feature_engineer:
        raise HTTPException(
            status_code=503,
            detail="Feature engineer not loaded"
        )
    
    try:
        popular = get_popular_startups(feature_engineer, n=n)
        return {
            "popular_startups": popular,
            "count": len(popular),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching popular startups: {str(e)}"
        )

def get_popular_startups(feature_engineer, n=10):
    """Helper function to get popular startups"""
    # Sort by view count and engagement
    popular = feature_engineer.startup_features.copy()
    popular['popularity_score'] = (
        popular['view_count'] * 0.4 + 
        popular['unique_viewers'] * 0.3 + 
        popular['avg_engagement'] * 0.3
    )
    popular = popular.sort_values('popularity_score', ascending=False)
    
    recommendations = []
    for _, startup in popular.head(n).iterrows():
        recommendations.append({
            'startup_id': int(startup['id']),
            'score': float(startup['popularity_score']),
            'predicted_engagement': 1,  # Default to "view"
            'company_name': startup['company_name'],
            'industry': startup['industry']
        })
    
    return recommendations

@app.post("/api/feedback")
async def record_feedback(
    user_id: int,
    startup_id: int,
    action: str
):
    """
    Record user feedback for future model retraining
    
    Actions: 'view', 'compare', 'watchlist'
    """
    # In production, save to database latur
    
    engagement_map = {
        'view': 1,
        'compare': 2,
        'watchlist': 3
    }
    
    engagement_level = engagement_map.get(action.lower(), 1)
    
    return {
        "message": "Feedback recorded",
        "user_id": user_id,
        "startup_id": startup_id,
        "action": action,
        "engagement_level": engagement_level,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("API_PORT", 8000))
    
    print("\n" + "="*60)
    print("Starting Fundora ML Recommendation Service")
    print("="*60)
    print(f"API will be available at: http://localhost:{port}")
    print(f"API docs at: http://localhost:{port}/docs")
    print("="*60 + "\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False
    )