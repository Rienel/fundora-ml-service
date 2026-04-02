import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from datetime import datetime


# ──────────────────────────────────────────────────────────
# Main AI Recommender
# ──────────────────────────────────────────────────────────

class AIRecommender:
    """
    Hybrid recommendation engine:
      • 40 % — Random Forest ML (engagement prediction)
      • 30 % — Collaborative Filtering (similar investors)
      • 30 % — Content-Based (TF-IDF startup similarity)
    """

    def __init__(self):
        # ML model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight="balanced",
        )

        # NLP
        self.text_vectorizer = TfidfVectorizer(max_features=100, stop_words="english")

        # State
        self.feature_columns = None
        self.is_trained = False
        self.startup_embeddings: dict = {}
        self.user_similarity_matrix = None
        self.user_ids: list = []

    # ── text helpers ──────────────────────────────────────

    def _startup_text(self, row) -> str:
        parts = []
        for field in ("company_name", "industry", "tagline", "description"):
            val = row.get(field)
            if val and pd.notna(val):
                parts.append(str(val))
        return " ".join(parts)

    # ── embedding layer ───────────────────────────────────

    def build_startup_embeddings(self, feature_engineer):
        print("Building TF-IDF startup embeddings…")
        texts, ids = [], []
        for _, row in feature_engineer.startup_features.iterrows():
            t = self._startup_text(row)
            if t:
                texts.append(t)
                ids.append(row["id"])

        if not texts:
            print("No text data found — skipping embeddings.")
            return

        matrix = self.text_vectorizer.fit_transform(texts).toarray()
        for sid, emb in zip(ids, matrix):
            self.startup_embeddings[sid] = emb
        print(f"  Built embeddings for {len(self.startup_embeddings)} startups.")

    # ── collaborative filtering ───────────────────────────

    def compute_user_similarity_matrix(self, feature_engineer):
        print("Computing user similarity matrix…")
        interactions = feature_engineer.interactions_df
        mat = interactions.pivot_table(
            index="user_id",
            columns="startup_id",
            values="engagement_level",
            fill_value=0,
        )
        if len(mat) > 1:
            self.user_similarity_matrix = cosine_similarity(mat)
            self.user_ids = mat.index.tolist()
            print(f"  Similarity computed for {len(self.user_ids)} users.")
        else:
            print("  Not enough users for collaborative filtering.")

    def get_similar_users(self, user_id: int, n: int = 5) -> list:
        if self.user_similarity_matrix is None or user_id not in self.user_ids:
            return []
        idx = self.user_ids.index(user_id)
        sims = self.user_similarity_matrix[idx]
        top = np.argsort(sims)[::-1][1 : n + 1]
        return [self.user_ids[i] for i in top]

    # ── scoring methods ───────────────────────────────────

    def _collaborative_score(self, user_id, startup_id, feature_engineer) -> float:
        similar = self.get_similar_users(user_id, n=10)
        if not similar:
            return 0.0
        df = feature_engineer.interactions_df
        sub = df[(df["user_id"].isin(similar)) & (df["startup_id"] == startup_id)]
        if sub.empty:
            return 0.0
        return sub["engagement_level"].mean() / 3.0

    def _content_score(self, user_id, startup_id, feature_engineer) -> float:
        if not self.startup_embeddings or startup_id not in self.startup_embeddings:
            return 0.0
        df = feature_engineer.interactions_df
        liked = df[(df["user_id"] == user_id) & (df["engagement_level"] >= 2)][
            "startup_id"
        ].tolist()
        embs = [self.startup_embeddings[s] for s in liked if s in self.startup_embeddings]
        if not embs:
            return 0.0
        avg = np.mean(embs, axis=0)
        sim = cosine_similarity(avg.reshape(1, -1), self.startup_embeddings[startup_id].reshape(1, -1))[0][0]
        return max(0.0, float(sim))

    def _ml_score(self, user_id, startup_id, feature_engineer) -> float:
        if not self.is_trained:
            return 0.0
        startup = feature_engineer.startup_features[
            feature_engineer.startup_features["id"] == startup_id
        ]
        if startup.empty:
            return 0.0
        startup = startup.iloc[0]
        pref = feature_engineer.user_preferences[
            feature_engineer.user_preferences["user_id"] == user_id
        ]
        pref = pref.iloc[0].to_dict() if not pref.empty else {"total_views": 0, "avg_engagement": 0}

        feat = {
            "revenue": startup.get("revenue", 0),
            "net_income": startup.get("net_income", 0),
            "profit_margin": startup.get("profit_margin", 0),
            "revenue_growth": startup.get("revenue_growth", 0),
            "expected_return": startup.get("expected_return", 0),
            "current_ratio": startup.get("current_ratio", 0),
            "debt_to_assets": startup.get("debt_to_assets", 0),
            "confidence_score": startup.get("confidence_score", 0),
            "is_deck_builder": startup.get("is_deck_builder", 0),
            "view_count": startup.get("view_count", 0),
            "unique_viewers": startup.get("unique_viewers", 0),
            "avg_engagement": startup.get("avg_engagement", 0),
            "industry_encoded": startup.get("industry_encoded", 0),
            "user_total_views": pref.get("total_views", 0),
            "user_avg_engagement": pref.get("avg_engagement", 0),
        }
        X = np.array([feat[c] for c in self.feature_columns]).reshape(1, -1)
        proba = self.model.predict_proba(X)[0]
        return (proba[0] * 1 + proba[1] * 2 + proba[2] * 3) / 3.0

    def hybrid_score(self, user_id, startup_id, feature_engineer, weights=None) -> dict:
        w = weights or {"ml": 0.4, "collaborative": 0.3, "content": 0.3}
        ml  = self._ml_score(user_id, startup_id, feature_engineer)
        col = self._collaborative_score(user_id, startup_id, feature_engineer)
        con = self._content_score(user_id, startup_id, feature_engineer)
        total = w["ml"] * ml + w["collaborative"] * col + w["content"] * con
        return {"total_score": total, "ml_score": ml, "collaborative_score": col, "content_score": con}

    # ── training ──────────────────────────────────────────

    def train(self, X, y, feature_columns, feature_engineer):
        print("\n" + "=" * 60)
        print("TRAINING FUNDORA AI HYBRID RECOMMENDER")
        print("=" * 60)

        self.feature_columns = feature_columns

        # 1. ML model
        print("\n1. Training Random Forest ML model…")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model.fit(X_train, y_train)
        acc = accuracy_score(y_val, self.model.predict(X_val))
        print(f"   Validation accuracy: {acc:.2%}")
        print(classification_report(y_val, self.model.predict(X_val),
                                    target_names=["View", "Compare", "Watchlist"]))

        # 2. NLP embeddings
        print("\n2. Building NLP content layer…")
        self.build_startup_embeddings(feature_engineer)

        # 3. Collaborative filtering
        print("\n3. Building collaborative filtering layer…")
        self.compute_user_similarity_matrix(feature_engineer)

        self.is_trained = True
        print("\n" + "=" * 60)
        print("HYBRID AI TRAINING COMPLETE")
        print("=" * 60)
        return self

    # ── recommend ─────────────────────────────────────────

    def recommend(
        self,
        user_id,
        feature_engineer,
        n_recommendations: int = 10,
        exclude_viewed: bool = True,
        weights=None,
        include_explanations: bool = False,
    ) -> pd.DataFrame:
        all_ids = feature_engineer.startup_features["id"].tolist()

        if exclude_viewed:
            viewed = feature_engineer.interactions_df[
                feature_engineer.interactions_df["user_id"] == user_id
            ]["startup_id"].tolist()
            candidates = [s for s in all_ids if s not in viewed]
        else:
            candidates = all_ids

        if not candidates:
            candidates = all_ids

        rows = []
        for sid in candidates:
            s = self.hybrid_score(user_id, sid, feature_engineer, weights)

            pref = feature_engineer.user_preferences[
                feature_engineer.user_preferences["user_id"] == user_id
            ]
            pref = pref.iloc[0].to_dict() if not pref.empty else {"total_views": 0, "avg_engagement": 0}

            startup = feature_engineer.startup_features[
                feature_engineer.startup_features["id"] == sid
            ].iloc[0]

            feat = {
                "revenue": startup.get("revenue", 0),
                "net_income": startup.get("net_income", 0),
                "profit_margin": startup.get("profit_margin", 0),
                "revenue_growth": startup.get("revenue_growth", 0),
                "expected_return": startup.get("expected_return", 0),
                "current_ratio": startup.get("current_ratio", 0),
                "debt_to_assets": startup.get("debt_to_assets", 0),
                "confidence_score": startup.get("confidence_score", 0),
                "is_deck_builder": startup.get("is_deck_builder", 0),
                "view_count": startup.get("view_count", 0),
                "unique_viewers": startup.get("unique_viewers", 0),
                "avg_engagement": startup.get("avg_engagement", 0),
                "industry_encoded": startup.get("industry_encoded", 0),
                "user_total_views": pref.get("total_views", 0),
                "user_avg_engagement": pref.get("avg_engagement", 0),
            }
            X = np.array([feat[c] for c in self.feature_columns]).reshape(1, -1)
            predicted_engagement = int(self.model.predict(X)[0]) if self.is_trained else 1

            row = {
                "startup_id": sid,
                "score": s["total_score"],
                "ml_score": s["ml_score"],
                "collaborative_score": s["collaborative_score"],
                "content_score": s["content_score"],
                "predicted_engagement": predicted_engagement,
                "ai_explanation": None,
            }
            rows.append(row)

        df = pd.DataFrame(rows).sort_values("score", ascending=False).head(n_recommendations)
        return df

    # ── explain single startup ────────────────────────────

    def explain_recommendation(self, user_id, startup_id, feature_engineer) -> dict:
        scores = self.hybrid_score(user_id, startup_id, feature_engineer)
        startup = feature_engineer.startup_features[
            feature_engineer.startup_features["id"] == startup_id
        ].iloc[0]

        return {
            "startup_name": startup["company_name"],
            "overall_score": scores["total_score"],
            "score_breakdown": {
                "ml": scores["ml_score"],
                "collaborative": scores["collaborative_score"],
                "content": scores["content_score"],
            },
            "ai_explanation": None,
        }

    # ── persistence ───────────────────────────────────────

    def save(self, path: str = "data/models/ai_hybrid_recommender_latest.pkl"):
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model!")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "model": self.model,
            "feature_columns": self.feature_columns,
            "is_trained": self.is_trained,
            "startup_embeddings": self.startup_embeddings,
            "user_similarity_matrix": self.user_similarity_matrix,
            "user_ids": self.user_ids,
            "trained_at": datetime.now().isoformat(),
            "model_type": "AIRecommender_Hybrid",
        }
        joblib.dump(data, path)
        print(f"Model saved to {path}")
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        joblib.dump(data, path.replace("_latest", f"_{ts}"))

    @staticmethod
    def load(path: str = "data/models/ai_hybrid_recommender_latest.pkl"):
        data = joblib.load(path)
        r = AIRecommender()
        r.model = data["model"]
        r.feature_columns = data["feature_columns"]
        r.is_trained = data["is_trained"]
        r.startup_embeddings = data["startup_embeddings"]
        r.user_similarity_matrix = data["user_similarity_matrix"]
        r.user_ids = data.get("user_ids", [])
        print(f"Model loaded — trained at {data.get('trained_at', 'unknown')}")
        return r