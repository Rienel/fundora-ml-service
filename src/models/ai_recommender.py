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
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
)


class AIRecommender:
    """
    Hybrid recommendation engine:
      - 40% Random Forest ML (engagement prediction)
      - 30% Collaborative Filtering (similar investors)
      - 30% Content-Based (TF-IDF startup similarity)

    FIXES APPLIED:
      1. Cold start in recommend() now uses proper popularity fallback
         (was using ML scoring for unknown users — now correctly routes
         to popularity-based ranking like _cold_start_recommendations())
      2. Popularity score uses unique_viewers as dominant factor (50%)
         to prevent one obsessed investor from inflating a startup's score
         (User 18 had 24/29 views for Foodly — skewing rankings)
    """

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        self.text_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.feature_columns = None
        self.is_trained = False
        self.startup_embeddings = {}
        self.user_similarity_matrix = None
        self.user_ids = []

    # ── text helpers ──────────────────────────────────────

    def encode_startup_text(self, startup_row):
        text_parts = []
        for field in ('company_name', 'industry', 'tagline', 'description'):
            val = startup_row.get(field)
            if val and pd.notna(val):
                text_parts.append(str(val))
        text = " ".join(text_parts)
        return text if text.strip() else None

    # ── popularity score ──────────────────────────────────

    def _popularity_score(self, row) -> float:
        """
        unique_viewers is the dominant factor (50%) to prevent
        one obsessed investor from inflating view_count.
        log1p(view_count) is used at 20% weight only.
        avg_engagement at 30% rewards genuine interest.
        """
        unique    = row.get('unique_viewers', 0) or 0
        avg_eng   = row.get('avg_engagement', 0) or 0
        log_views = np.log1p(row.get('view_count', 0) or 0)
        return (unique * 0.5) + (avg_eng * 0.3) + (log_views * 0.2)

    # ── cold start popularity fallback ────────────────────

    def _cold_start_recommendations(self, feature_engineer, n=10) -> pd.DataFrame:
        """
        For unknown users — returns popularity-based ranking.
        Uses _popularity_score() consistently.
        """
        df = feature_engineer.startup_features.copy()
        df['popularity_score'] = df.apply(self._popularity_score, axis=1)
        df = df.sort_values('popularity_score', ascending=False).head(n)

        rows = []
        for _, row in df.iterrows():
            rows.append({
                'startup_id':           int(row['id']),
                'score':                round(float(row['popularity_score']), 6),
                'ml_score':             0.0,
                'collaborative_score':  0.0,
                'content_score':        0.0,
                'predicted_engagement': 1,
                'fallback':             'popularity',
            })
        return pd.DataFrame(rows)

    # ── embedding layer ───────────────────────────────────

    def build_startup_embeddings(self, feature_engineer):
        print("Building startup text embeddings (using TF-IDF)...")
        texts, startup_ids = [], []

        for _, startup in feature_engineer.startup_features.iterrows():
            text = self.encode_startup_text(startup)
            if text:
                texts.append(text)
                startup_ids.append(startup['id'])

        if not texts:
            print("No text data found for embeddings")
            return

        try:
            embeddings_matrix = self.text_vectorizer.fit_transform(texts).toarray()
            for startup_id, embedding in zip(startup_ids, embeddings_matrix):
                self.startup_embeddings[startup_id] = embedding
            print(f"Built TF-IDF embeddings for {len(self.startup_embeddings)} startups")
        except Exception as e:
            print(f"Error building embeddings: {e}")

    # ── collaborative filtering ───────────────────────────

    def compute_user_similarity_matrix(self, feature_engineer):
        print("Computing user similarity matrix...")
        interactions = feature_engineer.interactions_df
        user_item_matrix = interactions.pivot_table(
            index='user_id',
            columns='startup_id',
            values='engagement_level',
            fill_value=0
        )
        if len(user_item_matrix) > 1:
            self.user_similarity_matrix = cosine_similarity(user_item_matrix)
            self.user_ids = user_item_matrix.index.tolist()
            print(f"Computed similarity for {len(self.user_ids)} users")
        else:
            print("Not enough users for collaborative filtering")
            self.user_similarity_matrix = None

    def get_similar_users(self, user_id, n=5):
        if self.user_similarity_matrix is None or user_id not in self.user_ids:
            return []
        try:
            user_idx = self.user_ids.index(user_id)
            similarities = self.user_similarity_matrix[user_idx]
            similar_indices = np.argsort(similarities)[::-1][1:n + 1]
            return [self.user_ids[i] for i in similar_indices]
        except (ValueError, IndexError):
            return []

    # ── scoring methods ───────────────────────────────────

    def collaborative_score(self, user_id, startup_id, feature_engineer):
        similar_users = self.get_similar_users(user_id, n=10)
        if not similar_users:
            return 0.0
        interactions = feature_engineer.interactions_df
        similar_interactions = interactions[
            (interactions['user_id'].isin(similar_users)) &
            (interactions['startup_id'] == startup_id)
        ]
        if len(similar_interactions) == 0:
            return 0.0
        return similar_interactions['engagement_level'].mean() / 3.0

    def content_based_score(self, user_id, startup_id, feature_engineer):
        if not self.startup_embeddings:
            return 0.0
        user_interactions = feature_engineer.interactions_df[
            (feature_engineer.interactions_df['user_id'] == user_id) &
            (feature_engineer.interactions_df['engagement_level'] >= 2)
        ]
        if len(user_interactions) == 0:
            return 0.0
        liked_startup_ids = user_interactions['startup_id'].tolist()
        liked_embeddings = [
            self.startup_embeddings[sid]
            for sid in liked_startup_ids
            if sid in self.startup_embeddings
        ]
        if not liked_embeddings or startup_id not in self.startup_embeddings:
            return 0.0
        avg_liked_embedding = np.mean(liked_embeddings, axis=0)
        candidate_embedding = self.startup_embeddings[startup_id]
        similarity = cosine_similarity(
            avg_liked_embedding.reshape(1, -1),
            candidate_embedding.reshape(1, -1)
        )[0][0]
        return max(0, similarity)

    def ml_prediction_score(self, user_id, startup_id, feature_engineer):
        if not self.is_trained:
            return 0.0
        startup = feature_engineer.startup_features[
            feature_engineer.startup_features['id'] == startup_id
        ]
        if len(startup) == 0:
            return 0.0
        startup = startup.iloc[0]
        user_pref = feature_engineer.user_preferences[
            feature_engineer.user_preferences['user_id'] == user_id
        ]
        user_pref = user_pref.iloc[0].to_dict() if len(user_pref) > 0 else {
            'total_views': 0, 'avg_engagement': 0
        }
        features = {
            'revenue':             startup.get('revenue', 0),
            'net_income':          startup.get('net_income', 0),
            'profit_margin':       startup.get('profit_margin', 0),
            'revenue_growth':      startup.get('revenue_growth', 0),
            'expected_return':     startup.get('expected_return', 0),
            'current_ratio':       startup.get('current_ratio', 0),
            'debt_to_assets':      startup.get('debt_to_assets', 0),
            'confidence_score':    startup.get('confidence_score', 0),
            'is_deck_builder':     startup.get('is_deck_builder', 0),
            'view_count':          startup.get('view_count', 0),
            'unique_viewers':      startup.get('unique_viewers', 0),
            'avg_engagement':      startup.get('avg_engagement', 0),
            'industry_encoded':    startup.get('industry_encoded', 0),
            'user_total_views':    user_pref.get('total_views', 0),
            'user_avg_engagement': user_pref.get('avg_engagement', 0),
        }
        X = np.array([features[col] for col in self.feature_columns]).reshape(1, -1)
        proba = self.model.predict_proba(X)[0]
        score = proba[0] * 1 + proba[1] * 2 + proba[2] * 3
        return score / 3.0

    def hybrid_score(self, user_id, startup_id, feature_engineer, weights=None):
        if weights is None:
            weights = {'ml': 0.4, 'collaborative': 0.3, 'content': 0.3}
        ml_score      = self.ml_prediction_score(user_id, startup_id, feature_engineer)
        collab_score  = self.collaborative_score(user_id, startup_id, feature_engineer)
        content_score = self.content_based_score(user_id, startup_id, feature_engineer)
        final_score = (
            weights['ml']            * ml_score +
            weights['collaborative'] * collab_score +
            weights['content']       * content_score
        )
        return {
            'total_score':         final_score,
            'ml_score':            ml_score,
            'collaborative_score': collab_score,
            'content_score':       content_score
        }

    # ── training ──────────────────────────────────────────

    def train(self, X, y, feature_columns, feature_engineer):
        print("\n" + "=" * 60)
        print("TRAINING AI HYBRID RECOMMENDER")
        print("=" * 60)

        self.feature_columns = feature_columns

        print("\n1. Training ML Model...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.model.fit(X_train, y_train)
        y_pred   = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"   ML Model Accuracy: {accuracy:.2%}")

        present_labels = sorted(list(set(y_val)))
        label_names    = {1: 'View', 2: 'Compare', 3: 'Watchlist'}
        present_names  = [label_names[l] for l in present_labels]
        print(classification_report(y_val, y_pred,
                                    labels=present_labels,
                                    target_names=present_names))

        cm = confusion_matrix(y_val, y_pred, labels=present_labels)
        print("Confusion Matrix (Rows=Actual, Cols=Predicted):", present_names)
        print(cm)

        feat_imp = pd.DataFrame({
            'feature':    self.feature_columns,
            'importance': self.model.feature_importances_,
        }).sort_values('importance', ascending=False)
        print("\nTop 10 Feature Importances:")
        print(feat_imp.head(10).to_string(index=False))

        print("\n2. Building NLP Components...")
        self.build_startup_embeddings(feature_engineer)

        print("\n3. Building Collaborative Filtering...")
        self.compute_user_similarity_matrix(feature_engineer)

        self.is_trained = True
        print("\n" + "=" * 60)
        print("HYBRID AI TRAINING COMPLETE")
        print("=" * 60)
        return self

    # ── recommend ─────────────────────────────────────────

    def recommend(self, user_id, feature_engineer, n_recommendations=10,
                  exclude_viewed=True, weights=None, include_explanations=False):

        # cold start check — unknown user routes to popularity fallback
        user_known = (
            self.is_trained
            and user_id in feature_engineer.user_preferences['user_id'].values
            and user_id in self.user_ids
        )

        if not user_known:
            print(f"  [Cold Start] User {user_id} unknown — using popularity fallback.")
            return self._cold_start_recommendations(feature_engineer, n=n_recommendations)

        # Known user — run full hybrid scoring
        all_startup_ids = feature_engineer.startup_features['id'].tolist()

        if exclude_viewed:
            viewed_ids = feature_engineer.interactions_df[
                feature_engineer.interactions_df['user_id'] == user_id
            ]['startup_id'].tolist()
            candidate_ids = [sid for sid in all_startup_ids if sid not in viewed_ids]
        else:
            candidate_ids = all_startup_ids

        if len(candidate_ids) == 0:
            candidate_ids = all_startup_ids

        recommendations = []
        for startup_id in candidate_ids:
            scores = self.hybrid_score(user_id, startup_id, feature_engineer, weights)

            if self.is_trained:
                startup = feature_engineer.startup_features[
                    feature_engineer.startup_features['id'] == startup_id
                ].iloc[0]
                user_pref = feature_engineer.user_preferences[
                    feature_engineer.user_preferences['user_id'] == user_id
                ]
                user_pref = user_pref.iloc[0].to_dict() if len(user_pref) > 0 else {
                    'total_views': 0, 'avg_engagement': 0
                }
                features = {
                    'revenue':             startup.get('revenue', 0),
                    'net_income':          startup.get('net_income', 0),
                    'profit_margin':       startup.get('profit_margin', 0),
                    'revenue_growth':      startup.get('revenue_growth', 0),
                    'expected_return':     startup.get('expected_return', 0),
                    'current_ratio':       startup.get('current_ratio', 0),
                    'debt_to_assets':      startup.get('debt_to_assets', 0),
                    'confidence_score':    startup.get('confidence_score', 0),
                    'is_deck_builder':     startup.get('is_deck_builder', 0),
                    'view_count':          startup.get('view_count', 0),
                    'unique_viewers':      startup.get('unique_viewers', 0),
                    'avg_engagement':      startup.get('avg_engagement', 0),
                    'industry_encoded':    startup.get('industry_encoded', 0),
                    'user_total_views':    user_pref.get('total_views', 0),
                    'user_avg_engagement': user_pref.get('avg_engagement', 0),
                }
                X = np.array([features[col] for col in self.feature_columns]).reshape(1, -1)
                predicted_engagement = int(self.model.predict(X)[0])
            else:
                predicted_engagement = 1

            recommendations.append({
                'startup_id':           startup_id,
                'score':                scores['total_score'],
                'ml_score':             scores['ml_score'],
                'collaborative_score':  scores['collaborative_score'],
                'content_score':        scores['content_score'],
                'predicted_engagement': predicted_engagement,
                'fallback':             'hybrid',
            })

        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df = recommendations_df.sort_values('score', ascending=False)
        return recommendations_df.head(n_recommendations)

    # ── explain ───────────────────────────────────────────

    def explain_recommendation(self, user_id, startup_id, feature_engineer):
        scores  = self.hybrid_score(user_id, startup_id, feature_engineer)
        startup = feature_engineer.startup_features[
            feature_engineer.startup_features['id'] == startup_id
        ].iloc[0]

        explanation = {
            'startup_name':  startup['company_name'],
            'overall_score': scores['total_score'],
            'reasons':       []
        }
        if scores['ml_score'] > 0.5:
            explanation['reasons'].append(
                f"Strong match based on your behavior patterns (ML Score: {scores['ml_score']:.2f})"
            )
        if scores['collaborative_score'] > 0.3:
            explanation['reasons'].append(
                f"Popular among users with similar interests (Collab Score: {scores['collaborative_score']:.2f})"
            )
        if scores['content_score'] > 0.5:
            explanation['reasons'].append(
                f"Similar to startups you previously engaged with (Content Score: {scores['content_score']:.2f})"
            )
        if startup.get('revenue_growth', 0) > 50:
            explanation['reasons'].append(
                f"High revenue growth: {startup['revenue_growth']:.1f}%"
            )
        if startup.get('expected_return', 0) > 20:
            explanation['reasons'].append(
                f"Strong expected return: {startup['expected_return']:.1f}%"
            )
        user_pref = feature_engineer.user_preferences[
            feature_engineer.user_preferences['user_id'] == user_id
        ]
        if len(user_pref) > 0:
            preferred_industry = user_pref.iloc[0].get('preferred_industry')
            if preferred_industry and startup.get('industry') == preferred_industry:
                explanation['reasons'].append(
                    f"Matches your preferred industry: {preferred_industry}"
                )
        return explanation

    # ── persistence ───────────────────────────────────────

    def save(self, path='data/models/ai_hybrid_recommender_latest.pkl'):
        if not self.is_trained:
            raise Exception("Cannot save untrained model!")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        model_data = {
            'model':                  self.model,
            'feature_columns':        self.feature_columns,
            'is_trained':             self.is_trained,
            'startup_embeddings':     self.startup_embeddings,
            'user_similarity_matrix': self.user_similarity_matrix,
            'user_ids':               getattr(self, 'user_ids', None),
            'trained_at':             datetime.now().isoformat(),
            'model_type':             'AIRecommender_Fixed'
        }
        joblib.dump(model_data, path)
        print(f"AI Hybrid Model saved to {path}")
        timestamp      = datetime.now().strftime('%Y%m%d_%H%M%S')
        versioned_path = path.replace('_latest', f'_{timestamp}')
        joblib.dump(model_data, versioned_path)
        print(f"Version saved to {versioned_path}")

    @staticmethod
    def load(path='data/models/ai_hybrid_recommender_latest.pkl'):
        model_data = joblib.load(path)
        recommender = AIRecommender()
        recommender.model                  = model_data['model']
        recommender.feature_columns        = model_data['feature_columns']
        recommender.is_trained             = model_data['is_trained']
        recommender.startup_embeddings     = model_data['startup_embeddings']
        recommender.user_similarity_matrix = model_data['user_similarity_matrix']
        recommender.user_ids               = model_data.get('user_ids', [])
        print(f"AI Hybrid Model loaded from {path}")
        print(f"   Trained at: {model_data.get('trained_at', 'Unknown')}")
        return recommender


# ── Training script ───────────────────────────────────────

if __name__ == "__main__":
    from src.features.feature_engineer import FeatureEngineer

    print("=" * 60)
    print("FUNDORA AI HYBRID RECOMMENDER - TRAINING")
    print("=" * 60)

    print("\n1. Loading feature engineer...")
    fe = FeatureEngineer.load('data/processed/feature_engineer.pkl')

    print("\n2. Preparing training data...")
    X, y            = fe.get_training_data()
    feature_columns = fe.feature_columns
    print(f"   Training examples: {len(X)}")
    print(f"   Features: {len(feature_columns)}")

    print("\n3. Training hybrid recommender...")
    recommender = AIRecommender()
    recommender.train(X, y, feature_columns, fe)

    print("\n4. Saving model...")
    recommender.save()

    print("\n5. Generating confusion matrix figure...")
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
    import matplotlib.pyplot as plt

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_pred = recommender.model.predict(X_val)

    present_labels = sorted(list(set(y_val)))
    label_map      = {1: 'View (1)', 2: 'Compare (2)', 3: 'Watchlist (3)'}
    present_names  = [label_map[l] for l in present_labels]

    cm   = confusion_matrix(y_val, y_pred, labels=present_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=present_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(values_format='d', cmap='Blues', ax=ax, colorbar=True)
    ax.set_title('Confusion Matrix of Engagement Prediction',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Label',      fontsize=12, fontweight='bold', labelpad=10)
    plt.tight_layout()
    plt.savefig('fig_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   Saved fig_confusion_matrix.png")

    print("\nDone! Model is trained, saved, and ready.")