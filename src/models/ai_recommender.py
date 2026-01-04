import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score
)


class AIRecommender:
    def __init__(self):
        # ML Model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        
        # Use TfidfVectorizer instead of transformers
        self.text_vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
        self.feature_columns = None
        self.is_trained = False
        
        # Cache for embeddings and similarities
        self.startup_embeddings = {}
        self.user_similarity_matrix = None
    
    def encode_startup_text(self, startup_row):
        """Convert startup text to vector (using TF-IDF instead of transformers)"""
        # Combine relevant text fields
        text_parts = []
        
        if 'company_name' in startup_row and pd.notna(startup_row['company_name']):
            text_parts.append(str(startup_row['company_name']))
        
        if 'industry' in startup_row and pd.notna(startup_row['industry']):
            text_parts.append(str(startup_row['industry']))
        
        if 'tagline' in startup_row and pd.notna(startup_row.get('tagline')):
            text_parts.append(str(startup_row['tagline']))
        
        text = " ".join(text_parts)
        
        if not text.strip():
            return None
        
        return text  # Return text, will be vectorized in batch
    
    def build_startup_embeddings(self, feature_engineer):
        """Pre-compute embeddings for all startups using TF-IDF"""
        print("Building startup text embeddings (using TF-IDF)...")
        
        texts = []
        startup_ids = []
        
        for idx, startup in feature_engineer.startup_features.iterrows():
            startup_id = startup['id']
            text = self.encode_startup_text(startup)
            if text:
                texts.append(text)
                startup_ids.append(startup_id)
        
        if not texts:
            print("No text data found for embeddings")
            return
        
        # Fit and transform all texts at once
        try:
            embeddings_matrix = self.text_vectorizer.fit_transform(texts).toarray()
            
            # Store embeddings
            for startup_id, embedding in zip(startup_ids, embeddings_matrix):
                self.startup_embeddings[startup_id] = embedding
            
            print(f"Built TF-IDF embeddings for {len(self.startup_embeddings)} startups")
        except Exception as e:
            print(f"Error building embeddings: {e}")
    
    def compute_user_similarity_matrix(self, feature_engineer):
        """Build user-user similarity matrix for collaborative filtering"""
        print("Computing user similarity matrix...")
        
        # Create user-startup interaction matrix
        interactions = feature_engineer.interactions_df
        
        # Pivot to create user-item matrix
        user_item_matrix = interactions.pivot_table(
            index='user_id',
            columns='startup_id',
            values='engagement_level',
            fill_value=0
        )
        
        # Compute cosine similarity between users
        if len(user_item_matrix) > 1:
            self.user_similarity_matrix = cosine_similarity(user_item_matrix)
            self.user_ids = user_item_matrix.index.tolist()
            print(f"Computed similarity for {len(self.user_ids)} users")
        else:
            print("Not enough users for collaborative filtering")
            self.user_similarity_matrix = None
    
    def get_similar_users(self, user_id, n=5):
        """Find N most similar users using collaborative filtering"""
        if self.user_similarity_matrix is None:
            return []
        
        try:
            user_idx = self.user_ids.index(user_id)
            similarities = self.user_similarity_matrix[user_idx]
            
            # Get top N similar users (excluding self)
            similar_indices = np.argsort(similarities)[::-1][1:n+1]
            similar_user_ids = [self.user_ids[i] for i in similar_indices]
            
            return similar_user_ids
        except (ValueError, IndexError):
            return []
    
    def collaborative_score(self, user_id, startup_id, feature_engineer):
        """Score based on what similar users liked"""
        similar_users = self.get_similar_users(user_id, n=10)
        
        if not similar_users:
            return 0.0
        
        # Check how similar users interacted with this startup
        interactions = feature_engineer.interactions_df
        similar_interactions = interactions[
            (interactions['user_id'].isin(similar_users)) &
            (interactions['startup_id'] == startup_id)
        ]
        
        if len(similar_interactions) == 0:
            return 0.0
        
        # Average engagement level from similar users
        avg_engagement = similar_interactions['engagement_level'].mean()
        
        # Normalize to 0-1
        return avg_engagement / 3.0
    
    def content_based_score(self, user_id, startup_id, feature_engineer):
        """Score based on similarity to startups user liked"""
        if not self.startup_embeddings:
            return 0.0
        
        # Get startups user has engaged with (engagement >= 2)
        user_interactions = feature_engineer.interactions_df[
            (feature_engineer.interactions_df['user_id'] == user_id) &
            (feature_engineer.interactions_df['engagement_level'] >= 2)
        ]
        
        if len(user_interactions) == 0:
            return 0.0
        
        liked_startup_ids = user_interactions['startup_id'].tolist()
        
        # Get embeddings for liked startups
        liked_embeddings = []
        for sid in liked_startup_ids:
            if sid in self.startup_embeddings:
                liked_embeddings.append(self.startup_embeddings[sid])
        
        if not liked_embeddings or startup_id not in self.startup_embeddings:
            return 0.0
        
        # Average embedding of liked startups
        avg_liked_embedding = np.mean(liked_embeddings, axis=0)
        
        # Compare with candidate startup
        candidate_embedding = self.startup_embeddings[startup_id]
        
        # Cosine similarity
        similarity = cosine_similarity(
            avg_liked_embedding.reshape(1, -1),
            candidate_embedding.reshape(1, -1)
        )[0][0]
        
        # Already in [0, 1] range for TF-IDF
        return max(0, similarity)
    
    def ml_prediction_score(self, user_id, startup_id, feature_engineer):
        """Score from ML model prediction"""
        if not self.is_trained:
            return 0.0
        
        # Get startup features
        startup = feature_engineer.startup_features[
            feature_engineer.startup_features['id'] == startup_id
        ]
        
        if len(startup) == 0:
            return 0.0
        
        startup = startup.iloc[0]
        
        # Get user preferences
        user_pref = feature_engineer.user_preferences[
            feature_engineer.user_preferences['user_id'] == user_id
        ]
        
        if len(user_pref) == 0:
            user_pref = {'total_views': 0, 'avg_engagement': 0}
        else:
            user_pref = user_pref.iloc[0].to_dict()
        
        # Create feature vector
        features = {
            'revenue': startup.get('revenue', 0),
            'net_income': startup.get('net_income', 0),
            'profit_margin': startup.get('profit_margin', 0),
            'revenue_growth': startup.get('revenue_growth', 0),
            'expected_return': startup.get('expected_return', 0),
            'current_ratio': startup.get('current_ratio', 0),
            'debt_to_assets': startup.get('debt_to_assets', 0),
            'confidence_score': startup.get('confidence_score', 0),
            'is_deck_builder': startup.get('is_deck_builder', 0),
            'view_count': startup.get('view_count', 0),
            'unique_viewers': startup.get('unique_viewers', 0),
            'avg_engagement': startup.get('avg_engagement', 0),
            'industry_encoded': startup.get('industry_encoded', 0),
            'user_total_views': user_pref.get('total_views', 0),
            'user_avg_engagement': user_pref.get('avg_engagement', 0),
        }
        
        # Convert to array
        X = np.array([features[col] for col in self.feature_columns]).reshape(1, -1)
        
        # Predict probability
        proba = self.model.predict_proba(X)[0]
        
        # Weighted score
        score = proba[0] * 1 + proba[1] * 2 + proba[2] * 3
        return score / 3.0  # Normalize to 0-1
    
    def hybrid_score(self, user_id, startup_id, feature_engineer, weights=None):
        """
        Combine all scoring methods
        
        weights: dict with keys 'ml', 'collaborative', 'content'
        Default: 40% ML, 30% Collaborative, 30% Content-based
        """
        if weights is None:
            weights = {
                'ml': 0.4,
                'collaborative': 0.3,
                'content': 0.3
            }
        
        ml_score = self.ml_prediction_score(user_id, startup_id, feature_engineer)
        collab_score = self.collaborative_score(user_id, startup_id, feature_engineer)
        content_score = self.content_based_score(user_id, startup_id, feature_engineer)
        
        final_score = (
            weights['ml'] * ml_score +
            weights['collaborative'] * collab_score +
            weights['content'] * content_score
        )
        
        return {
            'total_score': final_score,
            'ml_score': ml_score,
            'collaborative_score': collab_score,
            'content_score': content_score
        }
    
    def train(self, X, y, feature_columns, feature_engineer):
        """Train the hybrid recommender"""
        print("\n" + "="*60)
        print("TRAINING AI HYBRID RECOMMENDER")
        print("="*60)
        
        # Save feature columns
        self.feature_columns = feature_columns
        
        # 1. Train ML model
        print("\n1. Training ML Model...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"   ML Model Accuracy: {accuracy:.2%}")
        
        # 2. Build NLP embeddings
        print("\n2. Building NLP Components...")
        self.build_startup_embeddings(feature_engineer)
        
        # 3. Compute user similarities
        print("\n3. Building Collaborative Filtering...")
        self.compute_user_similarity_matrix(feature_engineer)
        
        self.is_trained = True
        
        print("\n" + "="*60)
        print("HYBRID AI TRAINING COMPLETE")
        print("="*60)
        
        return self
    
    def recommend(self, user_id, feature_engineer, n_recommendations=10, 
                  exclude_viewed=True, weights=None):
        """Generate hybrid AI recommendations"""
        
        # Get all startup IDs
        all_startup_ids = feature_engineer.startup_features['id'].tolist()
        
        # Optionally exclude viewed
        if exclude_viewed:
            viewed_ids = feature_engineer.interactions_df[
                feature_engineer.interactions_df['user_id'] == user_id
            ]['startup_id'].tolist()
            candidate_ids = [sid for sid in all_startup_ids if sid not in viewed_ids]
        else:
            candidate_ids = all_startup_ids
        
        if len(candidate_ids) == 0:
            candidate_ids = all_startup_ids
        
        # Score all candidates
        recommendations = []
        
        for startup_id in candidate_ids:
            scores = self.hybrid_score(user_id, startup_id, feature_engineer, weights)
            
            # Get predicted engagement
            if self.is_trained:
                startup = feature_engineer.startup_features[
                    feature_engineer.startup_features['id'] == startup_id
                ].iloc[0]
                
                user_pref = feature_engineer.user_preferences[
                    feature_engineer.user_preferences['user_id'] == user_id
                ]
                
                if len(user_pref) == 0:
                    user_pref = {'total_views': 0, 'avg_engagement': 0}
                else:
                    user_pref = user_pref.iloc[0].to_dict()
                
                features = {
                    'revenue': startup.get('revenue', 0),
                    'net_income': startup.get('net_income', 0),
                    'profit_margin': startup.get('profit_margin', 0),
                    'revenue_growth': startup.get('revenue_growth', 0),
                    'expected_return': startup.get('expected_return', 0),
                    'current_ratio': startup.get('current_ratio', 0),
                    'debt_to_assets': startup.get('debt_to_assets', 0),
                    'confidence_score': startup.get('confidence_score', 0),
                    'is_deck_builder': startup.get('is_deck_builder', 0),
                    'view_count': startup.get('view_count', 0),
                    'unique_viewers': startup.get('unique_viewers', 0),
                    'avg_engagement': startup.get('avg_engagement', 0),
                    'industry_encoded': startup.get('industry_encoded', 0),
                    'user_total_views': user_pref.get('total_views', 0),
                    'user_avg_engagement': user_pref.get('avg_engagement', 0),
                }
                
                X = np.array([features[col] for col in self.feature_columns]).reshape(1, -1)
                predicted_engagement = self.model.predict(X)[0]
            else:
                predicted_engagement = 1
            
            recommendations.append({
                'startup_id': startup_id,
                'score': scores['total_score'],
                'ml_score': scores['ml_score'],
                'collaborative_score': scores['collaborative_score'],
                'content_score': scores['content_score'],
                'predicted_engagement': int(predicted_engagement)
            })
        
        # Sort by total score
        recommendations_df = pd.DataFrame(recommendations)
        recommendations_df = recommendations_df.sort_values('score', ascending=False)
        
        return recommendations_df.head(n_recommendations)
    
    def explain_recommendation(self, user_id, startup_id, feature_engineer):
        """Explain why this startup was recommended"""
        scores = self.hybrid_score(user_id, startup_id, feature_engineer)
        
        startup = feature_engineer.startup_features[
            feature_engineer.startup_features['id'] == startup_id
        ].iloc[0]
        
        explanation = {
            'startup_name': startup['company_name'],
            'overall_score': scores['total_score'],
            'reasons': []
        }
        
        # ML-based reasons
        if scores['ml_score'] > 0.5:
            explanation['reasons'].append(
                f"Strong match based on your behavior patterns (ML Score: {scores['ml_score']:.2f})"
            )
        
        # Collaborative filtering reasons
        if scores['collaborative_score'] > 0.3:
            similar_users = self.get_similar_users(user_id, n=3)
            if similar_users:
                explanation['reasons'].append(
                    f"Popular among users with similar interests (Collab Score: {scores['collaborative_score']:.2f})"
                )
        
        # Content-based reasons
        if scores['content_score'] > 0.5:
            explanation['reasons'].append(
                f"Similar to startups you previously engaged with (Content Score: {scores['content_score']:.2f})"
            )
        
        # Startup-specific reasons
        if startup.get('revenue_growth', 0) > 50:
            explanation['reasons'].append(
                f"High revenue growth: {startup['revenue_growth']:.1f}%"
            )
        
        if startup.get('expected_return', 0) > 20:
            explanation['reasons'].append(
                f"Strong expected return: {startup['expected_return']:.1f}%"
            )
        
        # Industry preference
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
    
    def save(self, path='data/models/ai_hybrid_recommender_latest.pkl'):
        """Save the trained hybrid AI recommender"""
        if not self.is_trained:
            raise Exception("Cannot save untrained model!")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save everything except the NLP model (it's huge)
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained,
            'startup_embeddings': self.startup_embeddings,
            'user_similarity_matrix': self.user_similarity_matrix,
            'user_ids': getattr(self, 'user_ids', None),
            'trained_at': datetime.now().isoformat(),
            'model_type': 'AIRecommender'
        }
        
        joblib.dump(model_data, path)
        print(f"AI Hybrid Model saved to {path}")
        
        # Also save with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        versioned_path = path.replace('_latest', f'_{timestamp}')
        joblib.dump(model_data, versioned_path)
        print(f"Version saved to {versioned_path}")
    
    @staticmethod
    def load(path='data/models/ai_hybrid_recommender_latest.pkl'):
        """Load a trained hybrid AI recommender"""
        model_data = joblib.load(path)
        
        recommender = AIRecommender()
        recommender.model = model_data['model']
        recommender.feature_columns = model_data['feature_columns']
        recommender.is_trained = model_data['is_trained']
        recommender.startup_embeddings = model_data['startup_embeddings']
        recommender.user_similarity_matrix = model_data['user_similarity_matrix']
        recommender.user_ids = model_data.get('user_ids', [])
        
        print(f"AI Hybrid Model loaded from {path}")
        print(f"   Trained at: {model_data.get('trained_at', 'Unknown')}")
        
        return recommender


# Training script
if __name__ == "__main__":
    from src.features.feature_engineer import FeatureEngineer

    print("="*60)
    print("FUNDORA AI HYBRID RECOMMENDER - TRAINING")
    print("="*60)

    # 1. Load feature engineer and data
    print("\n1. Loading feature engineer...")
    fe = FeatureEngineer.load('data/processed/feature_engineer.pkl')

    print("\n2. Preparing training data...")
    X, y = fe.get_training_data()
    feature_columns = fe.feature_columns
    print(f"   Training examples: {len(X)}")
    print(f"   Features: {len(feature_columns)}")

    # 3. Train/validation split
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Create and train your ML model (same config you use inside the hybrid model)
    from sklearn.ensemble import RandomForestClassifier
    ml_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )

    print("\n3. Training ML model...")
    ml_model.fit(X_train, y_train)

    # 5. Evaluate
    from sklearn.metrics import (
        confusion_matrix,
        ConfusionMatrixDisplay,
        classification_report,
        accuracy_score
    )
    import matplotlib.pyplot as plt
    import numpy as np

    y_pred = ml_model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"   Validation Accuracy: {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['View', 'Compare', 'Watchlist']))

    # Confusion matrix figure
    cm = confusion_matrix(y_val, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['View (1)', 'Compare (2)', 'Watchlist (3)']
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(values_format='d', cmap='Blues', ax=ax, colorbar=True)
    
    # Improve formatting
    ax.set_title('Confusion Matrix of Engagement Prediction', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold', labelpad=10)
    ax.tick_params(axis='both', which='major', labelsize=11)
    
    plt.tight_layout()
    plt.savefig('fig_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("\nSaved fig_confusion_matrix.png and fig_feature_importance.png")
