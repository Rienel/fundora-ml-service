import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from datetime import datetime

class StartupRecommender:
    def __init__(self):
        # Using RandomForest for now (easier to start with, works well with small data)
        # Can upgrade to XGBoost or neural networks later
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced engagement levels
        )
        self.feature_engineer = None
        self.is_trained = False
        self.feature_columns = None
        self.startup_features_cache = None
        
    def train(self, X, y, feature_columns):
        """Train the recommendation model"""
        print("Starting model training...")
        
        # Save feature columns
        self.feature_columns = feature_columns
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} examples")
        print(f"Validation set: {X_val.shape[0]} examples")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        
        print(f"\nModel trained successfully!")
        print(f"Validation Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred, 
                                   target_names=['View', 'Compare', 'Watchlist']))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 5 Most Important Features:")
        print(feature_importance.head())
        
        self.is_trained = True
        
        return self
    
    def predict_engagement(self, user_id, startup_ids, feature_engineer):
        """Predict engagement level for user-startup pairs"""
        if not self.is_trained:
            raise Exception("Model not trained yet!")
        
        predictions = []
        
        for startup_id in startup_ids:
            # Get startup features
            startup = feature_engineer.startup_features[
                feature_engineer.startup_features['id'] == startup_id
            ]
            
            if len(startup) == 0:
                continue
            
            startup = startup.iloc[0]
            
            # Get user preferences
            user_pref = feature_engineer.user_preferences[
                feature_engineer.user_preferences['user_id'] == user_id
            ]
            
            if len(user_pref) == 0:
                # Cold start user - use defaults
                user_pref = {
                    'total_views': 0,
                    'avg_engagement': 0
                }
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
            
            # Convert to array in correct order
            X = np.array([features[col] for col in self.feature_columns]).reshape(1, -1)
            
            # Predict probability of each engagement level
            proba = self.model.predict_proba(X)[0]
            
            # Calculate weighted score (higher engagement = higher score)
            # engagement_level: 1=view, 2=compare, 3=watchlist
            score = proba[0] * 1 + proba[1] * 2 + proba[2] * 3
            
            predictions.append({
                'startup_id': startup_id,
                'score': score,
                'predicted_engagement': self.model.predict(X)[0],
                'proba_view': proba[0],
                'proba_compare': proba[1] if len(proba) > 1 else 0,
                'proba_watchlist': proba[2] if len(proba) > 2 else 0
            })
        
        return predictions
    
    def recommend(self, user_id, feature_engineer, n_recommendations=10, 
                  exclude_viewed=True):
        """Generate top N recommendations for a user"""
        
        # Get all startup IDs
        all_startup_ids = feature_engineer.startup_features['id'].tolist()
        
        # Optionally exclude already viewed startups
        if exclude_viewed:
            viewed_ids = feature_engineer.interactions_df[
                feature_engineer.interactions_df['user_id'] == user_id
            ]['startup_id'].tolist()
            
            candidate_ids = [sid for sid in all_startup_ids if sid not in viewed_ids]
        else:
            candidate_ids = all_startup_ids
        
        if len(candidate_ids) == 0:
            print(f"User {user_id} has viewed all startups!")
            candidate_ids = all_startup_ids
        
        # Predict scores for all candidates
        predictions = self.predict_engagement(user_id, candidate_ids, feature_engineer)
        
        # Sort by score (descending)
        predictions_df = pd.DataFrame(predictions)
        predictions_df = predictions_df.sort_values('score', ascending=False)
        
        # Return top N
        top_recommendations = predictions_df.head(n_recommendations)
        
        return top_recommendations
    
    def save(self, path='data/models/recommender_latest.pkl'):
        """Save the trained model"""
        if not self.is_trained:
            raise Exception("Cannot save untrained model!")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model with metadata
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained,
            'trained_at': datetime.now().isoformat(),
            'model_type': 'RandomForestClassifier'
        }
        
        joblib.dump(model_data, path)
        print(f"✅ Model saved to {path}")
        
        # Also save with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        versioned_path = path.replace('_latest', f'_{timestamp}')
        joblib.dump(model_data, versioned_path)
        print(f"Model version saved to {versioned_path}")
    
    @staticmethod
    def load(path='data/models/recommender_latest.pkl'):
        """Load a trained model"""
        model_data = joblib.load(path)
        
        recommender = StartupRecommender()
        recommender.model = model_data['model']
        recommender.feature_columns = model_data['feature_columns']
        recommender.is_trained = model_data['is_trained']
        
        print(f"   Model loaded from {path}")
        print(f"   Trained at: {model_data.get('trained_at', 'Unknown')}")
        
        return recommender


# Training script
if __name__ == "__main__":
    from src.features.feature_engineer import FeatureEngineer
    
    print("="*60)
    print("FUNDORA ML - MODEL TRAINING")
    print("="*60)
    
    # Load feature engineer
    print("\n1. Loading feature engineer...")
    fe = FeatureEngineer.load('data/processed/feature_engineer.pkl')
    
    # Get training data
    print("\n2. Preparing training data...")
    X, y = fe.get_training_data()
    print(f"   Training examples: {len(X)}")
    print(f"   Features: {len(fe.feature_columns)}")
    print(f"   Engagement distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for eng_level, count in zip(unique, counts):
        engagement_name = ['View', 'Compare', 'Watchlist'][int(eng_level)-1]
        print(f"      {engagement_name}: {count} ({count/len(y)*100:.1f}%)")
    
    # Train model
    print("\n3. Training model...")
    recommender = StartupRecommender()
    recommender.train(X, y, fe.feature_columns)
    
    # Save model
    print("\n4. Saving model...")
    recommender.save()
    
    # Test recommendations
    print("\n5. Testing recommendations...")
    print("-"*60)
    
    # Get a user who has some interactions
    test_user_id = fe.interactions_df['user_id'].value_counts().index[0]
    print(f"\nGenerating recommendations for User ID: {test_user_id}")
    
    recommendations = recommender.recommend(test_user_id, fe, n_recommendations=5)
    
    print("\nTop 5 Recommendations:")
    print(recommendations[['startup_id', 'score', 'predicted_engagement']].to_string(index=False))
    
    # Show startup names
    print("\nWith startup names:")
    for _, rec in recommendations.iterrows():
        startup = fe.startup_features[fe.startup_features['id'] == rec['startup_id']].iloc[0]
        engagement_name = ['View', 'Compare', 'Watchlist'][int(rec['predicted_engagement'])-1]
        print(f"  - {startup['company_name']} (Score: {rec['score']:.2f}, Predicted: {engagement_name})")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)