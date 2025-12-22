import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

class FeatureEngineer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, startups_path, interactions_path, users_path):
        """Load the exported CSV files"""
        self.startups_df = pd.read_csv(startups_path)
        self.interactions_df = pd.read_csv(interactions_path)
        self.users_df = pd.read_csv(users_path)
        
        print(f"Loaded {len(self.startups_df)} startups")
        print(f"Loaded {len(self.interactions_df)} interactions")
        print(f"Loaded {len(self.users_df)} users")
        
        return self
    
    def engineer_startup_features(self):
        """Create features from startup data"""
        df = self.startups_df.copy()
        
        # Fill missing values with 0
        numeric_cols = ['revenue', 'net_income', 'total_assets', 'total_liabilities',
                       'current_revenue', 'previous_revenue', 'current_valuation',
                       'expected_future_valuation', 'current_assets', 'current_liabilities',
                       'retained_earnings', 'ebit']
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Calculate financial ratios
        df['profit_margin'] = np.where(
            df['revenue'] > 0,
            df['net_income'] / df['revenue'],
            0
        )
        
        df['asset_turnover'] = np.where(
            df['total_assets'] > 0,
            df['revenue'] / df['total_assets'],
            0
        )
        
        df['current_ratio'] = np.where(
            df['current_liabilities'] > 0,
            df['current_assets'] / df['current_liabilities'],
            0
        )
        
        df['debt_to_assets'] = np.where(
            df['total_assets'] > 0,
            df['total_liabilities'] / df['total_assets'],
            0
        )
        
        # Growth rate
        df['revenue_growth'] = np.where(
            (df['previous_revenue'] > 0) & (df['current_revenue'] > 0),
            (df['current_revenue'] - df['previous_revenue']) / df['previous_revenue'],
            0
        )
        
        # Expected return (IRR approximation)
        df['expected_return'] = np.where(
            (df['current_valuation'] > 0) & 
            (df['expected_future_valuation'] > 0) & 
            (df['years_to_future_valuation'] > 0),
            ((df['expected_future_valuation'] / df['current_valuation']) ** 
             (1 / df['years_to_future_valuation']) - 1) * 100,
            0
        )
        
        # Encode categorical variables
        if 'industry' in df.columns:
            if 'industry' not in self.label_encoders:
                self.label_encoders['industry'] = LabelEncoder()
                df['industry_encoded'] = self.label_encoders['industry'].fit_transform(
                    df['industry'].fillna('Unknown')
                )
            else:
                df['industry_encoded'] = self.label_encoders['industry'].transform(
                    df['industry'].fillna('Unknown')
                )
        
        # Binary features
        df['is_deck_builder'] = df['is_deck_builder'].astype(int)
        
        # Confidence score
        df['confidence_score'] = df['confidence_percentage'] / 100.0
        
        self.startup_features = df
        print(f"Engineered {len(df.columns)} features for startups")
        
        return self
    
    def calculate_startup_popularity(self):
        """Calculate popularity metrics from interactions"""
        # Count views per startup
        view_counts = self.interactions_df.groupby('startup_id').size().reset_index(name='view_count')
        
        # Count unique viewers
        unique_viewers = self.interactions_df.groupby('startup_id')['user_id'].nunique().reset_index(name='unique_viewers')
        
        # Average engagement level
        avg_engagement = self.interactions_df.groupby('startup_id')['engagement_level'].mean().reset_index(name='avg_engagement')
        
        # Merge popularity features
        popularity = view_counts.merge(unique_viewers, on='startup_id', how='outer')
        popularity = popularity.merge(avg_engagement, on='startup_id', how='outer')
        
        # Fill missing values
        popularity = popularity.fillna(0)
        
        # Add to startup features
        self.startup_features = self.startup_features.merge(
            popularity, 
            left_on='id', 
            right_on='startup_id', 
            how='left'
        )
        
        # Fill missing popularity metrics with 0
        self.startup_features['view_count'] = self.startup_features['view_count'].fillna(0)
        self.startup_features['unique_viewers'] = self.startup_features['unique_viewers'].fillna(0)
        self.startup_features['avg_engagement'] = self.startup_features['avg_engagement'].fillna(0)
        
        print("Added popularity features")
        
        return self
    
    def create_user_preferences(self):
        """Create user preference profiles from their interaction history"""
        user_prefs = []
        
        for user_id in self.users_df['id'].unique():
            # Get user's interactions
            user_interactions = self.interactions_df[
                self.interactions_df['user_id'] == user_id
            ]
            
            if len(user_interactions) == 0:
                # Cold start: no preferences yet
                user_prefs.append({
                    'user_id': user_id,
                    'total_views': 0,
                    'avg_engagement': 0,
                    'preferred_industry': 'Unknown'
                })
                continue
            
            # Get viewed startups
            viewed_startups = self.startup_features[
                self.startup_features['id'].isin(user_interactions['startup_id'])
            ]
            
            # Calculate preferences
            user_pref = {
                'user_id': user_id,
                'total_views': len(user_interactions),
                'avg_engagement': user_interactions['engagement_level'].mean(),
                'preferred_industry': viewed_startups['industry'].mode()[0] if len(viewed_startups) > 0 else 'Unknown'
            }
            
            user_prefs.append(user_pref)
        
        self.user_preferences = pd.DataFrame(user_prefs)
        print(f"Created preference profiles for {len(self.user_preferences)} users")
        
        return self
    
    def prepare_training_data(self):
        """Prepare data for ML model training"""
        training_data = []
        
        # For each user interaction, create a training example
        for _, interaction in self.interactions_df.iterrows():
            user_id = interaction['user_id']
            startup_id = interaction['startup_id']
            engagement_level = interaction['engagement_level']
            
            # Get startup features
            startup = self.startup_features[self.startup_features['id'] == startup_id]
            if len(startup) == 0:
                continue
            startup = startup.iloc[0]
            
            # Get user preferences
            user_pref = self.user_preferences[self.user_preferences['user_id'] == user_id]
            if len(user_pref) == 0:
                continue
            user_pref = user_pref.iloc[0]
            
            # Combine features
            features = {
                'user_id': user_id,
                'startup_id': startup_id,
                
                # Startup features
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
                
                # User features
                'user_total_views': user_pref.get('total_views', 0),
                'user_avg_engagement': user_pref.get('avg_engagement', 0),
                
                # Target
                'label': engagement_level
            }
            
            training_data.append(features)
        
        self.training_df = pd.DataFrame(training_data)
        print(f"Created {len(self.training_df)} training examples")
        
        # Save feature names for later use
        self.feature_columns = [col for col in self.training_df.columns 
                                if col not in ['user_id', 'startup_id', 'label']]
        
        return self
    
    def get_training_data(self):
        """Return X, y for model training"""
        X = self.training_df[self.feature_columns].values
        y = self.training_df['label'].values
        
        return X, y
    
    def save(self, path='data/processed/feature_engineer.pkl'):
        """Save the feature engineer for later use"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Saved feature engineer to {path}")
    
    @staticmethod
    def load(path='data/processed/feature_engineer.pkl'):
        """Load a saved feature engineer"""
        return joblib.load(path)


# Test feature engineering
if __name__ == "__main__":
    import glob
    
    # Find the most recent data files
    startup_files = glob.glob('data/raw/startups_*.csv')
    interaction_files = glob.glob('data/raw/interactions_*.csv')
    user_files = glob.glob('data/raw/users_*.csv')
    
    if not startup_files:
        print("No data files found. Run data_collector.py first!")
        exit(1)
    
    # Use most recent files
    startup_file = sorted(startup_files)[-1]
    interaction_file = sorted(interaction_files)[-1]
    user_file = sorted(user_files)[-1]
    
    print(f"Using files:")
    print(f"  - {startup_file}")
    print(f"  - {interaction_file}")
    print(f"  - {user_file}\n")
    
    # Create feature engineer
    fe = FeatureEngineer()
    fe.load_data(startup_file, interaction_file, user_file)
    fe.engineer_startup_features()
    fe.calculate_startup_popularity()
    fe.create_user_preferences()
    fe.prepare_training_data()
    
    # Get training data
    X, y = fe.get_training_data()
    print(f"\n📊 Training Data Shape:")
    print(f"Features (X): {X.shape}")
    print(f"Labels (y): {y.shape}")
    print(f"Feature columns: {fe.feature_columns}")
    
    # Save
    fe.save()