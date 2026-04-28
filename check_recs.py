import sys
sys.path.insert(0, '.')
from src.features.feature_engineer import FeatureEngineer
from src.models.ai_recommender import AIRecommender

fe = FeatureEngineer.load('data/processed/feature_engineer.pkl')
model = AIRecommender.load('data/models/ai_hybrid_recommender_latest.pkl')

# Show interactions per user first
print("=== USER INTERACTIONS ===")
for uid in [11, 5, 6, 7]:
    print(f'\nUser {uid} interactions:')
    user_int = fe.interactions_df[fe.interactions_df['user_id'] == uid]
    for _, row in user_int.iterrows():
        startup = fe.startup_features[fe.startup_features['id'] == row['startup_id']].iloc[0]
        level = {1: 'View', 2: 'Compare', 3: 'Watchlist'}.get(row['engagement_level'], '?')
        print(f"  {startup['company_name']} ({startup['industry']}) — {level}")

# Show top 5 recommendations per user
print("\n=== TOP 5 RECOMMENDATIONS ===")
for uid in [11, 5]:
    print(f'\n--- User {uid} Top 5 ---')
    recs = model.recommend(uid, fe, n_recommendations=5, exclude_viewed=False)
    for _, row in recs.iterrows():
        startup = fe.startup_features[fe.startup_features['id'] == row['startup_id']].iloc[0]
        print(f"  {startup['company_name']} ({startup['industry']}) — score: {row['score']:.3f}")