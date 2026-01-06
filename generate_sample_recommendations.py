import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.features.feature_engineer import FeatureEngineer
from src.models.ai_recommender import AIRecommender

# -------- CONFIG --------
FE_PATH = "data/processed/feature_engineer.pkl"
MODEL_PATH = "data/models/ai_hybrid_recommender_latest.pkl"
OUTPUT_PNG = "sample_recommendations.png"
N_RECOMMENDATIONS = 5
# ------------------------

def pick_good_user(fe: FeatureEngineer, min_interactions: int = 5):
    """
    Pick a user with enough interactions so recommendations are meaningful.
    """
    counts = fe.interactions_df["user_id"].value_counts()
    good_users = counts[counts >= min_interactions].index.tolist()
    if not good_users:
        # fall back to the most active user
        return counts.index[0]
    return good_users[0]

def main():
    print("=" * 60)
    print("Generating sample recommendation PNG")
    print("=" * 60)

    # 1. Load feature engineer and model
    fe = FeatureEngineer.load(FE_PATH)
    recommender = AIRecommender.load(MODEL_PATH)

    # 2. Pick a user with rich history
    user_id = pick_good_user(fe, min_interactions=20)
    print(f"Using user_id = {user_id} for sample recommendations")

    # 3. Get recommendations
    rec_df = recommender.recommend(
        user_id=user_id,
        feature_engineer=fe,
        n_recommendations=N_RECOMMENDATIONS,
        exclude_viewed=True,
    )

    # 4. Join with startup info and select columns for display
    rows = []
    for _, row in rec_df.iterrows():
        startup = fe.startup_features[fe.startup_features["id"] == row["startup_id"]].iloc[0]
        rows.append({
            "Rank": len(rows) + 1,
            "Startup": startup["company_name"],
            "Industry": startup.get("industry", ""),
            "Revenue": f"{startup.get('revenue', 0):,.0f}",
            "Revenue Growth %": f"{startup.get('revenue_growth', 0):.1%}",
            "Expected Return %": f"{startup.get('expected_return', 0):.1f}",
            "Conf. Score": f"{startup.get('confidence_score', 0):.2f}",
            "Hybrid Score": f"{row['score']:.3f}",
        })

    table_df = pd.DataFrame(rows)

    # 5. Plot as a table and save to PNG
    fig, ax = plt.subplots(figsize=(12, 2 + 0.5 * len(table_df)))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    plt.title(f"Sample Recommendations for User {user_id}", fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=200)
    plt.close(fig)

    print(f"Saved sample recommendations to {OUTPUT_PNG}")

if __name__ == "__main__":
    main()
