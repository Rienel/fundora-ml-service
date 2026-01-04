import matplotlib.pyplot as plt

data = [
    {
        "startup": "sample for sim",
        "score": 0.313283,
        "ml": 0.431736,
        "collab": 0.333333,
        "content": 0.135295,
        "explanation": "Popular among similar users; strong expected return"
    },
    {
        "startup": "sample",
        "score": 0.155650,
        "ml": 0.389124,
        "collab": 0.000000,
        "content": 0.000000,
        "explanation": "Matches your preferred industry"
    },
    {
        "startup": "LoanLinker",
        "score": 0.155650,
        "ml": 0.389124,
        "collab": 0.000000,
        "content": 0.000000,
        "explanation": "Matches your preferred industry"
    },
]

def plot_recommendation_table():
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.axis('off')

    col_labels = [
        "Startup",
        "Hybrid Score",
        "ML Score",
        "Collaborative\nScore",
        "Content\nScore",
        "Explanation"
    ]

    table_data = [
        [
            d["startup"],
            f"{d['score']:.3f}",
            f"{d['ml']:.3f}",
            f"{d['collab']:.3f}",
            f"{d['content']:.3f}",
            d["explanation"]
        ]
        for d in data
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='center',
        cellLoc='left'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    plt.title("Example AI Hybrid Recommendations and Explanations", pad=10)
    plt.tight_layout()
    plt.savefig("fig_recommendation_example.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_recommendation_table()
    print("Saved fig_recommendation_example.png")
