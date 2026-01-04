import matplotlib.pyplot as plt

# Replace these with your actual counts if they change
total_startups = 16
total_users = 15
total_views = 70
total_watchlists = 4
total_comparisons = 9

# 2a. Entities: startups vs users
def plot_entities():
    labels = ['Startups', 'Users']
    values = [total_startups, total_users]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=['#4C72B0', '#55A868'])
    plt.title('Counts of Startups and Users')
    plt.ylabel('Count')
    for i, v in enumerate(values):
        plt.text(i, v + 0.5, str(v), ha='center')
    plt.tight_layout()
    plt.savefig('fig_entities_counts.png', dpi=300)
    plt.close()

# 2b. Interaction types
def plot_interactions():
    labels = ['Views', 'Watchlists', 'Comparisons']
    values = [total_views, total_watchlists, total_comparisons]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=['#4C72B0', '#C44E52', '#8172B2'])
    plt.title('Distribution of Interaction Types')
    plt.ylabel('Count')
    for i, v in enumerate(values):
        plt.text(i, v + 0.5, str(v), ha='center')
    plt.tight_layout()
    plt.savefig('fig_interaction_counts.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_entities()
    plot_interactions()
    print("Saved fig_entities_counts.png and fig_interaction_counts.png")
