import matplotlib.pyplot as plt
import numpy as np


def generate_radar_chart(data, datasets, methods, title="Performance Across Datasets (Radar Chart)",
                         save_path=None):
    """
    Generate a radar chart for comparing multiple settings across datasets.

    Parameters:
    - data: 2D numpy array of shape (num_datasets, num_methods)
    - datasets: List of dataset names
    - methods: List of method names (settings)
    - title: Chart title
    - save_path: Path to save the figure (optional)
    """
    num_vars = len(datasets)  # Fixed to 9 datasets
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for i, method in enumerate(methods[:5]):  # Limiting to 5 settings for readability
        values = data[:, i].tolist()
        values += values[:1]
        ax.plot(angles, values, marker="o", label=method, linewidth=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Ensure legend is visible and properly placed
    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.1), fontsize=10, frameon=True, title="Methods")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Example Usage
if __name__ == "__main__":
    datasets = ["Dataset 1", "Dataset 2", "Dataset 3", "Dataset 4", "Dataset 5", "Dataset 6", "Dataset 7", "Dataset 8",
                "Dataset 9"]
    methods = [f"Setting {i + 1}" for i in range(12)]  # 12 settings
    np.random.seed(42)
    scores = np.random.uniform(0.5, 0.9, (len(datasets), len(methods)))

    generate_radar_chart(scores, datasets, methods)