import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd


def generate_bar_chart(data, datasets, methods, title="Comparison of Settings Across Datasets",
                       colors=None, save_path=None):
    """
    Generate a grouped bar chart for F1-macro scores across datasets.

    Parameters:
    - data: 2D numpy array of shape (num_datasets, num_methods)
    - datasets: List of dataset names
    - methods: List of method names (settings)
    - title: Chart title
    - colors: Custom color list (optional)
    - save_path: Path to save the figure (optional)
    """
    num_datasets = len(datasets)
    num_methods = len(methods)
    x = np.arange(num_datasets)
    width = 0.8 / num_methods  # Adjust width to fit bars neatly

    plt.figure(figsize=(12, 6))
    if colors is None:
        colors = sns.color_palette("tab10", num_methods)  # Default color palette

    for i, method in enumerate(methods):
        plt.bar(x + i * width, data[:, i], width=width, label=method, color=colors[i % len(colors)])

    plt.xticks(x + width * num_methods / 2, datasets, rotation=45, fontsize=12)
    plt.ylabel("F1-Macro Score", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(ncol=3, fontsize="small", loc="upper center", bbox_to_anchor=(0.5, -0.25))
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

# Example Usage
if __name__ == "__main__":
    datasets = ["Dataset 1", "Dataset 2", "Dataset 3", "Dataset 4", "Dataset 5", "Dataset 6", "Dataset 7", "Dataset 8",
                "Dataset 9"]
    methods = [f"Setting {i + 1}" for i in range(12)]  # 12 settings
    np.random.seed(42)
    scores = np.random.uniform(0.5, 0.9, (len(datasets), len(methods)))

    generate_bar_chart(scores, datasets, methods)