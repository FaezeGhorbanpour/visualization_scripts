import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors


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
    width = 0.8 / num_methods  # Adjust width to fit bars neatly with space
    spacing = 0.02  # Small space between groups

    # Define color scheme using tab20c colormap
    cmap = cm.get_cmap("tab20c")
    task_colors = [cmap(2), cmap(0), cmap(1)]  # Different shades of blue
    label_colors = [cmap(5), cmap(6), cmap(4)]  # Different shades of orange

    colors = []
    for i in range(num_methods):
        if "Label-Specific" in methods[i]:
            colors.append(label_colors[i % len(label_colors)])
        else:
            colors.append(task_colors[i % len(task_colors)])

    plt.figure(figsize=(12, 6))
    bars = []
    for i, method in enumerate(methods):
        bar = plt.bar(x + i * width, data[:, i], width=width, label=method, color=colors[i])
        bars.append(bar)

    plt.xticks(x + width * num_methods / 2, datasets, rotation=45, fontsize=12)
    plt.ylim(30, max(data.max() + 5, 100))
    plt.ylabel("F1-Macro Score", fontsize=12)

    # Improve legend placement and avoid overlap
    plt.legend(ncol=3, fontsize="small", loc="upper center", bbox_to_anchor=(0.5, -0.4), frameon=True)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# Example Usage
datasets_2 = ["GermEval18 (binary)", "SRW16 (binary)", "HateSpeech18", "HASOC (de - binary)", "OLID (targeted)", "OLID (target)", "average"]
scores = np.array([
    [62.67, 70.98, 65.78, 71.38, 70.58, 70.75],
    [83.11, 85.27, 84.04, 86.69, 85.69, 86.80],
    [73.58, 76.43, 73.71, 78.21, 76.62, 77.03],
    [52.95, 53.30, 50.19, 51.92, 53.23, 53.57],
    [47.02, 61.47, 59.80, 63.91, 55.75, 67.71],
    [54.68, 57.51, 54.81, 59.40, 48.96, 56.73],
    [57.11, 61.32, 59.98, 61.95, 60.45, 62.51]
])

methods = [
    "Task-Specific Source SFT",
    "Label-Specific Source SFT",
    "Task-Specific Target SFT with Initialization",
    "Label-Specific Target SFT with Initialization",
    "Task-Specific Target SFT with Attention",
    "Label-Specific Target SFT with Attention",
]

generate_bar_chart(scores, datasets_2, methods)