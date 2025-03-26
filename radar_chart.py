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
        if 'Source' in method:
            ax.plot(angles, values, marker="*", label=method, linewidth=2)
        if 'Target' in method:
            ax.plot(angles, values, marker="o", label=method, linewidth=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(datasets, fontsize=12)

    # Modify y-axis to start from 0.2
    ylim_max = data.max() + 0.1  # Ensure full range is visible
    ax.set_ylim(40, ylim_max)
    # ax.set_yticks(np.linspace(30, ylim_max, 5))
    # ax.set_yticklabels([f'{x:.1f}' for x in np.linspace(0.2, ylim_max, 5)], fontsize=10)

    # ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Place legend inside the plot, lower right quadrant
    ax.legend(loc='lower right', bbox_to_anchor=(0.1, -0.1), fontsize=10,
              frameon=True, title="Methods")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Example Usage
if __name__ == "__main__":
    datasets_0 = {
        "GermEval18":[69.98, 70.46, 69.23, 70.37],
                "SRW16":[83.29, 85.94, 85.82, 85.77],
                "HateSpeech18":[72.57, 74.42, 74.94, 74.94],
                # "OLID":[76.01, 76.36, 76.20, 76.86],
                "OLID (target)":[49.31, 52.31, 48.62, 50.38],
                "OLID (targeted)":[54.28, 60.28, 50.96, 63.69],
                "HASOC19(de)":[55.23, 55.00, 55.22, 56.09],
                "average":[58.51, 60.01, 58.14, 60.89]}
    datasets_1 = {
                  # "GermEval18 (fine-grained)": [32.14, 40.23, 37.57, 42.33],
                "GermEval18":[62.67, 70.98, 70.58, 70.75],
                "SRW16":[83.11, 85.27, 85.69, 87.78],
                "HateSpeech18":[73.58, 76.43, 76.62, 77.03],
                # "SRW16 (fine-grained)":[57.55, 58.26, 58.60, 59.39],
                # "OLID":[77.48, 78.4, 78.65, 79.04],
                "OLID (target)":[52.02, 61.47, 55.75, 67.71],
                "OLID (targeted)":[54.68, 57.51, 48.96, 56.73],
                "HASOC19(de)":[52.95, 51.92, 53.23, 53.57],
                "average":[57.11, 61.32, 60.51, 62.51],
    }
    datasets_2 = {
        "GermEval18":[73.47, 72.12, 73.47, 73.93],
        "SRW16":[85.18, 85.92, 86.71, 87.22 ],
        "HateSpeech18":[76.02, 76.65, 78.66, 76.83],
        # "OLID":[78.38, 79.78, 78.74, 80.30],
        "OLID (targeted)":[57.57, 62.16, 58.67, 64.43],
        "OLID (target)":[46.31, 51.37, 49.77, 52.37],
        "HASOC19(de)":[55.10, 59.59, 52.90, 54.38],
        "average":[60.83, 61.06, 62.03, 62.39],
    }
    methods = ["Task-Specific Source SFT", "Label-Specific Source SFT", "Task-Specific Target SFT", "Label-Specific Target SFT"]
    np.random.seed(42)
    scores = np.array([np.array(values) for key,values in datasets_2.items()])
    generate_radar_chart(scores, datasets_2, methods)