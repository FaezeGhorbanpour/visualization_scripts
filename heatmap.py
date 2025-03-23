import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def generate_heatmap(source_labels, target_labels, relations, title="Heatmap of Label Relationships",
                     figsize=(10, 6), cmap="Blues", annot=True, linewidths=0.5, cbar=True, save_path=None):
    """
    Generate a professional heatmap for scientific papers with attention scores.

    Parameters:
    - source_labels: List of source labels (rows)
    - target_labels: List of target labels (columns)
    - relations: Dictionary mapping source labels to target labels with scores
    - title: Title of the heatmap (default: "Heatmap of Label Relationships")
    - figsize: Tuple specifying figure size (default: (10, 6))
    - cmap: Color scheme (default: "Blues")
    - annot: Whether to annotate cells with values (default: True)
    - linewidths: Line thickness between cells (default: 0.5)
    - cbar: Whether to display the color bar (default: True)
    - save_path: Path to save the figure (default: None, meaning not saved)
    """
    # Create matrix representation of relations
    matrix = np.zeros((len(source_labels), len(target_labels)))
    for source, targets in relations.items():
        for target, score in targets.items():  # Include score values
            if target in target_labels:  # Ensure target exists in list
                matrix[source_labels.index(source), target_labels.index(target)] = score  # Assign score

    df_heatmap = pd.DataFrame(matrix, index=source_labels, columns=target_labels)

    # Create figure
    plt.figure(figsize=figsize)
    ax = sns.heatmap(df_heatmap, annot=True, fmt=".2f", cmap=cmap, linewidths=linewidths, cbar=cbar, linecolor='gray',
                     square=True)

    # Add borders on all four sides
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')

    # Format plot
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Target Labels", fontsize=12)
    plt.ylabel("Source Labels", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    # Save if needed
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# Example Usage
if __name__ == "__main__":
    # Define sample labels
    source_labels = [f"Source Label {i + 1}" for i in range(6)]
    target_labels = [f"Target Label {i + 1}" for i in range(15)]

    # Define random relations with attention scores
    np.random.seed(42)
    relations = {source: {target: round(np.random.uniform(0, 1), 2) for target in
                          np.random.choice(target_labels, np.random.randint(2, 4), replace=False)} for source in
                 source_labels}

    # Generate heatmap
    generate_heatmap(source_labels, target_labels, relations)
