import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_heatmap(source_labels, target_labels, relations, title="Heatmap of Label Relationships",
                     figsize=(16, 10), cmap="Blues", annot=True, linewidths=0.5, cbar=True, save_path=None):
    """
    Generate a professional heatmap for scientific papers with attention scores.

    Parameters:
    - source_labels: List of source labels (rows)
    - target_labels: List of target labels (columns)
    - relations: Dictionary mapping source labels to target labels with scores
    - title: Title of the heatmap (default: "Heatmap of Label Relationships")
    - figsize: Tuple specifying figure size
    - cmap: Color scheme
    - annot: Whether to annotate cells with values
    - linewidths: Line thickness between cells
    - cbar: Whether to display the color bar
    - save_path: Path to save the figure
    """
    # Create matrix representation of relations
    matrix = np.zeros((len(source_labels), len(target_labels)))
    matrix_annotate = np.zeros((len(source_labels), len(target_labels)))
    for source, targets in relations.items():
        for target, score in targets.items():
            if target in target_labels:
                matrix[source_labels.index(source), target_labels.index(target)] = score
                matrix_annotate[source_labels.index(source), target_labels.index(target)] = str(score)
            else:
                matrix[source_labels.index(source), target_labels.index(target)] = 0
                matrix_annotate[source_labels.index(source), target_labels.index(target)] = ""


    df_heatmap = pd.DataFrame(matrix, index=source_labels, columns=target_labels)
    df_heatmap_annotate = pd.DataFrame(matrix_annotate, index=source_labels, columns=target_labels)

    # Create figure
    plt.figure(figsize=figsize)
    ax = sns.heatmap(df_heatmap, annot=df_heatmap_annotate, fmt=".2f", cmap=cmap, linewidths=linewidths,
                     cbar=cbar, linecolor='gray', square=False)

    # Add borders on all four sides
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')

    # Format plot
    # plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Source Labels", fontsize=14)
    plt.ylabel("Target Labels", fontsize=14)
    plt.xticks(rotation=90, ha='center', fontsize=11)
    plt.yticks(fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    source_target = {
        "GermEval 18 - Binary - normal": {"Olid - Offensive": 81.06, "Xdomain - Politics - Offensive": 1.66, "Xdomain - Religion - Normal": 1.28},
        "GermEval 18 - Binary - offensive": {"Hateval 19 - Aggressive - Normal": 11.5, "HASOC 19 - Fine-grained - Hate": 9.0, "HASOC 19 (de) - Binary - Normal": 6.04},
        "GermEval 18 - Fine-grained - Normal": {"Xdomain - Politics - Normal": 78.34, "Xplain - Target - Racism": 1.78, "HASOC 19 (de) - Binary - Normal": 6.04},
        "GermEval 18 - Fine-grained - Profanity": {"Xplain - Target - Religious": 98.81},
        "GermEval 18 - Fine-grained - Insult": {"HASOC19 (en) - Targeted - Unintentional": 6.57, "Hateval 19 - Target - Generic": 5.95, "HASOC 19 (de) - Fine-grained - Offensive": 5.92},
        "GermEval 18 - Fine-grained - Abuse": {"Xplain - Target - Racism": 61.57, "Olid - Offensive - Normal": 2.98, "Xdomain - Politics - Normal": 2.69},
        "HateSpeech 18 - Binary - Normal": {"Hateval 19 - Target - Individial": 10.54, "SRW16 - Fine-grained - Racism": 7.8, "HASOC 19 - Fine-grained - Profanity": 5.33},
        "HateSpeech 18 - Binary - Hate": {"Xdoamin - Politics - Hate": 91.02},
        "SRW 16 - Binary - Offensive": {"HASOC19 (de) - Fine-grained - Hate": 17.75, "HASOC19 (de) - Fine-grained - Offensive": 17.21, "HASOC19 (de) - Fine-grained - Profanity": 17.17},
        "SRW 16 - Binary - Normal": {"Olid - Offensive - Normal": 86.67},
        # "SRW 16 - Fine-grained - sexism": {"HASOC19 (de) - Fine-grained - Profanity": 12.04, "HASOC19 (de) - Fine-grained - Hate": 11.96, "HASOC19 (de) - Fine-grained - Offensive": 11.76},
        # "SRW 16 - Fine-grained - racism": {"HateSpeech18 - Binary - Hate": 15.1, "GermEval18 - Fine-grained - Profanity": 10.8, "Xdomain - Religion - Normal": 5.92},
        # "SRW 16 - Fine-grained - normal": {"HASOC19 (en) - Targeted - Untentional": 17.7, "HASOC19 (de) - Fine-grained - Hate": 7.29, "HASOC19 (de)- Fine-grained - Profanity": 6.81},
    }

    source_labels = list(source_target.keys())
    target_labels = sorted({label for targets in source_target.values() for label in targets.keys()})
    relations = {
        source: {
            target: source_target[source].get(target, 0) for target in target_labels
        } for source in source_labels
    }

    generate_heatmap(source_labels, target_labels, relations)
