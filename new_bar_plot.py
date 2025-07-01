import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set style to match example background
plt.style.use("seaborn-v0_8-darkgrid")

# Sample data from your earlier message
data = {
    "Lang": ["es", "pt", "hi", "ar", "fr", "it", "de", "tr"],
    "BloomZ Prompt": ["Classification", "Definition", "Cultural", "NLI", "NLI", "CoT", "CoT", "Role Play"],
    "BloomZ F1": [54.50, 63.92, 51.33, 58.67, 55.63, 55.50, 38.36, 55.20],
    "Aya101 Prompt": ["Definition", "Definition", "Classification", "Distinction", "Translation", "Vanilla", "Vanilla",
                      "-"],
    "Aya101 F1": [63.68, 71.51, 47.33, 64.67, 53.44, 74.82, 67.51, None],
    "Llama3 Zero-shot Prompt": ["Classification", "Role Play", "CoT", "Classification", "CoT", "Distinction",
                                "Role Play", "Classification"],
    "Llama3 Zero-shot F1": [63.13, 70.79, 52.09, 62.66, 55.22, 75.86, 50.16, 76.16],
    "Llama3 Few-shot Prompt": ["5 shot + CoT", "5 shot + Cultural", "5 shot + Role Play", "5 shot + Cultural",
                               "5 shot + Definition", "5 shot + CoT", "5 shot + Cultural", "5 shot + CoT"],
    "Llama3 Few-shot F1": [68.89, 73.70, 55.55, 66.93, 51.53, 76.18, 78.14, 81.76],
    "Qwen Zero-shot Prompt": ["Translation", "Role Play + CoT", "Distinction", "NLI", "NLI", "Cultural", "Target",
                              "Translation"],
    "Qwen Zero-shot F1": [64.79, 73.44, 53.76, 70.61, 55.59, 73.34, 50.19, 75.89],
    "Qwen Few-shot Prompt": ["5 shot + CoT", "5 shot + Role Play", "1 shot + CoT", "5 shot", "5 shot",
                             "5 shot + Cultural", "5 shot + Definition", "5 shot + CoT"],
    "Qwen Few-shot F1": [68.90, 72.56, 49.57, 65.88, 51.78, 79.00, 77.55, 77.03],
"XLM-T (FT) F1":[82.78, 72.62, 59.18, 70.31, 51.42, 78.82, 79.18, 88.32],
"XLM-T (FT) Prompt":[""]*8,
"mDeBERTa (FT) F1":[81.45, 73.22, 51.34, 68.34, 51.56, 79.71, 80.39, 92.72],
"mDeBERTa (FT) Prompt":[""]*8,
}
functional_data = { "Lang": ["es", "pt", "hi", "ar", "fr", "it", "de"],
                    "BloomZ Prompt": ["Definition", "Definition", "Role Play", "Definition", "CoT", "Role Play", "Role Play"],
                    "BloomZ F1": [64.88, 66.04, 51.99, 62.08, 63.34, 55.15, 51.75],
                    "Aya101 Prompt": ["Distinction", "Distinction", "Distinction", "Vanilla", "Distinction", "Distinction", "Distinction"],
                    "Aya101 F1": [73.19, 72.39, 65.95, 62.99, 71.94, 71.25, 72.64],
                    "Llama3 Zero-shot Prompt": ["Vanilla", "Classification", "Classification", "Impact", "Vanilla", "Role Play", "Classification"],
                    "Llama3 Zero-shot F1": [86.37, 83.37, 65.31, 64.00, 84.61, 79.72, 85.86],
                    "Llama3 Few-shot Prompt": ["5 shot", "3 shot", "1 shot + Cultural", "1 shot", "5 shot + Role Play", "5 shot + COT", "5 shot + Cultural"],
                    "Llama3 Few-shot F1": [86.45, 86.59, 65.36, 67.95, 84.37, 87.08, 89.65],
                    "Qwen Zero-shot Prompt": ["Vanilla", "CoT", "Definition", "Vanilla", "Vanilla", "Target", "Impact"],
                    "Qwen Zero-shot F1": [84.39, 82.15, 65.41, 70.42, 82.06, 78.35, 82.64],
                    "Qwen Few-shot Prompt": ["5 shot + Definition", "5 shot + Definition", "1 shot + Definition", "3 shot + Definition", "5 shot + Definition", "5 shot + Definition", "5 shot + Definition"],
                    "Qwen Few-shot F1": [86.43, 84.08, 66.61, 71.88, 86.08, 84.17, 86.62] ,

                    "XLM-T (FT) F1": [67.93, 57.28, 23.26, 25.47, 26.61, 52.05, 70.60],
                    "XLM-T (FT) Prompt": [""]*7,
                    "mDeBERTa (FT) F1": [60.94, 58.94, 24.91, 23.93, 25.89, 54.07, 74.36],
                    "mDeBERTa (FT) Prompt": [""]*7,
                    }

df = pd.DataFrame(functional_data)

# Language mapping
lang_map = {
    "es": "Spanish", "pt": "Portuguese", "hi": "Hindi", "ar": "Arabic",
    "fr": "French", "it": "Italian", "de": "German", "tr": "Turkish"
}
df["Language"] = df["Lang"].map(lang_map)
languages = df["Language"].tolist()
x = np.arange(len(languages))

# Model keys and colors
models = ["BloomZ", "Aya101", "Llama3 Zero-shot", "Llama3 Few-shot", "Qwen Zero-shot", "Qwen Few-shot", "XLM-T (FT)", "mDeBERTa (FT)"]
colors = {
    "BloomZ": "#005293",
    "Aya101": "#87BFFF",
    "Llama3 Zero-shot": "#6CA0DC",
    "Llama3 Few-shot": "#FF8C42",
    "Qwen Zero-shot": "#A2C4F2",
    "Qwen Few-shot": "#FFBC7D",
    "XLM-T (FT)": "#3CB371",
    "mDeBERTa (FT)": "#77DD77"
}

# Spacing setup
bar_width = 0.10
offsets = np.linspace(-bar_width * 3.5, bar_width * 3.5, len(models))

# Create figure
plt.figure(figsize=(20, 6))

# Plot each model
for idx, (model, offset) in enumerate(zip(models, offsets)):
    f1_vals = []
    prompts = []
    for i in range(len(df)):
        val = df.loc[i, model + " F1"]
        prompt = df.loc[i, model + " Prompt"]
        f1_vals.append(float(val) if pd.notna(val) else 0)
        prompts.append(prompt if pd.notna(val) else "")

    x_pos = x + offset
    bars = plt.bar(x_pos, f1_vals, width=bar_width, label=model, color=colors[model])

    # Annotate each bar with prompt
    for i, bar in enumerate(bars):
        if prompts[i]:
            plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 1.5, prompts[i],
                     ha='center', va='bottom', fontsize=11, rotation=90)


# Final plot formatting
plt.xticks(x, languages, fontsize=14)
plt.xlabel("Language", fontsize=14)
plt.ylabel("F1-Macro Score", fontsize=14)
plt.ylim(20, 110)
# plt.legend(title=None, fontsize=12, loc='lower left', bbox_to_anchor=(0, 1), ncol=8)


# Title-like label inside the plot
plt.text(0.5, 0.95, "Functional Test Set", ha='center', va='center',
         transform=plt.gca().transAxes, fontsize=18, fontweight='bold')

plt.tight_layout()
plt.savefig('figures/functional_test_set.svg', dpi=300, bbox_inches='tight')

plt.show()
