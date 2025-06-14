import matplotlib.pyplot as plt
import seaborn as sns

def plot_f1_vs_train_size(
    train_sizes,
    real_zero_shot,
    real_few_shot,
    real_xlmt,
    func_zero_shot,
    func_few_shot,
    func_xlmt,
    title="F1-Macro vs Train Size",
    ylabel="Macro F1",
    xlabel="Train Size",
    ylim=(20, 90),
    save_path=None
):
    """
    Plots F1-macro vs training size for two test groups with zero-shot, few-shot, and XLM-T results.

    Parameters:
        train_sizes (list): X-axis values (training sizes).
        real_zero_shot, real_few_shot, real_xlmt (list): Real group performance.
        func_zero_shot, func_few_shot, func_xlmt (list): Functional group performance.
        title, xlabel, ylabel (str): Axis labels and plot title.
        ylim (tuple): Y-axis limits.
        save_path (str): Path to save the plot image (e.g. "plot.png"). If None, just displays.
    """
    sns.set(style="darkgrid", context="paper", palette="muted", font_scale=1.2)
    plt.figure(figsize=(10, 6))

    # Real group - blue
    sns.lineplot(x=train_sizes, y=real_zero_shot, label='Real: Zero-Shot', linestyle='--', linewidth=2, color='tab:blue')
    sns.lineplot(x=train_sizes, y=real_few_shot, label='Real: Few-Shot', linestyle='-.', linewidth=2, color='tab:blue')
    sns.lineplot(x=train_sizes, y=real_xlmt, label='Real: XLM-T', linestyle='-', marker='o', linewidth=2, color='tab:blue')

    # Functional group - orange
    sns.lineplot(x=train_sizes, y=func_zero_shot, label='Func: Zero-Shot', linestyle='--', linewidth=2, color='tab:orange')
    sns.lineplot(x=train_sizes, y=func_few_shot, label='Func: Few-Shot', linestyle='-.', linewidth=2, color='tab:orange')
    sns.lineplot(x=train_sizes, y=func_xlmt, label='Func: XLM-T', linestyle='-', marker='s', linewidth=2, color='tab:orange')

    plt.title(title, fontsize=14)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(0, max(train_sizes) + 100)
    plt.ylim(*ylim)
    plt.legend(title='Model', fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()


plot_f1_vs_train_size(
    train_sizes=[10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000, 2000],
    real_zero_shot=[64.79]*12,
    real_few_shot=[68.9]*12,
    real_xlmt=[36.51, 49.91, 54.38, 57.63, 61.85, 65.03, 72.36, 74.40, 76.58, 77.14, 79.35, 81.08],
    func_zero_shot=[86.37]*12,
    func_few_shot=[87.4]*12,
    func_xlmt=[30.27, 36.68, 34.35, 26.70, 23.22, 30.50, 39.51, 50.89, 53.38, 55.28, 61.29, 63.92],
    title="Macro-F1 vs Training Set Size (Real vs Functional)",
    # save_path="f1_macro_clean_final.png"
)
