import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data location (where models are stored)
DATA_BASE_DIR = 'results/name_cloze'

# Output location (where the boxplot should be saved)
OUTPUT_DIR = 'scripts/Evaluation/analyses_graph/name_cloze'

# Define language groups
LANG_GROUPS = {
    "English": ["en"],
    "Translations": ["es", "tr", "vi"],
    "Cross-lingual": ["st", "yo", "tn", "ty", "mg", "mai"]
}

# Models to include in the graph
MODELS = [
    "gpt-4o-2024-11-20",
    "Llama-3.1-405b",
    "OLMo-2-1124-13B-Instruct",
    "Qwen2.5-7B-Instruct-1M",
    "EuroLLM-9B-Instruct"
]

def compute_language_group_accuracy(evaluated_df):
    """Computes accuracy for each language group by averaging accuracy across its languages."""
    group_counts = {group: (0, 0) for group in LANG_GROUPS.keys()}  # Initialize counts

    for group, langs in LANG_GROUPS.items():
        total_correct, total_attempts = 0, 0

        for lang in langs:
            correct_col = f"{lang}_correct"
            print("b")
            if correct_col in evaluated_df.columns:
                correct_series = evaluated_df[correct_col].map({'correct': 1, 'incorrect': 0})

                # correct_series = evaluated_df[correct_col].astype(int)
                total_correct += correct_series.sum()
                total_attempts += len(correct_series)

        if total_attempts > 0:
            group_counts[group] = (total_correct, total_attempts)

    return group_counts

def main():
    """Loads evaluation data and creates a boxplot for direct probing performance."""
    data = []

    # Iterate through all model folders inside direct_probe
    model_dirs = sorted([
        os.path.join(DATA_BASE_DIR, d)
        for d in os.listdir(DATA_BASE_DIR)
        if os.path.isdir(os.path.join(DATA_BASE_DIR, d)) and d in MODELS
    ])

    for model_dir in model_dirs:
        model = os.path.basename(model_dir)
        print("A")
        # Locate ne_one_shot folder inside the model directory
        ne_one_shot_dirs = [
            os.path.join(model_dir, d)
            for d in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, d))
        ]
        print(ne_one_shot_dirs)
        print("A")
        for ne_one_shot_dir in ne_one_shot_dirs:
            eval_dir = os.path.join(ne_one_shot_dir, "evaluation")
            if not os.path.isdir(eval_dir):
                print("h")
                continue  # Skip if evaluation folder is missing

            # Collect all *_eval.csv files
            eval_files = [
                os.path.join(eval_dir, f)
                for f in os.listdir(eval_dir)

            ]
            print("y"  )
            print(eval_files)
            for eval_file in eval_files:
                try:
                    print("c")
                    df = pd.read_csv(eval_file)
                except Exception as e:
                    print(f"Error reading {eval_file}: {e}")
                    continue

                cdict = compute_language_group_accuracy(df)
                print(cdict)
                for group, (corr, att) in cdict.items():
                    print("A")

                    if att > 0:
                        accuracy = 100.0 * corr / att
                        data.append({
                            "Model": model,
                            "Language Group": group,
                            "Accuracy": accuracy
                        })

    # Convert to DataFrame
    print(data)
    plot_df = pd.DataFrame(data)

    # Create the boxplot
   # Create the boxplot with transparency
    plt.figure(figsize=(8, 6))
    sns.boxplot(
    y="Language Group", x="Accuracy", data=plot_df,
    color='lightgray',  # Single box per group
    width=0.4,
    boxprops=dict(facecolor='none', edgecolor='gray', linewidth=1),
    whiskerprops=dict(color='gray', alpha=0.7),
    capprops=dict(color='gray', alpha=0.7),
    medianprops=dict(color='black', linewidth=1.5),
    flierprops=dict(marker='o', markersize=3, alpha=0.3)
)
    palette = {
        "gpt-4o-2024-11-20": "#1f77b4",
        "Llama-3.1-405b": "#ff7f0e",
        "OLMo-2-1124-13B-Instruct": "#2ca02c",
        "Qwen2.5-7B-Instruct-1M": "#d62728",
        "EuroLLM-9B-Instruct": "#9467bd"
    }


    # Add stripplot (dots)
    sns.stripplot(
        y="Language Group", x="Accuracy", data=plot_df, hue="Model",
        dodge=False, marker="o", alpha=0.9, jitter=True, linewidth=0, palette=palette
    )


    plt.ylabel("Language Group", fontsize=14)
    plt.xlabel("Accuracy (%)", fontsize=14)
    plt.title("Name Cloze Performance Across Language Group", fontsize=16)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    # Remove duplicate legends
# Remove duplicate legend entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(
    by_label.values(), by_label.keys(),
    title="Model",
    loc='upper right',  # Top-right corner *inside* the axes
    frameon=True,       # Optional: adds a box around legend
    fontsize=10,
    title_fontsize=11
)

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save the plot
    plot_filename = "nct_boxplot.png"
    plot_path = os.path.join(OUTPUT_DIR, plot_filename)
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved boxplot: {plot_path}")

if __name__ == "__main__":
    main() 