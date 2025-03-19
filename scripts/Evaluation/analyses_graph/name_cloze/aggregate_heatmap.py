import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# 📂 Data location (where models are stored)
DATA_BASE_DIR = '/Users/alishasrivastava/BEAM-scripts/BEAM/results/name_cloze'

# 📂 Output location (where heatmap & CSV should be saved)
OUTPUT_DIR = '/Users/alishasrivastava/BEAM-scripts/BEAM/scripts/Evaluation/analyses_graphs/name_cloze'

# 🔹 Define language groups
LANG_GROUPS = {
    "English": ["en"],
    "Translations": ["es", "tr", "vi"],
    "Cross-lingual": ["st", "yo", "tn", "ty", "mai", "mg"]
}

def compute_language_group_accuracy(evaluated_df):
    """Computes accuracy for each language group using boolean conversion (correct → True)."""
    group_counts = {group: (0, 0) for group in LANG_GROUPS.keys()}  # Initialize counts

    print(f"🔎 Checking columns in CSV: {list(evaluated_df.columns)}")  # Debug: Show available columns

    for group, langs in LANG_GROUPS.items():
        total_correct, total_attempts = 0, 0

        for lang in langs:
            correct_col = f"{lang}_correct"
            if correct_col in evaluated_df.columns:
                # Convert 'correct' → True and 'incorrect' → False
                correct_series = evaluated_df[correct_col].astype(str).str.lower().map({"correct": True, "incorrect": False})
                correct_series = correct_series.fillna(False)  # Treat NaNs as False

                correct_count = correct_series.sum()
                attempt_count = len(correct_series)

                print(f"📊 {correct_col}: {correct_count} correct out of {attempt_count} attempts")  # Debug

                total_correct += correct_count  # Count of True values
                total_attempts += attempt_count  # Total number of attempts

        if total_attempts > 0:
            group_counts[group] = (total_correct, total_attempts)

    print(f"✅ Computed accuracy: {group_counts}")  # Debug output
    return group_counts

def create_grouped_heatmap(aggregate_dict, output_dir, filename_suffix, title_suffix):
    """Generates and saves a grouped accuracy heatmap sorted by accuracy."""
    df_list = []
    
    for model, group_dict in aggregate_dict.items():
        if not group_dict:
            continue

        final_acc = {
            group: (100.0 * corr / att) if att > 0 else 0.0
            for group, (corr, att) in group_dict.items()
        }

        df_list.append(pd.DataFrame(final_acc, index=[model]))

    if not df_list:
        print(f"❌ No data found for heatmap. Skipping.")
        return

    agg_df = pd.concat(df_list)
    agg_df = agg_df[["English", "Translations", "Cross-lingual"]]

    # Sort models by overall accuracy (average across columns)
    agg_df["Average Accuracy"] = agg_df.mean(axis=1)
    agg_df = agg_df.sort_values(by="Average Accuracy", ascending=False).drop(columns=["Average Accuracy"])

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save aggregate CSV
    csv_filename = f"os_aggregate_data_{filename_suffix}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    agg_df.to_csv(csv_path, index=True)
    print(f"📂 Saved aggregate CSV: {csv_path}")

    # Custom colormap
    custom_cmap = LinearSegmentedColormap.from_list(
        'custom_bupu',
        ['#f7fcfd', '#bfd3e6', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'],
        N=256
    )

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        agg_df, annot=True, fmt=".1f", cmap=custom_cmap,
        vmin=0, vmax=100,
        cbar_kws={"label": "Accuracy (%)"}, annot_kws={"size": 12}
    )

    plt.xlabel("Language Group", fontsize=14)
    plt.ylabel("Model (Sorted by Accuracy)", fontsize=14)
    plt.title(f"Name Cloze: One-Shot Accuracy Heatmap", fontsize=16)

    # Save heatmap
    heatmap_filename = f"os_aggregate_heatmap_{filename_suffix}.png"
    heatmap_path = os.path.join(output_dir, heatmap_filename)
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"📊 Saved heatmap: {heatmap_path}")

def main():
    """Loads evaluation data from one-shot/evaluation/ inside each model folder."""
    aggregate_results = {}

    # Iterate through all model folders inside name_cloze
    model_dirs = sorted([
        os.path.join(DATA_BASE_DIR, d)
        for d in os.listdir(DATA_BASE_DIR)
        if os.path.isdir(os.path.join(DATA_BASE_DIR, d))
    ])

    for model_dir in model_dirs:
        model = os.path.basename(model_dir)

        # Locate one-shot folder inside the model directory
        one_shot_dirs = [
            os.path.join(model_dir, d)
            for d in os.listdir(model_dir)
            if d.startswith("one-shot") and os.path.isdir(os.path.join(model_dir, d))
        ]

        for one_shot_dir in one_shot_dirs:
            eval_dir = os.path.join(one_shot_dir, "evaluation")
            if not os.path.isdir(eval_dir):
                continue  # Skip if evaluation folder is missing

            # ✅ Only collect _eval.csv files
            eval_files = [
                os.path.join(eval_dir, f)
                for f in os.listdir(eval_dir)
                if f.endswith('_eval.csv')  # 🔥 Now we're correctly selecting evaluation files!
            ]

            if eval_files:
                per_file_counts = []
                for eval_file in eval_files:
                    try:
                        df = pd.read_csv(eval_file)

                        print(f"📄 Processing {eval_file}")  # Debugging message
                        print(f"🔍 Available columns: {list(df.columns)}")  # Show available columns

                        cdict = compute_language_group_accuracy(df)
                        per_file_counts.append(cdict)

                    except Exception as e:
                        print(f"❌ Error reading {eval_file}: {e}")
                        continue

                if per_file_counts:
                    group_aggregates = {group: (0, 0) for group in LANG_GROUPS.keys()}

                    for cdict in per_file_counts:
                        for group, (corr, att) in cdict.items():
                            prev_corr, prev_att = group_aggregates[group]
                            group_aggregates[group] = (prev_corr + corr, prev_att + att)

                    aggregate_results[model] = group_aggregates

    create_grouped_heatmap(aggregate_results, OUTPUT_DIR, "grouped", "with Language Groups")

if __name__ == "__main__":
    main()
