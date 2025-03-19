import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# Data location (where models are stored)
DATA_BASE_DIR = '/Users/alishasrivastava/BEAM/results/direct_probe'

# Output location (where heatmap & CSV should be saved)
OUTPUT_DIR = '/Users/alishasrivastava/BEAM/scripts/Evaluation/analyses_graph/direct_probe/shuffled'

# Define language groups
LANG_GROUPS = {
    "English": ["en_shuffled"],
    "Translations": ["es_shuffled", "tr_shuffled", "vi_shuffled"],
    "Cross-lingual": ["st_shuffled", "yo_shuffled", "tn_shuffled", "ty_shuffled", "mai_shuffled", "mg_shuffled"]
}

def compute_language_group_accuracy(evaluated_df):
    """Computes accuracy for each language group by averaging accuracy across its languages."""
    group_counts = {group: (0, 0) for group in LANG_GROUPS.keys()}  # Initialize counts

    for group, langs in LANG_GROUPS.items():
        total_correct, total_attempts = 0, 0

        for lang in langs:
            correct_col = f"{lang}_results_both_match"
            if correct_col in evaluated_df.columns:
                correct_series = evaluated_df[correct_col].astype(int)
                total_correct += correct_series.sum()
                total_attempts += len(correct_series)

        if total_attempts > 0:
            group_counts[group] = (total_correct, total_attempts)

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
        print(f"No data found for heatmap. Skipping.")
        return

    agg_df = pd.concat(df_list)
    agg_df = agg_df[["English", "Translations", "Cross-lingual"]]

    # Sort models by overall accuracy (average across columns)
    agg_df["Average Accuracy"] = agg_df.mean(axis=1)
    agg_df = agg_df.sort_values(by="Average Accuracy", ascending=False).drop(columns=["Average Accuracy"])

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save aggregate CSV
    csv_filename = f"zs_shuffled_masked_aggregate_data_{filename_suffix}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    agg_df.to_csv(csv_path, index=True)
    print(f"Saved aggregate CSV: {csv_path}")

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
    plt.title(f"Direct Probe: Shuffled Masked Zero-Shot Accuracy", fontsize=16)

    # Save heatmap
    heatmap_filename = f"zs_shuffled_masked_aggregate_heatmap_{filename_suffix}.png"
    heatmap_path = os.path.join(output_dir, heatmap_filename)
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"Saved heatmap: {heatmap_path}")

def main():
    """Loads evaluation data from ne_masked_shot/evaluation/ inside each model folder."""
    aggregate_results = {}

    # Iterate through all model folders inside direct_probe
    model_dirs = sorted([
        os.path.join(DATA_BASE_DIR, d)
        for d in os.listdir(DATA_BASE_DIR)
        if os.path.isdir(os.path.join(DATA_BASE_DIR, d))
    ])

    for model_dir in model_dirs:
        model = os.path.basename(model_dir)

        # Locate ne_one_shot folder inside the model directory
        ne_one_shot_dirs = [
            os.path.join(model_dir, d)
            for d in os.listdir(model_dir)
            if d.startswith("masked_zero_shot") and os.path.isdir(os.path.join(model_dir, d))
        ]

        for ne_one_shot_dir in ne_one_shot_dirs:
            eval_dir = os.path.join(ne_one_shot_dir, "evaluation")
            if not os.path.isdir(eval_dir):
                continue  # Skip if evaluation folder is missing

            # Collect all *_eval.csv files
            eval_files = [
                os.path.join(eval_dir, f)
                for f in os.listdir(eval_dir)
                if f.endswith('_eval.csv')
            ]

            if eval_files:
                per_file_counts = []
                for eval_file in eval_files:
                    try:
                        df = pd.read_csv(eval_file)
                    except Exception as e:
                        print(f"Error reading {eval_file}: {e}")
                        continue

                    cdict = compute_language_group_accuracy(df)
                    per_file_counts.append(cdict)

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
