import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def compute_language_counts(evaluated_df):
    """
    Returns a dictionary of {language: (total_correct, total_attempts)}
    based on columns ending in "_results_both_match".
    Each row in those columns should be 1 (True) if correct, 0 (False) otherwise.
    """
    lang_counts = {}
    for col in evaluated_df.columns:
        if "_results_both_match" in col:
            # e.g. "en_results_both_match" => lang="en"
            lang = col.split("_results_both_match")[0]
            correct_series = evaluated_df[col].astype(int)  # 1 for correct, 0 for incorrect
            total_correct = correct_series.sum()
            total_attempts = len(correct_series)

            # Accumulate counts
            if lang not in lang_counts:
                lang_counts[lang] = (0, 0)
            prev_correct, prev_attempts = lang_counts[lang]
            lang_counts[lang] = (
                prev_correct + total_correct,
                prev_attempts + total_attempts
            )
    return lang_counts

def create_aggregate_heatmap(aggregate_dict, base_dir, filename_suffix, title_suffix, preferred_langs):
    """
    Given a dictionary of aggregated accuracies:
      aggregate_dict[experiment][model] = {language: accuracy%, ...}
    it creates CSV files and heatmaps for each experiment.
    
    :param aggregate_dict: dict of { experiment_name: { model_name: { lang: accuracy% } } }
    :param base_dir: The base results directory (e.g., 'results/direct_probe').
    :param filename_suffix: Suffix for naming output files, e.g. 'normal' or '2024'.
    :param title_suffix: Suffix to append to heatmap titles, e.g. '2024 books'.
    :param preferred_langs: A list of languages to show first in columns.
    """
    for experiment, model_dict in aggregate_dict.items():
        # Build a DataFrame with rows = models and columns = languages
        df_list = []
        for model, lang_dict in model_dict.items():
            if not lang_dict:
                continue
            df_list.append(pd.DataFrame(lang_dict, index=[model]))

        if not df_list:
            print(f"No data for experiment '{experiment}' with suffix '{filename_suffix}'. Skipping heatmap.")
            continue

        agg_df = pd.concat(df_list)

        # If the DataFrame is empty or has no numeric values, skip
        if agg_df.empty or agg_df.dropna(how="all").empty:
            print(f"No numeric data for experiment '{experiment}' with suffix '{filename_suffix}'. Skipping heatmap.")
            continue

        # Reorder columns: put preferred languages first, then the rest in sorted order
        all_langs = list(agg_df.columns)
        ordered_langs = [lang for lang in preferred_langs if lang in all_langs]
        remaining = sorted([lang for lang in all_langs if lang not in ordered_langs])
        final_cols = ordered_langs + remaining
        agg_df = agg_df.reindex(columns=final_cols)

        # Make sure there's a directory to store the outputs
        aggregate_dir = os.path.join(base_dir, "aggregate_heatmaps")
        os.makedirs(aggregate_dir, exist_ok=True)

        # Save aggregated CSV data
        if filename_suffix == '2024':
            csv_filename = f"2024/aggregate_data_{experiment}_{filename_suffix}.csv"
        else:
            csv_filename = f"aggregate_data_{experiment}_{filename_suffix}.csv"

        csv_path = os.path.join(aggregate_dir, csv_filename)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        agg_df.to_csv(csv_path, index=True)
        print(f"Saved aggregate CSV for experiment '{experiment}': {csv_path}")

        # Create the heatmap using a custom color map with vmin=0 and vmax=100
        custom_cmap = LinearSegmentedColormap.from_list(
            'custom_bupu',
            ['#f7fcfd', '#bfd3e6', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'],
            N=256
        )

        plt.figure(figsize=(16, 6))
        sns.heatmap(
            agg_df, annot=True, fmt=".1f", cmap=custom_cmap,
            vmin=0, vmax=100,
            cbar_kws={"label": "Accuracy (%)"}, annot_kws={"size": 12}
        )

        plt.xlabel("Language", fontsize=14)
        plt.ylabel("Model", fontsize=14)
        plt.title(f"Aggregate Accuracy Heatmap for Experiment: {experiment} {title_suffix}", fontsize=16)

        # Save the heatmap image
        if filename_suffix == '2024':
            heatmap_filename = f"2024/aggregate_heatmap_{experiment}_{filename_suffix}.png"
        else:
            heatmap_filename = f"aggregate_heatmap_{experiment}_{filename_suffix}.png"

        heatmap_path = os.path.join(aggregate_dir, heatmap_filename)
        os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        print(f"Saved aggregate heatmap for experiment '{experiment}': {heatmap_path}")

def main():
    # Set the base directory where your results are stored.
    base_dir = 'results/direct_probe'

    # We'll accumulate aggregated accuracy per experiment per model.
    # Structure: aggregate[experiment][model] = {lang: accuracy%}
    aggregate_normal = {}  # For files in evaluation/
    aggregate_2024 = {}    # For files in evaluation/2024

    # List model directories (each folder directly under base_dir)
    model_dirs = sorted([
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ])

    # Loop over each model folder
    for model_dir in model_dirs:
        model = os.path.basename(model_dir)
        # List experiment directories inside each model folder
        experiment_dirs = [
            os.path.join(model_dir, d)
            for d in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, d))
        ]

        for exp_dir in experiment_dirs:
            experiment = os.path.basename(exp_dir)

            # Build path to the normal evaluation folder
            eval_dir = os.path.join(exp_dir, "evaluation")
            if not os.path.isdir(eval_dir):
                continue  # skip if no evaluation folder

            # Process normal eval CSVs (in evaluation/, not in a subfolder)
            eval_files = [
                os.path.join(eval_dir, f)
                for f in os.listdir(eval_dir)
                if f.endswith('_eval.csv')
            ]
            if eval_files:
                # We'll accumulate raw counts across all these files
                per_file_counts = []
                for eval_file in eval_files:
                    try:
                        df = pd.read_csv(eval_file)
                    except Exception as e:
                        print(f"Error reading {eval_file}: {e}")
                        continue

                    # Get raw correct/attempt counts for each language
                    cdict = compute_language_counts(df)
                    per_file_counts.append(cdict)

                # Combine the raw counts into a single dictionary
                if per_file_counts:
                    lang_aggregates = {}
                    for cdict in per_file_counts:
                        for lang, (corr, att) in cdict.items():
                            if lang not in lang_aggregates:
                                lang_aggregates[lang] = (0, 0)
                            prev_corr, prev_att = lang_aggregates[lang]
                            lang_aggregates[lang] = (prev_corr + corr, prev_att + att)

                    # Compute final accuracy for each language
                    final_acc = {}
                    for lang, (total_corr, total_att) in lang_aggregates.items():
                        if total_att > 0:
                            final_acc[lang] = 100.0 * total_corr / total_att
                        else:
                            final_acc[lang] = 0.0

                    # Store in aggregate_normal
                    aggregate_normal.setdefault(experiment, {})[model] = final_acc

            # Process 2024 eval CSVs (if the evaluation folder has a subfolder "2024")
            eval_2024_dir = os.path.join(eval_dir, "2024")
            if os.path.isdir(eval_2024_dir):
                eval_files_2024 = [
                    os.path.join(eval_2024_dir, f)
                    for f in os.listdir(eval_2024_dir)
                    if f.endswith('_eval.csv')
                ]
                if eval_files_2024:
                    per_file_counts = []
                    for eval_file in eval_files_2024:
                        try:
                            df = pd.read_csv(eval_file)
                        except Exception as e:
                            print(f"Error reading {eval_file}: {e}")
                            continue

                        cdict = compute_language_counts(df)
                        per_file_counts.append(cdict)

                    if per_file_counts:
                        lang_aggregates = {}
                        for cdict in per_file_counts:
                            for lang, (corr, att) in cdict.items():
                                if lang not in lang_aggregates:
                                    lang_aggregates[lang] = (0, 0)
                                pc, pa = lang_aggregates[lang]
                                lang_aggregates[lang] = (pc + corr, pa + att)

                        # Compute final accuracy for each language
                        final_acc = {}
                        for lang, (tcorr, tatt) in lang_aggregates.items():
                            if tatt > 0:
                                final_acc[lang] = 100.0 * tcorr / tatt
                            else:
                                final_acc[lang] = 0.0

                        # Filter if you only want certain languages
                        # (en, en_shuffled, en_masked, en_masked_shuffled)
                        allowed_langs = ['en', 'en_shuffled', 'en_masked', 'en_masked_shuffled']
                        filtered_acc = {
                            lang: acc for lang, acc in final_acc.items()
                            if lang in allowed_langs
                        }

                        aggregate_2024.setdefault(experiment, {})[model] = filtered_acc

    # Create heatmaps for "normal" files
    # Choose the order you want the languages to appear
    preferred_normal = [
        'en', 'es', 'tr', 'vi',
        'en_masked', 'es_masked', 'tr_masked', 'vi_masked',
        'st', 'yo', 'tn', 'ty', 'mai', 'mg',
        'en_shuffled','es_shuffled','tr_shuffled','vi_shuffled',
        'en_masked_shuffled','tr_masked_shuffled','vi_masked_shuffled','es_masked_shuffled',
        'st_shuffled','yo_shuffled','tn_shuffled','ty_shuffled','mai_shuffled','mg_shuffled'
    ]
    create_aggregate_heatmap(
        aggregate_normal,
        base_dir,
        "normal",
        "",
        preferred_normal
    )

    # Create heatmaps for "2024" files (only en, en_shuffled, en_masked, en_masked_shuffled)
    preferred_2024 = ['en', 'en_shuffled', 'en_masked', 'en_masked_shuffled']
    create_aggregate_heatmap(
        aggregate_2024,
        base_dir,
        "2024",
        "2024 books",
        preferred_2024
    )

if __name__ == "__main__":
    main()
