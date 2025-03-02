import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def compute_language_accuracy(evaluated_df):
    """
    Computes average accuracy (in percentage) per language.
    It searches for columns that contain "_correct" and uses the substring before that.
    """
    lang_acc = {}
    for col in evaluated_df.columns:
        if "_correct" in col:
            lang = col.split("_correct")[0]
            acc = evaluated_df[col].astype(int).mean() * 100
            lang_acc.setdefault(lang, []).append(acc)
    for lang in lang_acc:
        lang_acc[lang] = sum(lang_acc[lang]) / len(lang_acc[lang])
    return lang_acc

# Set the base directory where your results are stored.
base_dir = 'results/name_cloze copy'

# We will accumulate aggregated accuracy per experiment per model.
# Structure: aggregate[experiment][model] = {lang: avg_accuracy, ...}
aggregate_normal = {}  # For files in evaluation/
aggregate_2024 = {}    # For files in evaluation/2024

# List model directories (each folder directly under base_dir)
model_dirs = sorted([os.path.join(base_dir, d) for d in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, d))])

# Loop over each model folder
for model_dir in model_dirs:
    model = os.path.basename(model_dir)
    # List experiment directories inside each model folder
    experiment_dirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir)
                       if os.path.isdir(os.path.join(model_dir, d))]
    for exp_dir in experiment_dirs:
        experiment = os.path.basename(exp_dir)
        # Build path to the normal evaluation folder
        eval_dir = os.path.join(exp_dir, "evaluation")
        if not os.path.isdir(eval_dir):
            continue  # skip if no evaluation folder

        # Process normal eval CSVs (in evaluation/ but not in a subfolder)
        eval_files = [os.path.join(eval_dir, f) for f in os.listdir(eval_dir)
                      if f.endswith('.csv')]

        if eval_files:
            per_file_acc = []
            for eval_file in eval_files:
                try:
                    df = pd.read_csv(eval_file)
                except Exception as e:
                    print(f"Error reading {eval_file}: {e}")
                    continue
                acc_dict = compute_language_accuracy(df)
                per_file_acc.append(acc_dict)
            if per_file_acc:
                avg_acc = {}
                langs = set()
                for d in per_file_acc:
                    langs.update(d.keys())
                for lang in langs:
                    values = [d.get(lang) for d in per_file_acc if lang in d]
                    if values:
                        avg_acc[lang] = sum(values) / len(values)
                aggregate_normal.setdefault(experiment, {})[model] = avg_acc

        # Process 2024 eval CSVs (if the evaluation folder has a subfolder "2024")
        eval_2024_dir = os.path.join(eval_dir, "2024")
        if os.path.isdir(eval_2024_dir):
            eval_files_2024 = [os.path.join(eval_2024_dir, f) for f in os.listdir(eval_2024_dir)
                               if f.endswith('_eval.csv')]
            if eval_files_2024:
                per_file_acc = []
                for eval_file in eval_files_2024:
                    try:
                        df = pd.read_csv(eval_file)
                    except Exception as e:
                        print(f"Error reading {eval_file}: {e}")
                        continue
                    acc_dict = compute_language_accuracy(df)
                    per_file_acc.append(acc_dict)
                if per_file_acc:
                    avg_acc = {}
                    langs = set()
                    for d in per_file_acc:
                        langs.update(d.keys())
                    for lang in langs:
                        values = [d.get(lang) for d in per_file_acc if lang in d]
                        if values:
                            avg_acc[lang] = sum(values) / len(values)
                    # Save only en and en_shuffled for the 2024 group.
                    filtered_avg = {lang: acc for lang, acc in avg_acc.items() if lang in ['en', 'en_shuffled', 'en_masked', 'en_masked_shuffled']}
                    aggregate_2024.setdefault(experiment, {})[model] = filtered_avg

# Function to create and save a heatmap given an aggregate dictionary.
def create_aggregate_heatmap(aggregate_dict, base_dir, filename_suffix, title_suffix, preferred_langs):
    for experiment, model_dict in aggregate_dict.items():
        # Create a DataFrame with rows = models and columns = languages.
        df_list = []
        for model, lang_dict in model_dict.items():
            df_list.append(pd.DataFrame(lang_dict, index=[model]))
        if not df_list:
            print(f"No data for experiment {experiment} with suffix {filename_suffix}. Skipping heatmap.")
            continue
        agg_df = pd.concat(df_list)
        # Check if the DataFrame is empty or has no numeric values.
        if agg_df.empty or agg_df.dropna(how="all").empty:
            print(f"No numeric data for experiment {experiment} with suffix {filename_suffix}. Skipping heatmap.")
            continue
        
        # Reorder columns: put preferred languages first then the rest in sorted order.
        all_langs = list(agg_df.columns)
        ordered_langs = [lang for lang in preferred_langs if lang in all_langs]
        remaining = sorted([lang for lang in all_langs if lang not in preferred_langs])
        final_cols = ordered_langs + remaining
        agg_df = agg_df.reindex(columns=final_cols)
        
        # Save the aggregated CSV data
        aggregate_dir = os.path.join(base_dir, "aggregate_heatmaps")
        os.makedirs(aggregate_dir, exist_ok=True)
        if filename_suffix == '2024':
            csv_filename = f"2024/aggregate_data_{experiment}_{filename_suffix}.csv"
        else:
            csv_filename = f"aggregate_data_{experiment}_{filename_suffix}.csv"
        csv_path = os.path.join(aggregate_dir, csv_filename)
        agg_df.to_csv(csv_path, index=True)
        print(f"Saved aggregate CSV for experiment {experiment}: {csv_path}")
        
        # Create the heatmap using a custom color map with vmin=0 and vmax=100.
        custom_cmap = LinearSegmentedColormap.from_list(
            'custom_bupu',
            ['#f7fcfd', '#bfd3e6', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'],
            N=256
        )
        
        plt.figure(figsize=(16, 6))
        sns.heatmap(agg_df, annot=True, fmt=".1f", cmap=custom_cmap,
                    vmin=0, vmax=100,
                    cbar_kws={"label": "Accuracy (%)"}, annot_kws={"size": 12})
        plt.xlabel("Language", fontsize=14)
        plt.ylabel("Model", fontsize=14)
        plt.title(f"Aggregate Accuracy Heatmap for Experiment: {experiment} {title_suffix}", fontsize=16)
        
        # Save the heatmap image
        if filename_suffix == '2024':
            heatmap_path = os.path.join(aggregate_dir, f"2024/aggregate_heatmap_{experiment}_{filename_suffix}.png")
        else:
            heatmap_path = os.path.join(aggregate_dir, f"aggregate_heatmap_{experiment}_{filename_suffix}.png")
        os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        print(f"Saved aggregate heatmap for experiment {experiment}: {heatmap_path}")


# Create aggregate heatmaps for normal files (all languages; preferred order as in your code)
preferred_normal = ['en', 'es', 'tr', 'vi', 'en_masked', 'es_masked', 'tr_masked', 'vi_masked','st','yo','tn','ty','mai','mg','en_shuffled','es_shuffled','tr_shuffled','vi_shuffled', 'en_masked_shuffled','tr_masked_shuffled','vi_masked_shuffled','es_masked_shuffled', 'st_shuffled','yo_shuffled','tn_shuffled','ty_shuffled','mai_shuffled','mg_shuffled']
create_aggregate_heatmap(aggregate_normal, base_dir, "normal", "", preferred_normal)

# Create aggregate heatmaps for 2024 files (only en and en_shuffled)
preferred_2024 = ['en', 'en_shuffled']
create_aggregate_heatmap(aggregate_2024, base_dir, "2024", "2024 books", preferred_2024)