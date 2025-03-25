import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ===================== USER SETTINGS =====================
BASE_DIR = "results/prefix_probe"
LANG_LIST = ["en", "es", "vi", "tr"]
EXCLUDE_FILES = [
    "Below_Zero", "Bride", "You_Like", "First_Lie_Wins", "If_Only",
    "Just_for", "Lies_and", "Paper_Towns", "Ministry", "Paradise", "Funny_Story"
]
SCORE_TYPES = ["BLEU", "ChrF++", "ROUGE-L"]  # We want 3 heatmaps, one for each

# We assume BLEU & ChrF++ are stored 0–100 in the CSV, 
# but ROUGE-L is 0–1 and must be multiplied by 100.

# Colors / style for the heatmap
custom_cmap = LinearSegmentedColormap.from_list(
    'custom_bupu',
    ['#f7fcfd', '#bfd3e6', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'],
    N=256
)
HEATMAP_OUT = "prefix_probe_heatmap_ALL_METRICS.png"

# ============== 1) Folder Discovery ==============
def find_evaluation_folders(base_dir):
    """
    Looks for 'eval/csv' (one-shot) and 'eval/csv/zero-shot' (zero-shot)
    within each model's folder.
    
    Returns a list of tuples: (model_name, folder_path, experiment_type)
    """
    found = []
    for model_name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_path):
            continue
        # one-shot folder: eval/csv
        eval_path = os.path.join(model_path, "eval", "csv")
        if os.path.isdir(eval_path):
            found.append((model_name, eval_path, "one-shot"))
        # zero-shot folder: eval/csv/zero-shot
        zero_path = os.path.join(model_path, "eval", "csv", "zero-shot")
        if os.path.isdir(zero_path):
            found.append((model_name, zero_path, "zero-shot"))
    return found

# ============== 2) Gather the data for ONE metric ==============
def load_and_process_data_for_metric(score_type):
    """
    1) Finds all one-shot and zero-shot folders (with experiment type).
    2) For each CSV, skipping EXCLUDE_FILES, and for each language column (e.g. "en_ROUGE-L"),
       gathers *all row values* into aggregator[experiment_type][model][lang].
    3) Returns aggregator => aggregator[exp_type][model][lang] = [all values].
    """
    aggregator = {"one-shot": {}, "zero-shot": {}}
    folders = find_evaluation_folders(BASE_DIR)
    for (model_name, folder_path, exp_type) in folders:
        if model_name not in aggregator[exp_type]:
            aggregator[exp_type][model_name] = {lang: [] for lang in LANG_LIST}
        for file in os.listdir(folder_path):
            if not file.endswith(".csv"):
                continue
            if any(excl in file for excl in EXCLUDE_FILES):
                continue
            csv_path = os.path.join(folder_path, file)
            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"Error reading {csv_path}: {e}")
                continue
            
            for lang in LANG_LIST:
                col = f"{lang}_{score_type}"
                if col not in df.columns:
                    continue
                # Extend with ALL values in that column (global, not per-model average)
                aggregator[exp_type][model_name][lang].extend(df[col].tolist())
    return aggregator

# ============== 3) Build a DF with rows = LANG_LIST, columns = [one-shot, zero-shot] ==============
def build_lang_vs_experiment_df(aggregator, score_type):
    """
    aggregator is aggregator[exp_type][model][lang] => list of values
    We want a small DataFrame with:
        index = LANG_LIST,
        columns = ["one-shot", "zero-shot"]
    Each cell = global average across *all* models for that language, for that experiment type.
    
    If score_type == "ROUGE-L", we multiply by 100 to get a 0..100 scale.
    """
    exp_types = ["one-shot", "zero-shot"]
    
    # Prepare an empty DataFrame: rows=LANG_LIST, columns=["one-shot", "zero-shot"]
    df = pd.DataFrame(index=LANG_LIST, columns=exp_types, dtype=float)
    
    for exp_type in exp_types:
        # aggregator[exp_type][model][lang] = list
        # We want the global average across *all models* for each lang.
        all_models = aggregator[exp_type].keys()  # list of model names
        for lang in LANG_LIST:
            # gather all model lists for this lang
            combined_list = []
            for m in all_models:
                combined_list.extend(aggregator[exp_type][m][lang])
            val = np.nanmean(combined_list) if len(combined_list) > 0 else np.nan
            # If it's ROUGE-L, scale by 100
            if score_type == "ROUGE-L" and not np.isnan(val):
                val *= 100.0
            df.loc[lang, exp_type] = val
    
    return df

# ============== 4) Plot the 3 heatmaps side by side ==============
def plot_three_heatmaps():
    """
    1. For each of the 3 metrics in SCORE_TYPES, we:
       - gather aggregator data
       - build a (lang x experiment) DF with global means
       - plot a heatmap (subplots with sharey)
    2. We'll have a single figure with 1 row x 3 columns, each column for one metric.
    3. A single common colorbar from 0..100
    """
    # Create a figure with 3 subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)
    
    for i, metric in enumerate(SCORE_TYPES):
        aggregator = load_and_process_data_for_metric(metric)
        df_plot = build_lang_vs_experiment_df(aggregator, metric)  # shape: 4 x 2
        # Plot heatmap with vmin=0, vmax=100, no colorbar
        hm = sns.heatmap(
            df_plot,
            annot=True, fmt=".2f",
            cmap=custom_cmap,
            vmin=0, vmax=100,
            cbar=False,  # We'll add one common colorbar after
            ax=axes[i]
        )
        # Title = metric name
        axes[i].set_title(metric)
        # x-axis label = "Experiment Setting"
        axes[i].set_xlabel("Experiment Setting")
        # y-axis label = "Language" only on the first heatmap
        if i == 0:
            axes[i].set_ylabel("Language")
        else:
            axes[i].set_ylabel("")
        
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space for colorbar to the right
    
    # Add a single colorbar, using the last heatmap's mappable or the first one
    cbar = fig.colorbar(
        axes[0].collections[0],  # mappable from the first heatmap
        ax=axes.ravel().tolist(),
        orientation="vertical",
        fraction=0.04,
        pad=0.02
    )
    cbar.set_label("Score (0-100)")
    
    fig.suptitle("Global Average Scores by Language", fontsize=14, y=1.02)
    plt.savefig(HEATMAP_OUT, dpi=300, bbox_inches="tight")
    plt.show()

def main():
    plot_three_heatmaps()

if __name__ == "__main__":
    main()