import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ===================== USER SETTINGS =====================
SCORE_TYPE = "ROUGE-L"  # or "ChrF++" or "ROUGE-L"
BASE_DIR = "results/prefix_probe"
LANG_LIST = ["en", "es", "vi", "tr"]  # or add "st","tn","ty" if you want them
EXCLUDE_FILES = ["Below_Zero", "Bride", "You_Like", "First_Lie_Wins", "If_Only",
                 "Just_for", "Lies_and", "Paper_Towns", "Ministry", "Paradise", "Funny_Story"]

# Colors / style for the heatmap
custom_cmap = LinearSegmentedColormap.from_list(
    'custom_bupu',
    ['#f7fcfd', '#bfd3e6', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'],
    N=256
)
COMBINED_HEATMAP_OUT = f"prefix_probe_heatmap_{SCORE_TYPE}_combined.png"

# ============== 1) Folder Discovery ==============
def find_evaluation_folders(base_dir):
    """
    For each model in prefix_probe, look for two experiment folders:
      - one-shot: eval/csv
      - zero-shot: eval/csv/zero-shot
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

# ============== 2) Gather the data ==============
def load_and_process_data(score_type):
    """
    1) Finds all model/eval/csv & model/eval/csv/zero-shot folders (with experiment type)
    2) For each CSV (skipping EXCLUDE_FILES), and for each language column (e.g. "en_ROUGE-L"),
       extend the aggregator with all row values.
    3) Returns aggregator as a dict:
         aggregator[experiment_type][model][lang] => list of values
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
                aggregator[exp_type][model_name][lang].extend(df[col].tolist())
    return aggregator

# ============== 3) Build DataFrame ==============
def build_dataframe(aggregator_subdict):
    """
    Given aggregator_subdict[model][lang] => list of all values (for a specific experiment type),
    compute the global average for each model and language,
    and build a DataFrame with rows=models and columns=languages.
    """
    model_names = sorted(aggregator_subdict.keys())
    df_rows = []
    for model in model_names:
        row_dict = {"Model": model}
        for lang in LANG_LIST:
            vals = aggregator_subdict[model][lang]
            row_dict[lang] = np.nanmean(vals) if vals else np.nan
        df_rows.append(row_dict)
    df = pd.DataFrame(df_rows)
    df.set_index("Model", inplace=True)
    return df

# ============== 4) Plot Combined Heatmaps ==============
def plot_heatmaps_side_by_side(df_one_shot, df_zero_shot, score_type):
    """
    Plots two heatmaps side by side with:
      - A single big title on top.
      - One common colorbar (scaled from 0 to 100) for both heatmaps.
    """
    # Create subplots that share the y-axis
    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(df_one_shot) * 0.5)))

    
    # Plot one-shot heatmap with a fixed color scale and no individual colorbar.
    hm1 = sns.heatmap(
        df_one_shot,
        annot=True, fmt=".2f",
        cmap=custom_cmap,
        vmin=0, vmax=1,
        cbar=False,
        ax=axes[0]
    )
    axes[0].set_title("Prefix Probe Heatmap - One-shot")
    axes[0].set_xlabel("Language")
    axes[0].set_ylabel("Model")
    
    # Plot zero-shot heatmap with a fixed color scale and no individual colorbar.
    hm2 = sns.heatmap(
        df_zero_shot,
        annot=True, fmt=".2f",
        cmap=custom_cmap,
        vmin=0, vmax=1,
        cbar=False,
        ax=axes[1]
    )
    axes[1].set_title("Prefix Probe Heatmap - Zero-shot")
    axes[1].set_xlabel("Language")
    axes[1].set_ylabel("")
    
    # Add one big title on top of the entire figure.
    fig.suptitle(f"Prefix Probe Heatmap ({SCORE_TYPE}) score", fontsize=16, y=0.95)
    
    # Adjust layout to make space for the title and common colorbar.
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.subplots_adjust(wspace=0.9)
    
    # Create one common colorbar for both heatmaps.
    cbar = fig.colorbar(
        hm1.collections[0],
        ax=axes.ravel().tolist(),
        orientation="vertical",
        fraction=0.05,
        pad=0.04
    )
    cbar.set_label(f"{score_type} Score")
    
    plt.savefig(COMBINED_HEATMAP_OUT, dpi=300, bbox_inches="tight")
    print(f"Saved combined heatmap to {COMBINED_HEATMAP_OUT}")
    plt.show()

def main():
    # Load aggregator with separated one-shot and zero-shot data.
    aggregator = load_and_process_data(SCORE_TYPE)
    
    # Build dataframes for each experiment type.
    df_one_shot = build_dataframe(aggregator["one-shot"])
    df_zero_shot = build_dataframe(aggregator["zero-shot"])
    
    print("Aggregated DataFrame - One-shot:\n", df_one_shot)
    print("Aggregated DataFrame - Zero-shot:\n", df_zero_shot)
    
    # Plot heatmaps side by side.
    plot_heatmaps_side_by_side(df_one_shot, df_zero_shot, SCORE_TYPE)

if __name__ == "__main__":
    main()
