import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ===================== USER SETTINGS =====================
BASE_PROMPT_PATH = "scripts/Prompts copy"
BASE_DIR = "results/direct_probe"

LANG_GROUPS = {
    "English": {"en"},
    "Translated": {"es", "tr", "vi"},
    "Cross-lingual": {"st", "tn", "ty", "mai", "mg", "yo"},
    "en_shuffled": {"en_shuffled", "en_masked_shuffled"},
    "trans_shuffled": {
        "es_shuffled", "es_masked_shuffled",
        "tr_shuffled", "tr_masked_shuffled",
        "vi_shuffled", "vi_masked_shuffled"
    },
    "xling_shuffled": {
        "st_shuffled", "tn_shuffled", "ty_shuffled", "mai_shuffled",
        "mg_shuffled", "yo_shuffled",
        "st_masked_shuffled", "tn_masked_shuffled", "ty_masked_shuffled",
        "mai_masked_shuffled", "mg_masked_shuffled", "yo_masked_shuffled"
    }
}

EXCLUDE_FILES = [
    "Below_Zero", "Bride", "You_Like", "First_Lie_Wins", "If_Only",
    "Just_for", "Lies_and", "Paper_Towns", "Ministry", "Paradise", "Funny_Story"
]

def find_evaluation_folders(base_dir, experiment_type):
    """
    Return (model_name, shot_type, eval_folder) for existing subfolders:
      base_dir/<model_name>/<experiment_type>_{one_shot|zero_shot}/evaluation
    """
    found = []
    shot_types = ["one_shot", "zero_shot"]
    for model_name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_path):
            continue
        for s in shot_types:
            exp_folder = f"{experiment_type}_{s}"
            exp_path = os.path.join(model_path, exp_folder)
            if os.path.isdir(exp_path):
                eval_path = os.path.join(exp_path, "evaluation")
                if os.path.isdir(eval_path):
                    found.append((model_name, s, eval_path))
    return found

def load_book_metadata(csv_path="scripts/Evaluation/analyses_graph/metadata.csv"):
    """
    CSV with columns: en_title, Copyrighted
    Returns dict => {book_title_lowercase: True/False}
    """
    if not os.path.exists(csv_path):
        print(f"Warning: No metadata CSV at {csv_path}")
        return {}
    df = pd.read_csv(csv_path)
    meta = {}
    for _, row in df.iterrows():
        title_str = str(row["en_title"]).strip().lower()
        c = bool(row["Copyrighted"])
        meta[title_str] = c
    print("Loaded book metadata with", len(meta), "entries.\n")
    return meta

def extract_book_name(filename):
    """
    E.g. 'War_and_Peace_name_cloze_gpt_eval.csv' => 'War and Peace'
    """
    base = filename.split("_eval")[0].replace("_", " ")
    return base.strip()

def load_unmasked_passages(book_name):
    """
    Optional check for unmasked CSV. If not found, you may skip or handle differently.
    """
    folder_name = book_name.replace(" ", "_")
    path = os.path.join(BASE_PROMPT_PATH, folder_name, f"{folder_name}_unmasked_passages.csv")
    if not os.path.exists(path):
        print(f"Warning: Unmasked passages not found => {path}")
        return None
    return pd.read_csv(path)

def gather_accuracies_global(experiment_type, meta_dict):
    """
    Gathers GLOBAL accuracy for each (model, shot_type, language_group, copyrighted).
    That means we sum up matched_count and total_count across all lines (not book-level averaging).
    Steps:
      1) For each CSV (representing some portion of a single book):
         - Count how many lines are 'True' => matched_count
         - total_count = number of lines
         - Accumulate those counts in records.
      2) Group by (model, shot_type, language_group, copyrighted):
         => sum(matched_count), sum(total_count)
         => final accuracy = (sum(matched_count)/ sum(total_count)) * 100
    """
    records = []
    found_folders = find_evaluation_folders(BASE_DIR, experiment_type)

    for (model_name, shot_type, eval_folder) in found_folders:
        for file in os.listdir(eval_folder):
            if not file.endswith(".csv") or 'aggregate' in file:
                continue
            if any(excl in file for excl in EXCLUDE_FILES):
                continue

            file_path = os.path.join(eval_folder, file)
            df_eval = pd.read_csv(file_path, dtype=str)

            book_name = extract_book_name(file)
            cflag = meta_dict.get(book_name.lower(), True)  # default True if missing

            # optional: skip if no unmasked found
            if load_unmasked_passages(book_name) is None:
                continue

            # For each language group, accumulate matches
            for lang_group, lang_cols in LANG_GROUPS.items():
                for lang_col in lang_cols:
                    col_name = f"{lang_col}_results_both_match"
                    if col_name in df_eval.columns:
                        match_series = df_eval[col_name].str.lower() == "true"
                        matched_count = match_series.sum()
                        total_count = len(match_series)
                        # accumulate
                        if total_count > 0:
                            records.append({
                                "model": model_name,
                                "shot_type": shot_type,
                                "language_group": lang_group,
                                "copyrighted": cflag,
                                "matched_count": matched_count,
                                "total_count": total_count
                            })

    if not records:
        # return empty df with expected columns
        return pd.DataFrame(columns=[
            "model","shot_type","language_group","copyrighted",
            "matched_count","total_count","accuracy"
        ])

    df_all = pd.DataFrame(records)

    # Now group by (model, shot_type, language_group, copyrighted) and sum matched/total
    grouped = df_all.groupby(
        ["model","shot_type","language_group","copyrighted"], as_index=False
    ).agg({"matched_count":"sum","total_count":"sum"})

    # Compute global accuracy
    grouped["accuracy"] = (grouped["matched_count"] / grouped["total_count"]) * 100.0

    # Return final DataFrame with these columns
    return grouped[[
        "model","shot_type","language_group","copyrighted","accuracy","matched_count","total_count"
    ]]

def plot_four_subplots_global(df, experiment_type):
    """
    Creates a 2x2 grid of heatmaps:
      (0,0) = one_shot + copyrighted
      (0,1) = one_shot + public
      (1,0) = zero_shot + copyrighted
      (1,1) = zero_shot + public

    Rows = model, Cols = language_group, Values = global accuracy
    """
    # Sort model names
    models_sorted = sorted(df["model"].unique())
    # Define language group order
    lang_group_order = [
        "English", "Translated", "Cross-lingual",
        "en_shuffled", "trans_shuffled", "xling_shuffled"
    ]

    # Prepare a custom colormap
    custom_cmap = LinearSegmentedColormap.from_list(
        'custom_bupu',
        ['#f7fcfd', '#bfd3e6', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'],
        N=256
    )

    shot_types = ["one_shot", "zero_shot"]
    cflags = [True, False]
    titles = {
        (True, "one_shot"): "One-shot + Copyrighted",
        (False, "one_shot"): "One-shot + Public",
        (True, "zero_shot"): "Zero-shot + Copyrighted",
        (False, "zero_shot"): "Zero-shot + Public"
    }

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))

    for i, st in enumerate(shot_types):
        for j, cf in enumerate(cflags):
            subset = df[(df["shot_type"] == st) & (df["copyrighted"] == cf)]
            # Pivot
            pivoted = subset.pivot(
                index="model", columns="language_group", values="accuracy"
            )
            # Reindex to fix row/column order
            pivoted = pivoted.reindex(index=models_sorted, columns=lang_group_order)

            ax = axes[i, j]
            sns.heatmap(
                pivoted,
                ax=ax,
                cmap=custom_cmap,
                vmin=0, vmax=100,
                annot=True, fmt=".1f",
                cbar=False  # single colorbar at the end
            )
            ax.set_title(titles[(cf, st)], fontsize=12)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    # Single colorbar on the right
    fig.subplots_adjust(right=0.88, wspace=0.3, hspace=0.3)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    # The last subplot's heatmap is axes[-1, -1], so we can reference its "QuadMesh"
    mesh = axes[-1, -1].collections[0]
    cbar = fig.colorbar(mesh, cax=cbar_ax)
    cbar.set_label("Global Accuracy (%)")

    fig.suptitle(f"Direct Probe - {experiment_type.upper()}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.88, 1])

    out_name = f"dp_heatmaps_public_vs_copyright_{experiment_type}.png"
    plt.savefig(out_name, dpi=300, bbox_inches="tight")
    print("Saved figure:", out_name)
    plt.show()

def main_global_accuracy(experiment_type):
    meta_dict = load_book_metadata()
    df_acc = gather_accuracies_global(experiment_type, meta_dict)
    if df_acc.empty:
        print(f"No data found for {experiment_type}")
        return
    plot_four_subplots_global(df_acc, experiment_type)

if __name__ == "__main__":
    for e_type in ["ne", "masked", "non_ne"]:
        main_global_accuracy(e_type)
