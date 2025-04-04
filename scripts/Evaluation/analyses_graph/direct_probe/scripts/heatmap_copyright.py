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

def gather_accuracies(experiment_type, meta_dict):
    """
    1) Walk through <experiment_type>_{one_shot, zero_shot}/evaluation
    2) Skip excluded
    3) For each CSV => parse 'both_match' columns for each language group
    4) Build DataFrame: [model, shot_type, language_group, copyrighted, accuracy]
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

            # If you want to skip if no unmasked file found:
            if load_unmasked_passages(book_name) is None:
                continue

            for lang_group, lang_cols in LANG_GROUPS.items():
                for lang_col in lang_cols:
                    col_name = f"{lang_col}_results_both_match"
                    if col_name in df_eval.columns:
                        match_series = df_eval[col_name].str.lower() == "true"
                        n_total = len(match_series)
                        if n_total > 0:
                            acc = (match_series.sum() / n_total) * 100.0
                            records.append({
                                "model": model_name,
                                "shot_type": shot_type,
                                "language_group": lang_group,
                                "copyrighted": cflag,
                                "accuracy": acc
                            })

    if not records:
        return pd.DataFrame(columns=["model","shot_type","language_group","copyrighted","accuracy"])

    df_all = pd.DataFrame(records)
    # If same book combo repeated, group again
    grouped = df_all.groupby(["model","shot_type","language_group","copyrighted"], as_index=False)["accuracy"].mean()
    return grouped

def plot_two_wide_heatmaps_seaborn(df, experiment_type):
    """
    Creates 2 subplots side by side (Copyrighted vs. Public).
    Each subplot has columns = (shot_type, language_group) in a repeated fashion.

    We use seaborn.heatmap to display the pivoted tables.
    """

    # Separate into two subsets: copyrighted / public
    df_c = df[df["copyrighted"] == True]
    df_p = df[df["copyrighted"] == False]

    # Sort rows (models)
    models_sorted = sorted(df["model"].unique())

    # We define the order for shot_type and language_group
    shot_type_order = ["one_shot", "zero_shot"]
    lang_group_order = ["English", "Translated", "Cross-lingual"]

    # Build a list of all possible column combos
    # e.g. ("one_shot", "English"), ("one_shot", "Translated"), ...
    col_combos = []
    for st in shot_type_order:
        for lg in lang_group_order:
            col_combos.append((st, lg))

    # Pivot
    pivot_c = df_c.pivot(index="model", columns=["shot_type", "language_group"], values="accuracy")
    pivot_p = df_p.pivot(index="model", columns=["shot_type", "language_group"], values="accuracy")

    # Reindex to ensure consistent row/column ordering
    pivot_c = pivot_c.reindex(index=models_sorted, columns=col_combos)
    pivot_p = pivot_p.reindex(index=models_sorted, columns=col_combos)

    # Flatten columns for nicer axis labels in seaborn
    # e.g. ("one_shot","English") => "one_shot-English"
    pivot_c.columns = [f"{st}-{lg}" for (st, lg) in pivot_c.columns]
    pivot_p.columns = [f"{st}-{lg}" for (st, lg) in pivot_p.columns]

    # Set up figure with 2 subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    # custom colormap
    custom_cmap = LinearSegmentedColormap.from_list(
        'custom_bupu',
        ['#f7fcfd', '#bfd3e6', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'],
        N=256
    )
    
    # LEFT: COPYRIGHTED
    ax_left = axes[0]
    sns.heatmap(
        pivot_c, ax=ax_left,
        cmap=custom_cmap, vmin=0, vmax=100,
        annot=True, fmt=".1f",  # display numeric labels
        cbar=False  # we'll add a single colorbar on the right subplot
    )
    ax_left.set_title("Copyrighted")
    ax_left.set_xticklabels(ax_left.get_xticklabels(), rotation=45, ha="right")

    # RIGHT: PUBLIC
    ax_right = axes[1]
    # We do cbar=True here, so we get one colorbar for the whole figure.
    # But to unify them, we might want to set cbar_ax manually. Let’s do a simpler approach:
    im = sns.heatmap(
        pivot_p, ax=ax_right,
        cmap=custom_cmap, vmin=0, vmax=100,
        annot=True, fmt=".1f",  # display numeric labels
        cbar=True, cbar_kws={"shrink": 0.8, "label": "Accuracy (%)"}
    )
    ax_right.set_title("Public")
    ax_right.set_xticklabels(ax_right.get_xticklabels(), rotation=45, ha="right")

    # Adjust layout
    fig.suptitle(f"Direct Probe - {experiment_type.upper()}", fontsize=14)
    plt.tight_layout()

    out_name = f"dp_heatmaps_public_vs_copyright_{experiment_type}.png"
    plt.savefig(out_name, dpi=300, bbox_inches="tight")
    print("Saved figure:", out_name)
    plt.show()

def main_heatmap(experiment_type):
    meta_dict = load_book_metadata()
    df_acc = gather_accuracies(experiment_type, meta_dict)
    if df_acc.empty:
        print(f"No data found for {experiment_type}")
        return
    # Plot with seaborn
    plot_two_wide_heatmaps_seaborn(df_acc, experiment_type)

if __name__ == "__main__":
    for e in ["ne", "masked", "non_ne"]:
        main_heatmap(e)
