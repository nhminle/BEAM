import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc

# ===================== USER SETTINGS (Reused) =====================
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

FLARE_COLORS = {
    "English": "#FB5607",      # Orange
    "Translated": "#FF006E",   # Pink
    "Cross-lingual": "#8338EC" # Purple
}

def lighten_color(color, amount=0.4):
    """Lighten the given color by mixing with white."""
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = mc.to_rgb(c)
    white = (1,1,1)
    return tuple((1 - amount) * comp + amount * wcomp for comp, wcomp in zip(c, white))

def get_color_and_style(base_color, is_copyright):
    """
    If copyrighted => base color, solid.
    If public => lighten color, dashed.
    (We may or may not need this in a heatmap scenario.)
    """
    if is_copyright:
        return (base_color, "solid")
    else:
        return (lighten_color(base_color, 0.4), "dashed")

# ============== Part 1: Folder Discovery ==============
def find_evaluation_folders(base_dir, experiment_type):
    """
    For each ModelName in direct_probe, we look for subfolders:
      - {experiment_type}_one_shot/evaluation
      - {experiment_type}_zero_shot/evaluation
    Returns a list of (model_name, shot_type, eval_folder_path).
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

# ============== 2) Book Metadata for Copyright ==============
def load_book_metadata(csv_path="scripts/Evaluation/analyses_graph/metadata.csv"):
    """
    CSV with columns: en_title, Copyrighted
    Returns a dict => { "title": True/False }
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

# ============== 3) Extract Book Name from CSV ==============
def extract_book_name(filename):
    """
    e.g. 'War_and_Peace_name_cloze_gpt_eval.csv' => 'War and Peace'
    We'll just remove anything after '_eval', then replace underscores with spaces.
    """
    base = filename.split("_eval")[0].replace("_", " ").strip()
    return base

# ============== 4) Attempt to load unmasked passages (Optional) ==============
def load_unmasked_passages(book_name):
    """
    If you want to confirm the passage is valid or do extra checks. 
    You can omit token counting if not needed.
    """
    folder_name = book_name.replace(" ", "_")
    unmasked_path = os.path.join(BASE_PROMPT_PATH, folder_name, f"{folder_name}_unmasked_passages.csv")
    if not os.path.exists(unmasked_path):
        print(f"Warning: Unmasked passages not found => {unmasked_path}")
        return None
    return pd.read_csv(unmasked_path)

# ============== 5) Gather accuracies ignoring token buckets ==============
def gather_accuracies(experiment_type, meta_dict):
    """
    1) Find evaluation folders for each (model, shot_type).
    2) For each evaluation CSV inside them (skipping EXCLUDE_FILES):
       - Determine the book name => check metadata for copyright flag.
       - For each language group, gather whether 'both_match' is True/False.
       - Collect (model, shot_type, lang_group, cflag, match_found).
    3) Group by (model, shot_type, lang_group, cflag) => compute accuracy.
    Returns a DataFrame with columns:
        [model, shot_type, language_group, copyrighted, accuracy]
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
            df_eval = pd.read_csv(file_path, dtype=str)  # read all as str to unify 'True','False' checks

            book_name = extract_book_name(file)
            cflag = meta_dict.get(book_name.lower(), True)  # default to True if missing

            # optional: load unmasked to confirm we have it
            # but if we do NOT need token length, we can ignore the data
            unmasked_df = load_unmasked_passages(book_name)
            if unmasked_df is None:
                # skip if the unmasked CSV doesn't exist
                # or you might choose to keep going if you only care about the evaluation file
                continue

            # For each language group, check columns like: en_results_both_match
            for lang_group, lang_cols in LANG_GROUPS.items():
                # Each group might have multiple actual language columns
                for lang_col in lang_cols:
                    match_col = f"{lang_col}_results_both_match"
                    if match_col in df_eval.columns:
                        # interpret 'True','False'
                        # We'll treat "true"/"True"/"TRUE" as True
                        is_true_series = df_eval[match_col].str.lower() == "true"
                        # compute fraction that is True
                        n_total = len(is_true_series)
                        n_matched = is_true_series.sum()
                        if n_total > 0:
                            acc = n_matched / n_total
                            # store result
                            records.append({
                                "model": model_name,
                                "shot_type": shot_type,
                                "language_group": lang_group,
                                "copyrighted": cflag,
                                "accuracy": acc * 100.0
                            })

    # Combine all records
    if not records:
        # Return an empty DataFrame with expected columns
        return pd.DataFrame(columns=["model","shot_type","language_group","copyrighted","accuracy"])
    df_all = pd.DataFrame(records)
    # Possibly group again if some CSV had multiple lines for the same combo
    # (usually it’s aggregated by now, but just in case)
    grouped = df_all.groupby(["model","shot_type","language_group","copyrighted"], as_index=False)["accuracy"].mean()
    return grouped

# ============== 6) Plot Heatmaps ==============
def plot_heatmaps(df, experiment_type):
    """
    Given a DataFrame with columns [model, shot_type, language_group, copyrighted, accuracy],
    produce four subplots:
      (1) one_shot + copyrighted
      (2) one_shot + public
      (3) zero_shot + copyrighted
      (4) zero_shot + public

    Each subplot:
      - x axis: language groups
      - y axis: model names
      - cell color: accuracy
    """

    # Get unique shot_types and language_groups in sorted order
    shot_types = ["one_shot", "zero_shot"]
    lang_groups = sorted(df["language_group"].unique())
    # We'll sort model names so they appear in a consistent order
    models = sorted(df["model"].unique())

    # Create 2x2 subplots
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    # For labeling
    subplot_titles = [("one_shot", True), ("one_shot", False),
                      ("zero_shot", True), ("zero_shot", False)]
    # Indices for subplots
    # row 0 => one_shot, row 1 => zero_shot
    # col 0 => copyrighted True, col 1 => copyrighted False

    for idx, (shot_t, cflag) in enumerate(subplot_titles):
        ax = axes[idx//2, idx%2]
        subset = df[(df["shot_type"] == shot_t) & (df["copyrighted"] == cflag)]

        # Pivot so rows=Model, columns=LangGroup, values=Accuracy
        pivoted = subset.pivot_table(
            index="model", columns="language_group", values="accuracy", aggfunc="mean"
        )
        # Ensure all models/lang_groups are present
        pivoted = pivoted.reindex(index=models, columns=lang_groups)

        # Convert to numpy array
        heat_data = pivoted.values

        # Plot with imshow
        im = ax.imshow(heat_data, aspect="auto", cmap="viridis", vmin=0, vmax=100)

        # Label x-axis with language groups
        ax.set_xticks(np.arange(len(lang_groups)))
        ax.set_xticklabels(lang_groups, rotation=45, ha="right")

        # Label y-axis with model names
        ax.set_yticks(np.arange(len(models)))
        ax.set_yticklabels(models)

        # Title
        c_text = "Copyrighted" if cflag else "Public"
        ax.set_title(f"{shot_t.replace('_',' ').title()} - {c_text}")

        # Optionally overlay text with exact percentage
        for i in range(len(models)):
            for j in range(len(lang_groups)):
                val = pivoted.iloc[i, j]
                if pd.notnull(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", color="white", fontsize=9)

    fig.suptitle(f"Heatmap of Accuracy - {experiment_type.upper()}", fontsize=16)
    fig.tight_layout()

    # Add a single colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
    cbar.set_label("Accuracy (%)", rotation=90)

    # Save or show
    out_name = f"heatmap_{experiment_type}_accuracy.png"
    # plt.savefig(out_name, dpi=300, bbox_inches="tight")
    print("Saved figure:", out_name)
    plt.show()

# ============== 7) Main driver for Heatmap ==============
def main_heatmap(experiment_type):
    meta_dict = load_book_metadata()
    # Gather overall accuracies ignoring token length
    df_acc = gather_accuracies(experiment_type, meta_dict)
    if df_acc.empty:
        print(f"No data found for experiment_type={experiment_type}")
        return
    # Plot
    plot_heatmaps(df_acc, experiment_type)

# ============== 8) Example: run for each experiment type ==============
if __name__ == "__main__":
    for e_type in ["ne", "masked", "non_ne"]:
        main_heatmap(e_type)
