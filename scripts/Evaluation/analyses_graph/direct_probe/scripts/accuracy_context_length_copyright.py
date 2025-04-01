import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mc

# ===================== USER SETTINGS =====================
BASE_PROMPT_PATH = "scripts/Prompts copy"
BASE_DIR = "results/direct_probe"

# We'll do the same language grouping and token buckets as before
LANG_GROUPS = {
    "English": {"en"},
    "Translated": {"es", "tr", "vi"},
    "Cross-lingual": {"st", "tn", "ty", "mai", "mg", "yo"},
}

TOKEN_BUCKETS = [(0, 50), (50, 100), (100, 150), (150, 250), (250, 400), (400, float("inf"))]

# If a CSV filename has these substrings, skip it
EXCLUDE_FILES = [
    "Below_Zero", "Bride", "You_Like", "First_Lie_Wins", "If_Only",
    "Just_for", "Lies_and", "Paper_Towns", "Ministry", "Paradise", "Funny_Story"
]

# Color palette per language group
FLARE_COLORS = {
    "English": "#FB5607",      # Orange
    "Translated": "#FF006E",   # Pink
    "Cross-lingual": "#8338EC" # Purple
}

# Utility: lighten a color
def lighten_color(color, amount=0.4):
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = mc.to_rgb(c)
    white = (1,1,1)
    return tuple((1-amount)*comp + amount*white_comp
                 for comp,white_comp in zip(c,white))

def get_color_and_style(base_color, is_copyright):
    """
    If copyrighted => original color + solid line,
    Else => lighten color + dashed line.
    """
    if is_copyright:
        return (base_color, "solid")
    else:
        lighter = lighten_color(base_color, 0.4)
        return (lighter, "dashed")

# ============== Part 1: Folder Discovery ==============
def find_evaluation_folders(base_dir, experiment_type):
    """
    For each ModelName in direct_probe, we look for subfolders:
      - {experiment_type}_one_shot/evaluation
      - {experiment_type}_zero_shot/evaluation
    Returns a list of those evaluation folder paths.
    """
    found = []
    # e.g. results/direct_probe/<ModelName>/<experiment_type>_one_shot/evaluation
    # or <experiment_type>_zero_shot/evaluation
    for model_name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_path):
            continue
        for shot_type in ["one_shot","zero_shot"]:
            exp_folder = f"{experiment_type}_{shot_type}"
            exp_path = os.path.join(model_path, exp_folder)
            if os.path.isdir(exp_path):
                eval_path = os.path.join(exp_path, "evaluation")
                if os.path.isdir(eval_path):
                    found.append(eval_path)
    return found

# ============== Tiktoken for token counting ==============
import tiktoken
tokenizer = tiktoken.get_encoding("o200k_base")

def get_token_count(text):
    if pd.isna(text) or not isinstance(text, str):
        return 0
    return len(tokenizer.encode(text))

# ============== Book Metadata for Copyright ==============
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
    print("Loaded book metadata with", len(meta), "entries.")
    print()
    return meta

# ============== 2) Data Aggregation ==============
def extract_book_name(filename):
    # everything before "_name_cloze"
    # e.g. "War_and_Peace_name_cloze_gpt_eval.csv" => "War and Peace"
    base = filename.split("_eval")[0].replace("_"," ").strip()
    return base

def load_unmasked_passages(book_name):
    """
    We'll gather text from the unmasked_passages CSV for token counting
    e.g. "scripts/Prompts copy/War_and_Peace/War_and_Peace_unmasked_passages.csv"
    """
    folder_name = book_name.replace(" ","_")
    unmasked_path = os.path.join(BASE_PROMPT_PATH, folder_name, f"{folder_name}_unmasked_passages.csv")
    if not os.path.exists(unmasked_path):
        print(f"Warning: Unmasked passages not found => {unmasked_path}")
        return None
    return pd.read_csv(unmasked_path)

def load_and_process_data(experiment_type, meta_dict):
    """
    1) finds all relevant evaluation folders
    2) for each CSV, parse the {lang}_results_both_match => True/False
    3) load the unmasked passages file => token count
    4) build aggregator data
    """
    eval_folders = find_evaluation_folders(BASE_DIR, experiment_type)
    all_data = []

    for folder_path in eval_folders:
        for file in os.listdir(folder_path):
            if not file.endswith(".csv") or 'aggregate' in file:
                continue
            if any(excl in file for excl in EXCLUDE_FILES):
                continue
            # parse the CSV
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)

            book_name = extract_book_name(file)
            key_ = book_name.lower()
            if key_ in meta_dict:
                cflag = meta_dict[key_]
            else:
                print(f"Book '{book_name}' not in metadata => default Copyright")
                cflag = True

            # load unmasked passages => token counting
            unmasked_df = load_unmasked_passages(book_name)
            if unmasked_df is None:
                continue

            # For each language group, see if we have e.g. "en_results_both_match"
            for lang_group, lang_cols in LANG_GROUPS.items():
                for lang_col in lang_cols:
                    match_col = f"{lang_col}_results_both_match"  # True/False
                    if match_col in df.columns and lang_col in unmasked_df.columns:
                        # interpret "True"/"False" or "true"/"false"
                        is_match = df[match_col].astype(str).str.lower() == "true"
                        token_counts = unmasked_df[lang_col].apply(get_token_count)
                        temp_df = pd.DataFrame({
                            "language_group": lang_group,
                            "tokens": token_counts,
                            "match_found": is_match,
                            "copyrighted": cflag
                        })
                        all_data.append(temp_df)

    if not all_data:
        return pd.DataFrame(columns=["language_group","tokens","match_found","copyrighted"])
    return pd.concat(all_data, ignore_index=True)

# ============== 3) Bucket-based Accuracy ==============
def compute_accuracy(data):
    results = []
    for (min_tok,max_tok) in TOKEN_BUCKETS:
        # subset rows
        subset = data[(data["tokens"]>=min_tok)&(data["tokens"]<max_tok)]
        label = f"{min_tok}-{int(max_tok) if max_tok != float('inf') else '400+'}"
        if len(subset)>0:
            acc = subset["match_found"].mean()*100
            results.append((label,acc))
        else:
            results.append((label,np.nan))
    return results

# ============== 4) Plotting ==============
def main(EXPERIMENT_TYPE):
    meta_dict = load_book_metadata()

    SHOT_TYPES = ["one_shot", "zero_shot"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    x_positions = np.arange(len(TOKEN_BUCKETS))

    def parse_colname(col):
        parts = col.rsplit("_", 1)
        grp = parts[0]
        cflag = (parts[1] == "C")
        return (grp, cflag)

    for i, shot_type in enumerate(SHOT_TYPES):
        ax = axes[i]
        aggregator = {}

        # Load and filter data specific to shot_type
        eval_folders = find_evaluation_folders(BASE_DIR, EXPERIMENT_TYPE)
        eval_folders = [f for f in eval_folders if f"{EXPERIMENT_TYPE}_{shot_type}" in f]
        all_data = []

        for folder_path in eval_folders:
            for file in os.listdir(folder_path):
                if not file.endswith(".csv") or 'aggregate' in file:
                    continue
                if any(excl in file for excl in EXCLUDE_FILES):
                    continue

                file_path = os.path.join(folder_path, file)
                df = pd.read_csv(file_path)
                book_name = extract_book_name(file)
                key_ = book_name.lower()
                cflag = meta_dict.get(key_, True)  # Default to Copyright

                unmasked_df = load_unmasked_passages(book_name)
                if unmasked_df is None:
                    continue

                for lang_group, lang_cols in LANG_GROUPS.items():
                    for lang_col in lang_cols:
                        match_col = f"{lang_col}_results_both_match"
                        if match_col in df.columns and lang_col in unmasked_df.columns:
                            is_match = df[match_col].astype(str).str.lower() == "true"
                            token_counts = unmasked_df[lang_col].apply(get_token_count)
                            temp_df = pd.DataFrame({
                                "language_group": lang_group,
                                "tokens": token_counts,
                                "match_found": is_match,
                                "copyrighted": cflag
                            })
                            all_data.append(temp_df)

        if not all_data:
            print(f"No data for {shot_type}")
            continue

        aggregated_data = pd.concat(all_data, ignore_index=True)
        grouped = aggregated_data.groupby(["language_group", "copyrighted"])

        # Compute accuracy bucket-wise
        for (grp, cflag), subset in grouped:
            acc_list = compute_accuracy(subset)
            aggregator[(grp, cflag)] = acc_list

        # Build result dataframe
        bucket_labels = [
            f"{min_}-{int(max_) if max_ != float('inf') else '400+'}"
            for (min_, max_) in TOKEN_BUCKETS
        ]
        result_df = pd.DataFrame(index=bucket_labels)
        for key, val in aggregator.items():
            grp, cflag = key
            col_name = f"{grp}_{'C' if cflag else 'NC'}"
            val_map = dict(val)
            result_df[col_name] = [val_map.get(lab, np.nan) for lab in bucket_labels]

        # Plot
        for col in result_df.columns:
            grp, cflag = parse_colname(col)
            base_color = FLARE_COLORS.get(grp, "gray")
            color_, style_ = get_color_and_style(base_color, cflag)
            y_vals = result_df[col]
            label_ = f"{grp} ({'Copyright' if cflag else 'Public'})"
            ax.plot(x_positions, y_vals,
                    color=color_, linestyle=style_,
                    marker="o", linewidth=2, label=label_)
            for x, y in zip(x_positions, y_vals):
                if not np.isnan(y):
                    ax.text(x, y + 0.5, f"{y:.1f}%", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(bucket_labels)
        ax.set_xlabel("Context Length (Tokens)")
        if i == 0:
            ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{shot_type.replace('_', ' ').title()}")

        ax.grid(True)

    # Legend (single, unified)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(),
           loc="upper center", bbox_to_anchor=(0.5, -0.01),
           ncol=2, frameon=True)

    fig.suptitle(f"Direct Probe ({EXPERIMENT_TYPE}) - Accuracy vs. Context Length", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_name = f"direct_probe_{EXPERIMENT_TYPE}_accuracy_vs_context_length_by_shot_type.png"
    fig.savefig(out_name, dpi=300, bbox_inches="tight")
    print("Saved figure:", out_name)
    plt.show()

# ============== Helper for color/style ==============
def get_color_and_style(base_color, is_copyright):
    """
    If copyrighted => base color, solid
    if public => lighten color, dashed
    """
    if is_copyright:
        return (base_color,"solid")
    else:
        return (lighten_color(base_color,0.4),"dashed")

if __name__ == "__main__":
    for e in ["ne","masked","non_ne"]:
        main(e)
