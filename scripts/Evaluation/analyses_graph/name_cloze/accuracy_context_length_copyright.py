import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tiktoken
import matplotlib.colors as mc

# ============ USER SETTINGS ============

EXPERIMENT_TYPE = "zero-shot"  # or "zero-shot"
BASE_NAME = "1s" if EXPERIMENT_TYPE == "one-shot" else "0s"  # used for filenames

BASE_PROMPT_PATH = "scripts/Prompts copy"

# We'll discover folders automatically:
def find_evaluation_folders(base_dir, experiment_type):
    """
    Crawls the directory structure under `base_dir` (e.g. 'results/name_cloze')
    looking for subfolders of the form: model/experiment_type/evaluation
    Returns a list of such folder paths.
    """
    found_paths = []
    # Example structure:
    # results/name_cloze/ModelName/(one-shot or zero-shot)/evaluation
    for model_name in os.listdir(base_dir):
        model_path = os.path.join(base_dir, model_name)
        if not os.path.isdir(model_path):
            continue
        # check if there's a subdir named experiment_type
        exp_subdir = os.path.join(model_path, experiment_type)
        if not os.path.isdir(exp_subdir):
            continue
        # check if there's an "evaluation" folder inside that
        eval_folder = os.path.join(exp_subdir, "evaluation")
        if os.path.isdir(eval_folder):
            found_paths.append(eval_folder)
    return found_paths

# Language groups
LANG_GROUPS = {
    "English": {"en"},
    "Translated": {"es", "tr", "vi"},
    "Cross-lingual": {"st", "tn", "ty", "mai", "mg", "yo"},
    # "en_shuffled": {"en_shuffled", "en_masked_shuffled"},
    # "trans_shuffled": {"es_shuffled", "es_masked_shuffled", "tr_shuffled", "tr_masked_shuffled",
    #                    "vi_shuffled", "vi_masked_shuffled"},
    # "xling_shuffled": {"st_shuffled", "tn_shuffled", "ty_shuffled", "mai_shuffled", "mg_shuffled", "yo_shuffled",
    #                    "st_masked_shuffled", "tn_masked_shuffled", "ty_masked_shuffled", "mai_masked_shuffled",
    #                    "mg_masked_shuffled", "yo_masked_shuffled"}
}

# Tokenization buckets
TOKEN_BUCKETS = [(0, 50), (50, 100), (100, 150), (150, 250), (250, 400), (400, float("inf"))]

# Skip files with these words
EXCLUDE_FILES = [
    "Below_Zero", "Bride", "You_Like", "First_Lie_Wins", "If_Only",
    "Just_for", "Lies_and", "Paper_Towns", "Ministry", "Paradise", "Funny_Story"
]

# Flare color palette (for each language group)
FLARE_COLORS = {
    "English": "#FB5607",      # Bright orange
    "Translated": "#FF006E",   # Vivid pink
    "Cross-lingual": "#8338EC" # Deep purple
}

# ============ Utility: lighten a color ============

def lighten_color(color, amount=0.4):
    """
    Lightens the given color by mixing with white.
    `amount=0.0` => color is unchanged;
    `amount=1.0` => color becomes white.
    """
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = mc.to_rgb(c)
    # blend c towards white
    white = (1.0, 1.0, 1.0)
    out = tuple((1 - amount)*comp + amount*white_comp
                for comp, white_comp in zip(c, white))
    return out

# Linestyles for copyrighted vs not
def get_color_and_style(base_color, is_copyright):
    """
    Returns (color, linestyle).
    Copyright => original color, solid
    Public => lighter color, dashed
    """
    if is_copyright:
        return (base_color, "solid")
    else:
        lighter = lighten_color(base_color, amount=0.4)
        return (lighter, "dashed")


# ============ Tiktoken ============

tokenizer = tiktoken.get_encoding("o200k_base")

def get_token_count(text):
    if pd.isna(text) or not isinstance(text, str):
        return 0
    return len(tokenizer.encode(text))

# ============ 1) Load Book Metadata ============

def load_book_metadata(csv_path):
    """
    CSV with columns: en_title, Copyrighted
    Returns { "title": True/False, ... }
    """
    if not os.path.exists(csv_path):
        print(f"Warning: No metadata CSV at {csv_path}")
        return {}
    df = pd.read_csv(csv_path)
    meta = {}
    for _, row in df.iterrows():
        t = str(row["en_title"]).strip().lower()
        c = bool(row["Copyrighted"])
        meta[t] = c
    print("Loaded book metadata with", len(meta), "entries.")
    return meta

# ============ 2) Gather the data ============

def extract_book_name(filename):
    # everything before "_name_cloze"
    base = filename.split("_name_cloze")[0].replace("_", " ").strip()
    return base

def load_masked_passages(book_name):
    folder_name = book_name.replace(" ", "_")
    masked_path = os.path.join(BASE_PROMPT_PATH, folder_name, f"{folder_name}_masked_passages.csv")
    if not os.path.exists(masked_path):
        print(f"Warning: Masked passages file not found for {masked_path}")
        return None
    return pd.read_csv(masked_path)

def load_and_process_data(metadata_dict, base_dir, experiment_type):
    """
    Finds all evaluation folders automatically, loads CSVs, merges with masked passages,
    computes token counts, determines correct vs. not, etc.
    """
    # find all evaluation folders:
    EVAL_FOLDERS = find_evaluation_folders(base_dir, experiment_type)
    all_data = []

    for folder_path in EVAL_FOLDERS:
        for file in os.listdir(folder_path):
            if any(excluded in file for excluded in EXCLUDE_FILES):
                continue
            if file.endswith(".csv"):
                file_path = os.path.join(folder_path, file)
                df = pd.read_csv(file_path)

                book_name = extract_book_name(file)
                key_ = book_name.lower()
                if key_ in metadata_dict:
                    is_copyrighted = metadata_dict[key_]
                else:
                    print(f"Book '{book_name}' not in metadata => default Public.")
                    is_copyrighted = False

                masked_df = load_masked_passages(book_name)
                if masked_df is None:
                    continue

                for lang_group, lang_cols in LANG_GROUPS.items():
                    for lang_col in lang_cols:
                        correct_col = f"{lang_col}_correct"
                        if correct_col in df.columns and lang_col in masked_df.columns:
                            df[correct_col] = df[correct_col].astype(str).str.lower().str.strip()
                            token_counts = masked_df[lang_col].apply(get_token_count)
                            match_found = df[correct_col] == "correct"
                            tmp = pd.DataFrame({
                                "language_group": lang_group,
                                "tokens": token_counts,
                                "match_found": match_found,
                                "copyrighted": is_copyrighted
                            })
                            all_data.append(tmp)
    if not all_data:
        return pd.DataFrame(columns=["language_group","tokens","match_found","copyrighted"])
    return pd.concat(all_data, ignore_index=True)

# ============ 3) Bucket-based Accuracy ============

def compute_accuracy(data):
    out = []
    for (min_tok, max_tok) in TOKEN_BUCKETS:
        subset = data[(data["tokens"] >= min_tok) & (data["tokens"] < max_tok)]
        label = f"{min_tok}-{int(max_tok) if max_tok!=float('inf') else '400+'}"
        if len(subset) > 0:
            acc = subset["match_found"].mean() * 100
            out.append((label, acc))
        else:
            out.append((label, np.nan))
    return out

# ============ 4) Plotting ============

def main():
    metadata_dict = load_book_metadata("scripts/Evaluation/analyses_graph/metadata.csv")
    shot_types = ["one-shot", "zero-shot"]
    base_dir = "results/name_cloze"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    x_positions = np.arange(len(TOKEN_BUCKETS))

    def parse_colname(col):
        parts = col.rsplit("_", 1)
        grp = parts[0]
        cflag = (parts[1] == "C")
        return grp, cflag

    bucket_labels = [
        f"{min_}-{int(max_) if max_!=float('inf') else '400+'}"
        for (min_, max_) in TOKEN_BUCKETS
    ]

    for i, shot in enumerate(shot_types):
        ax = axes[i]
        print(f"Processing: {shot}")
        data_df = load_and_process_data(metadata_dict, base_dir, shot)
        if data_df.empty:
            print(f"Warning: No data for {shot}")
            continue

        grouped = data_df.groupby(["language_group", "copyrighted"])
        aggregator = {}
        for (lang_grp, cflag), subset in grouped:
            acc_list = compute_accuracy(subset)
            aggregator[(lang_grp, cflag)] = acc_list

        result_df = pd.DataFrame(index=bucket_labels)
        for key, val in aggregator.items():
            lang_grp, cflag = key
            col_name = f"{lang_grp}_{'C' if cflag else 'NC'}"
            val_map = dict(val)
            result_df[col_name] = [val_map.get(lab, np.nan) for lab in bucket_labels]

        for col in result_df.columns:
            grp, cflag = parse_colname(col)
            y_vals = result_df[col]
            base_color = FLARE_COLORS.get(grp, "gray")
            color_, style_ = get_color_and_style(base_color, cflag)
            label = f"{grp} ({'Copyright' if cflag else 'Public'})"
            ax.plot(x_positions, y_vals,
                    color=color_, linestyle=style_,
                    marker="o", linewidth=2, label=label)
            for x, y in zip(x_positions, y_vals):
                if not np.isnan(y):
                    ax.text(x, y + 0.5, f"{y:.1f}%", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(bucket_labels)
        ax.set_xlabel("Context Length (Tokens)")
        if i == 0:
            ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{shot.replace('-', ' ').title()}")

        ax.grid(True)

    # Deduplicated Legend
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(),
               loc="upper center", bbox_to_anchor=(0.5,-0.01),
               ncol=3, frameon=True)

    fig.suptitle("Name Cloze Task: Accuracy vs. Context Length (One-Shot vs. Zero-Shot)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = "name_cloze_accuracy_vs_context_length_by_shot.jpg"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print("Saved plot to:", out_path)
    plt.show()

# ============ HELPERS ============

def get_color_and_style(base_color, is_copyright):
    """
    If copyrighted => use base_color, solid line
    If public => lighten base_color, dashed line
    """
    if is_copyright:
        return (base_color, "solid")
    else:
        lighter = lighten_color(base_color, amount=0.4)
        return (lighter, "dashed")

def lighten_color(color, amount=0.4):
    """
    Lightens the given `color` by mixing with white.
    0 => no change, 1 => white
    """
    import matplotlib.colors as mc
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = mc.to_rgb(c)
    white = (1,1,1)
    return tuple((1-amount)*comp + amount*white_comp
                 for comp,white_comp in zip(c,white))

if __name__ == "__main__":
    main()
