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
    # user picks "one-shot" or "zero-shot" above
    metadata_dict = load_book_metadata("scripts/Evaluation/analyses_graph/metadata.csv")
    aggregated_data = load_and_process_data(
        metadata_dict,
        base_dir="results/name_cloze",
        experiment_type=EXPERIMENT_TYPE
    )

    # aggregator[(lang_group, c_flag)] -> list of (bucket_label, accuracy)
    aggregator = {}
    grouped = aggregated_data.groupby(["language_group", "copyrighted"])
    for (lang_grp, c_flag), subset in grouped:
        bucket_acc = compute_accuracy(subset)
        aggregator[(lang_grp, c_flag)] = bucket_acc

    # Build final result DataFrame
    bucket_labels = [
        f"{min_}-{int(max_) if max_!=float('inf') else '400+'}"
        for (min_,max_) in TOKEN_BUCKETS
    ]
    result_df = pd.DataFrame(index=bucket_labels)

    for key, val in aggregator.items():
        lang_grp, c_flag = key
        col_name = f"{lang_grp}_{'C' if c_flag else 'NC'}"
        # val is list of (label,acc)
        val_map = dict(val)
        col_values = [ val_map.get(lab, np.nan) for lab in bucket_labels ]
        result_df[col_name] = col_values

    print("Final Data:\n", result_df)

    # Plot
    plt.figure(figsize=(10,6))
    x_positions = np.arange(len(bucket_labels))

    def parse_colname(col):
        # "English_C" => ("English", True)
        parts = col.rsplit("_",1)
        grp = parts[0]
        cflag = (parts[1] == "C")
        return grp, cflag

    for col in result_df.columns:
        grp, cflag = parse_colname(col)
        y_vals = result_df[col]
        base_color = FLARE_COLORS.get(grp, "gray")
        color_, style_ = get_color_and_style(base_color, cflag)
        lbl = f"{grp} ({'Copyright' if cflag else 'Public'})"
        plt.plot(x_positions, y_vals, color=color_, linestyle=style_, marker="o", linewidth=2, label=lbl)

    plt.xticks(x_positions, bucket_labels)
    plt.xlabel("Context Length (Tokens)")
    plt.ylabel("Accuracy (%)")
    # Title depends on experiment
    if EXPERIMENT_TYPE == "one-shot":
        plt.title("Name Cloze Task: One-Shot Accuracy vs. Context Length (Split by Copyright)")
        out_fname = f"{BASE_NAME}_accuracy_vs_context_length_copyright.png"
    else:
        plt.title("Name Cloze Task: Zero-Shot Accuracy vs. Context Length (Split by Copyright)")
        out_fname = f"{BASE_NAME}_accuracy_vs_context_length_copyright.png"

    plt.grid(True)
    # De-duplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), loc="best")

    plt.savefig(out_fname, dpi=300, bbox_inches="tight")
    print("Saved plot to:", out_fname)
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
