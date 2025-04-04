import os
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== USER CONFIG ==========
LANG = "en"  # "en", "es", "tr", "vi"
EVAL_BASE = "results/name_cloze"
ENTITY_CSV = "scripts/Evaluation/analyses_graph/named_entities.csv"

SHOT_TYPES = ["zero-shot", "one-shot"]

EXCLUDE_FILES = [
    "Below_Zero", "Bride", "You_Like", "First_Lie_Wins", "If_Only",
    "Just_for", "Lies_and", "Paper_Towns", "Ministry", "Paradise", "Funny_Story"
]

# ========== Load entity CSV ==========
def load_entity_csv(path):
    df = pd.read_csv(path)

    def parse_variant_str(s):
        try:
            return set(x.strip().lower() for x in ast.literal_eval(s) if x.strip())
        except:
            print("Warning: could not parse character list:", s)
            return set()

    df["Variants"] = df["Character List"].apply(parse_variant_str)
    df["Book_lc"] = df["Book"].str.strip().str.lower()

    mask = ~df["Book"].apply(lambda b: any(excl in b for excl in EXCLUDE_FILES))
    df = df[mask].reset_index(drop=True)
    return df

# ========== Find evaluation folders for each shot type ==========
def find_eval_folders(base_dir, shot_type):
    """
    Returns a list of folder paths for a single shot_type (zero-shot or one-shot).
    E.g. results/name_cloze/<model>/<shot_type>/evaluation
    """
    found = []
    for model in os.listdir(base_dir):
        eval_path = os.path.join(base_dir, model, shot_type, "evaluation")
        if os.path.isdir(eval_path):
            found.append(eval_path)
    return found

def parse_single_ent(s):
    try:
        return set(x.strip().lower() for x in ast.literal_eval(s) if x.strip())
    except:
        return set()

# ========== Combine zero-shot & one-shot into a single aggregator ==========
def compute_entity_accuracies_global(entity_df):
    """
    1) Finds all evaluation folders for BOTH zero-shot and one-shot
    2) Summation of correct_guesses across them all
    3) Divides by Count * total_number_of_folders => single global accuracy
    """
    all_folders = []
    for shot in SHOT_TYPES:
        shot_folders = find_eval_folders(EVAL_BASE, shot)
        all_folders.extend(shot_folders)

    num_folders = len(all_folders) 
    print(f"Found {num_folders} total evaluation folders (zero + one).")

    entity_df = entity_df.copy()
    entity_df["correct_guesses"] = 0

    for folder in all_folders:
        for file in os.listdir(folder):
            if not file.endswith(".csv"):
                continue
            if any(excl in file for excl in EXCLUDE_FILES):
                continue

            book_name = file.split("_name_cloze")[0].lower().strip()
            fpath = os.path.join(folder, file)

            eval_df = pd.read_csv(fpath)
            if "Single_ent" not in eval_df.columns or f"{LANG}_correct" not in eval_df.columns:
                continue

            book_entities = entity_df[entity_df["Book_lc"] == book_name]

            for _, row in eval_df.iterrows():
                single_ent_set = parse_single_ent(row["Single_ent"])
                if not single_ent_set:
                    continue

                correct_raw = str(row[f"{LANG}_correct"]).strip().lower()
                is_correct = (correct_raw == "correct")

                for idx, ent_row in book_entities.iterrows():
                    if single_ent_set & ent_row["Variants"]:
                        if is_correct:
                            entity_df.at[idx, "correct_guesses"] += 1
                        break

    entity_df["accuracy"] = entity_df["correct_guesses"] / (entity_df["Count"] * num_folders)
    return entity_df[["Book", "Character List", "Count", "correct_guesses", "accuracy", 'en']]

# ========== Plotting ==========
def plot_entity_accuracies(df):
    plt.figure(figsize=(10, 7))
    plt.scatter(df["en"], df["accuracy"] * 100, alpha=0.6)

    plt.xlabel("Character Mention Count in Book")
    plt.ylabel("Accuracy (%)")
    plt.title("NCT Accuracy (Zero + One-Shot) for English passages per Entity")
    print(df)
    x = df["en"]
    y = df["accuracy"] * 100

    if len(df) > 1 and len(set(x)) > 1:
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m * x + b, color="red", label=f"y = {m:.2f}x + {b:.2f}")
        r = np.corrcoef(x, y)[0, 1]
        plt.legend(title=f"r = {r:.3f}")
    else:
        plt.legend(["Not enough data for regression"])

    plt.grid(True)
    plt.tight_layout()

    out_plot = f"entity_accuracy_vs_count_{LANG}_global.png"
    out_csv = f"entity_accuracy_data_{LANG}_global.csv"
    plt.savefig(out_plot, dpi=300)
    print("Saved plot:", out_plot)

    df.to_csv(out_csv, index=False)
    print("Saved CSV:", out_csv)

    plt.show()

# ========== Main ==========
def main():
    entity_df = load_entity_csv(ENTITY_CSV)
    results_df = compute_entity_accuracies_global(entity_df)
    plot_entity_accuracies(results_df)

if __name__ == "__main__":
    main()