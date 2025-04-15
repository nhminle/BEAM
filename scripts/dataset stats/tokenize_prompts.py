import os
import csv
import pandas as pd
import tiktoken
from statistics import mean, median, stdev

# Tokenizer setup
encoding = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    try:
        return len(encoding.encode(str(text)))
    except:
        return 0

def is_lang_col(colname, lang):
    return f"only{lang}" in colname or colname.startswith(lang + "_")

def group_column(colname):
    if colname == "en":
        return "English"
    elif colname in ["es", "vi", "tr"]:
        return "Translations"
    elif colname in ["st", "yo", "tn", "ty", "mai", "mg"]:
        return "Crosslingual"
    else:
        return "none"
def process_csvs_under_prompts(root_dir="/home/ekorukluoglu_umass_edu/beam2/BEAM/scripts/Prompts", output_csv="token_stats_summary_non_ne.csv"):
    grouped_token_counts = {
        "English": [],
        "Translations": [],
        "Crosslingual": []
    }

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".csv") and "non_NE" in filename:
                print(filename)
                filepath = os.path.join(dirpath, filename)
                try:
                    df = pd.read_csv(filepath)
                except Exception as e:
                    print(f"Skipping {filepath} due to error: {e}")
                    continue

                for col in df.columns:

                    group = group_column(col)
                    if group != "none":
                        token_counts = df[col].dropna().astype(str).map(count_tokens).tolist()
                        grouped_token_counts[group].extend(token_counts)

    # Compute summary statistics
    summary = []
    for group in ["English", "Translations", "Crosslingual"]:
        values = grouped_token_counts[group]
        if values:
            summary.append({
                "Group": group,
                "Count": len(values),
                "Mean": round(mean(values), 2),
                "Median": round(median(values), 2),
                "Min": min(values),
                "Max": max(values),
                "Stdev": round(stdev(values), 2) if len(values) > 1 else 0.0
            })
        else:
            summary.append({
                "Group": group,
                "Count": 0,
                "Mean": "",
                "Median": "",
                "Min": "",
                "Max": "",
                "Stdev": ""
            })

    # Save to CSV
    out_df = pd.DataFrame(summary)
    out_df.to_csv(output_csv, index=False)
    print(f"Saved summary to {output_csv}")

# Run it
process_csvs_under_prompts()
