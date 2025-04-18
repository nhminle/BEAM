import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tiktoken
import sys
# Hardcoded paths
BASE_PROMPT_PATH = "/home/ekorukluoglu_umass_edu/beam2/BEAM/scripts/Prompts"
BOOK_FOLDERS = [
    
    "/home/ekorukluoglu_umass_edu/beam2/BEAM/results/direct_probe/gpt-4o-2024-11-20/masked_one_shot/evaluation",
    "/home/ekorukluoglu_umass_edu/beam2/BEAM/results/direct_probe/EuroLLM-9B-Instruct/masked_one_shot/evaluation",
    "/home/ekorukluoglu_umass_edu/beam2/BEAM/results/direct_probe/Llama-3.1-8B-Instruct-quantized.w4a16/masked_one_shot/evaluation",
    "/home/ekorukluoglu_umass_edu/beam2/BEAM/results/direct_probe/Llama-3.1-8B-Instruct-quantized.w8a16/masked_one_shot/evaluation",
    "/home/ekorukluoglu_umass_edu/beam2/BEAM/results/direct_probe/Llama-3.1-70B-Instruct_/masked_one_shot/evaluation",
    "/home/ekorukluoglu_umass_edu/beam2/BEAM/results/direct_probe/Llama-3.1-70B-Instruct-quantized.w4a16/masked_one_shot/evaluation",
    "/home/ekorukluoglu_umass_edu/beam2/BEAM/results/direct_probe/Llama-3.1-70B-Instruct-quantized.w8a16/masked_one_shot/evaluation",
    "/home/ekorukluoglu_umass_edu/beam2/BEAM/results/direct_probe/Llama-3.1-405b/masked_one_shot/evaluation",
    "/home/ekorukluoglu_umass_edu/beam2/BEAM/results/direct_probe/Llama-3.1-8B-Instruct_/masked_one_shot/evaluation",
    "/home/ekorukluoglu_umass_edu/beam2/BEAM/results/direct_probe/OLMo-2-1124-7B-Instruct/masked_one_shot/evaluation",
    "/home/ekorukluoglu_umass_edu/beam2/BEAM/results/direct_probe/OLMo-2-1124-13B-Instruct/masked_one_shot/evaluation",
    "/home/ekorukluoglu_umass_edu/beam2/BEAM/results/direct_probe/Qwen2.5-7B-Instruct-1M/masked_one_shot/evaluation"
]

# Language groups
LANG_GROUPS = {
    "English": ["en"],
    "Translated": ["es", "tr", "vi"],
    "Cross-lingual": ["st", "yo", "tn", "ty", "mai", "mg"]
}

# Tokenization buckets
TOKEN_BUCKETS = [(0, 50), (50, 100), (100, float("inf"))]

# Skip files with these words
EXCLUDE_FILES = ["Below_Zero", "Bride", "You_Like", "First_Lie_Wins", "If_Only", 
                 "Just_for", "Lies_and", "Paper_Towns", "Ministry", "Paradise", "Funny_Story", "2024"]

# Flare color palette
FLARE_COLORS = {
    "English": "#FB5607",
    "Translated": "#FF006E",
    "Cross-lingual": "#8338EC"
}

# Load tiktoken tokenizer
tokenizer = tiktoken.get_encoding("o200k_base")

def get_token_count(text):
    """ Tokenizes text and returns the token count. """
    if pd.isna(text) or not isinstance(text, str):
        return 0
    return len(tokenizer.encode(text))

def extract_book_name(filename):
    """ Extracts book name from the filename and replaces spaces with underscores. """
    name = os.path.splitext(filename)[0]  # Remove file extension
    name = name.split("_direct_probe")[0]
    name = name.split("_eval")[0]
    return name.replace(" ", "_").strip()  # Convert spaces to underscores

def load_masked_passages(book_name):
    """ Loads the masked passages file for the given book. """
    formatted_name = book_name.replace(" ", "_")  # Convert spaces to underscores
    masked_path = os.path.join(BASE_PROMPT_PATH, formatted_name, f"{formatted_name}_masked_passages.csv")
    
    if not os.path.exists(masked_path):
        print(f"Warning: Masked passages file not found for {book_name}. Expected at: {masked_path}")
        return None
    
    return pd.read_csv(masked_path)

def load_and_process_data():
    """ Loads and processes all book CSV files. """
    all_data = []

    for folder_path in BOOK_FOLDERS:
        for file in os.listdir(folder_path):
            if any(excluded in file for excluded in EXCLUDE_FILES):
                print(f"Skipping file: {file}")
                continue

            if not file.endswith(".csv") or "aggregate_data" in file:
                print(f"Skipping non-book file: {file}")
                continue

            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)

            book_name = extract_book_name(file)
            masked_df = load_masked_passages(book_name)
            print(f"WORKING ON FILE WITH THE NAME : {file}")
            if masked_df is None:
                continue

            for lang in LANG_GROUPS.keys():
                lang_columns = LANG_GROUPS[lang]

                for lang_col in lang_columns:
                    correct_col = f"{lang_col}_results_both_match"
                    masked_col = lang_col #masked_col = f"{lang_col}_masked" if lang in ["English", "Translated"] else lang_col IF MASKED DATA else masked_col = lang_col
                    # print(df.columns)
                    # print("dassak\n")
                    # print(masked_df.columns)
                    # sys.exit()
                    if correct_col in df.columns and "en_masked" in masked_df.columns:
                        print("a")
                        df[correct_col] = df[correct_col].astype(str).str.lower().str.strip()
                        
                        # Calculate token counts using the 'en' column for non-shuffled data
                        token_counts = masked_df["en_masked"].apply(get_token_count)
                        match_found = df[correct_col] == "true"

                        temp_df = pd.DataFrame({
                            "language_group": lang,
                            "tokens": token_counts,
                            "match_found": match_found
                        })
                        all_data.append(temp_df)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame(columns=["language_group", "tokens", "match_found"])

# Load all data
print("Processing book CSVs...")
aggregated_data = load_and_process_data()

def compute_accuracy(data):
    """ Computes accuracy per token bucket. """
    bucket_accuracies = []
    for min_tokens, max_tokens in TOKEN_BUCKETS:
        bucket_data = data[(data["tokens"] >= min_tokens) & (data["tokens"] < max_tokens)]
        accuracy = bucket_data["match_found"].mean() * 100 if len(bucket_data) > 0 else np.nan
        bucket_accuracies.append((f"{min_tokens}-{int(max_tokens) if max_tokens != float('inf') else '400+'}", accuracy))
    return bucket_accuracies

# Compute accuracy per bucket for each language group
accuracy_results = (
    aggregated_data.groupby("language_group", group_keys=False)
    .apply(compute_accuracy)
    .to_dict()
)

# Ensure all expected language groups are present
expected_groups = ["English", "Translated", "Cross-lingual"]
for group in expected_groups:
    if group not in accuracy_results:
        print(f"Warning: No data found for {group}. Filling with NaN values.")
        accuracy_results[group] = [(bucket, np.nan) for bucket, _ in TOKEN_BUCKETS]

# Convert accuracy results into DataFrame
accuracy_df = pd.DataFrame({
    "Context Length Bucket": [x[0] for x in accuracy_results["English"]],
    "English": [x[1] for x in accuracy_results["English"]],
    "Translated": [x[1] for x in accuracy_results["Translated"]],
    "Cross-lingual": [x[1] for x in accuracy_results["Cross-lingual"]]
})

# Plot with Flare colors
# Plot with Flare colors
# Plot with Flare colors
plt.figure(figsize=(10, 6))
for group in ["English", "Translated", "Cross-lingual"]:
    plt.plot(accuracy_df["Context Length Bucket"], accuracy_df[group], marker="o", label=group, 
             color=FLARE_COLORS[group], linewidth=2)

# Get x positions for each context length bucket
x_positions = np.arange(len(accuracy_df["Context Length Bucket"]))

# Add black dashed arrows with deltas
for i in range(len(x_positions)):
    try:
        en_acc = accuracy_df["English"][i]
        trans_acc = accuracy_df["Translated"][i]
        xling_acc = accuracy_df["Cross-lingual"][i]

        if i == 0:
            xytext_pos = (5, 0)
            ha_pos = 'left'
        else:
            xytext_pos = (-5, 0)
            ha_pos = 'right'

        if not np.isnan(en_acc) and not np.isnan(trans_acc):
            plt.annotate("", xy=(x_positions[i], trans_acc), xytext=(x_positions[i], en_acc),
                         arrowprops=dict(arrowstyle="->", linestyle="dashed", color="black", linewidth=1.5))
            plt.annotate(f"Δ{en_acc - trans_acc:.1f}%", xy=(x_positions[i], (en_acc + trans_acc) / 2),
                         xytext=xytext_pos, textcoords="offset points", ha=ha_pos, fontsize=10, color="black", fontweight="bold")

        if not np.isnan(trans_acc) and not np.isnan(xling_acc):
            plt.annotate("", xy=(x_positions[i], xling_acc), xytext=(x_positions[i], trans_acc),
                         arrowprops=dict(arrowstyle="->", linestyle="dashed", color="black", linewidth=1.5))
            plt.annotate(f"Δ{trans_acc - xling_acc:.1f}%", xy=(x_positions[i], (trans_acc + xling_acc) / 2),
                         xytext=xytext_pos, textcoords="offset points", ha=ha_pos, fontsize=10, color="black", fontweight="bold")

    except IndexError:
        pass

# Recalculate passage counts for the updated token buckets
passage_counts = aggregated_data.groupby(pd.cut(aggregated_data['tokens'], [0, 50, 100, float('inf')])).size()

# Annotate the number of passages above each x-tick
for i, count in enumerate(passage_counts):
    plt.text(x_positions[i], -5, f"n={count}", ha='center', fontsize=10, fontweight='bold', color='black', transform=plt.gca().transAxes)

plt.xlabel("Context Length (Tokens)", fontweight='bold')
plt.ylabel("Accuracy (%)", fontweight='bold')
plt.title("Direct Probe: Accuracy vs. Context Length", fontweight='bold')
plt.legend(frameon=False)
plt.grid(False)
plt.xticks(x_positions, accuracy_df["Context Length Bucket"], rotation=45, fontweight='bold')
plt.yticks(fontweight='bold')

# Remove the box around the graph and set light gray lines for x and y axes
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('lightgray')
ax.spines['bottom'].set_color('lightgray')

plt.savefig("os_non_shuffled_one_shot_accuracy_vs_context_length.pdf", format='pdf', dpi=300, bbox_inches="tight")
plt.show()

# Print the number of passages for each context length bucket
print("Passage counts for each context length bucket:")
for i, count in enumerate(passage_counts):
    print(f"{TOKEN_BUCKETS[i][0]}-{TOKEN_BUCKETS[i][1]}: {count} passages")
