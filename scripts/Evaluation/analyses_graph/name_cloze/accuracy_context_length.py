import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tiktoken

# Hardcoded paths
BASE_PROMPT_PATH = "/Users/alishasrivastava/BEAM-scripts/BEAM/scripts/Prompts"
BOOK_FOLDERS = [
    "/Users/alishasrivastava/BEAM-scripts/BEAM/results/name_cloze/gpt-4o-2024-11-20/one-shot/evaluation",
    "/Users/alishasrivastava/BEAM-scripts/BEAM/results/name_cloze/EuroLLM-9B-Instruct/one-shot/evaluation",
    "/Users/alishasrivastava/BEAM-scripts/BEAM/results/name_cloze/Llama-3.1-8B-Instruct-quantized.w4a16/one-shot/evaluation",
    "/Users/alishasrivastava/BEAM-scripts/BEAM/results/name_cloze/Llama-3.1-8B-Instruct-quantized.w8a16/one-shot/evaluation",
    "/Users/alishasrivastava/BEAM-scripts/BEAM/results/name_cloze/Llama-3.1-70B-Instruct_/one-shot/evaluation",
    "/Users/alishasrivastava/BEAM-scripts/BEAM/results/name_cloze/Llama-3.1-70B-Instruct-quantized.w4a16/one-shot/evaluation",
    "/Users/alishasrivastava/BEAM-scripts/BEAM/results/name_cloze/Llama-3.1-70B-Instruct-quantized.w8a16/one-shot/evaluation",
    "/Users/alishasrivastava/BEAM-scripts/BEAM/results/name_cloze/Llama-3.1-405b/one-shot/evaluation",
    "/Users/alishasrivastava/BEAM-scripts/BEAM/results/name_cloze/Llama-3.1-8B-Instruct_/one-shot/evaluation",
    "/Users/alishasrivastava/BEAM-scripts/BEAM/results/name_cloze/OLMo-2-1124-7B-Instruct/one-shot/evaluation",
    "/Users/alishasrivastava/BEAM-scripts/BEAM/results/name_cloze/OLMo-2-1124-13B-Instruct/one-shot/evaluation",
    "/Users/alishasrivastava/BEAM-scripts/BEAM/results/name_cloze/Qwen2.5-7B-Instruct-1M/one-shot/evaluation"
]
# Language groups
LANG_GROUPS = {
    "English": ["en"],
    "Translated": ["es", "tr", "vi"],
    "Cross-lingual": ["st", "yo", "tn", "ty", "mai", "mg"]
}

# Tokenization buckets
TOKEN_BUCKETS = [(0, 50), (50, 100), (100, 150), (150, 250), (250, 400), (400, float("inf"))]

# Skip files with these words
EXCLUDE_FILES = ["Below_Zero", "Bride", "You_Like", "First_Lie_Wins", "If_Only", 
                 "Just_for", "Lies_and", "Paper_Towns", "Ministry", "Paradise", "Funny_Story"]

# Flare color palette
FLARE_COLORS = {
    "English": "#FB5607",  # Bright orange
    "Translated": "#FF006E",  # Vivid pink
    "Cross-lingual": "#8338EC"  # Deep purple
}

# Load tiktoken tokenizer
tokenizer = tiktoken.get_encoding("o200k_base")

# Function to tokenize text
def get_token_count(text):
    if pd.isna(text) or not isinstance(text, str):
        return 0
    return len(tokenizer.encode(text))

# Function to extract book name (removes "_name_cloze" and everything after)
def extract_book_name(filename):
    return filename.split("_name_cloze")[0]  # Extract full book name before "_name_cloze"

# Function to load corresponding masked passages file
def load_masked_passages(book_name):
    masked_path = os.path.join(BASE_PROMPT_PATH, book_name, f"{book_name}_masked_passages.csv")
    
    if not os.path.exists(masked_path):
        print(f"Warning: Masked passages file not found for {book_name}")
        return None
    
    return pd.read_csv(masked_path)

# Function to process all book CSVs in multiple folders
def load_and_process_data():
    all_data = []

    for folder_path in BOOK_FOLDERS:
        for file in os.listdir(folder_path):
            if any(excluded in file for excluded in EXCLUDE_FILES):
                print(f"Skipping file: {file}")
                continue

            if file.endswith(".csv"):
                file_path = os.path.join(folder_path, file)
                df = pd.read_csv(file_path)

                book_name = extract_book_name(file)
                masked_df = load_masked_passages(book_name)

                if masked_df is None:
                    continue

                for lang in LANG_GROUPS.keys():
                    lang_columns = LANG_GROUPS[lang]

                    for lang_col in lang_columns:
                        correct_col = f"{lang_col}_correct"
                        masked_col = f"{lang_col}_masked" if lang in ["English", "Translated"] else lang_col

                        if correct_col in df.columns and masked_col in masked_df.columns:
                            df[correct_col] = df[correct_col].astype(str).str.lower().str.strip()
                            
                            token_counts = masked_df[masked_col].apply(get_token_count)
                            match_found = df[correct_col] == "correct"

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

# Function to compute accuracy per token bucket
def compute_accuracy(data):
    bucket_accuracies = []
    for min_tokens, max_tokens in TOKEN_BUCKETS:
        bucket_data = data[(data["tokens"] >= min_tokens) & (data["tokens"] < max_tokens)]
        if len(bucket_data) > 0:
            accuracy = bucket_data["match_found"].mean() * 100
            bucket_accuracies.append((f"{min_tokens}-{int(max_tokens) if max_tokens != float('inf') else '400+'}", accuracy))
        else:
            bucket_accuracies.append((f"{min_tokens}-{int(max_tokens) if max_tokens != float('inf') else '400+'}", np.nan))
    return bucket_accuracies

# Compute accuracy per bucket for each language group
accuracy_results = aggregated_data.groupby("language_group").apply(compute_accuracy).to_dict()

# Convert accuracy results into DataFrame
accuracy_df = pd.DataFrame({
    "Context Length Bucket": [x[0] for x in accuracy_results["English"]],
    "English": [x[1] for x in accuracy_results["English"]],
    "Translated": [x[1] for x in accuracy_results["Translated"]],
    "Cross-lingual": [x[1] for x in accuracy_results["Cross-lingual"]]
})

# Convert x-axis categories to numeric positions
x_labels = accuracy_df["Context Length Bucket"]
x_positions = range(len(x_labels))

# Plot with Flare colors
plt.figure(figsize=(10, 6))

for group in ["English", "Translated", "Cross-lingual"]:
    y = accuracy_df[group]
    plt.plot(x_positions, y, marker="o", label=group, color=FLARE_COLORS[group], linewidth=2)

# Add black dashed arrows with deltas
for i in range(len(x_positions)):
    try:
        en_acc = accuracy_df["English"][i]
        trans_acc = accuracy_df["Translated"][i]
        xling_acc = accuracy_df["Cross-lingual"][i]

        if not np.isnan(en_acc) and not np.isnan(trans_acc):
            plt.annotate("", xy=(x_positions[i], trans_acc), xytext=(x_positions[i], en_acc),
                         arrowprops=dict(arrowstyle="->", linestyle="dashed", color="black", linewidth=1.5))
            plt.annotate(f"Δ{en_acc - trans_acc:.1f}%", xy=(x_positions[i], (en_acc + trans_acc) / 2),
                         xytext=(-5, 0), textcoords="offset points", ha='right', fontsize=10, color="black", fontweight="bold")

        if not np.isnan(trans_acc) and not np.isnan(xling_acc):
            plt.annotate("", xy=(x_positions[i], xling_acc), xytext=(x_positions[i], trans_acc),
                         arrowprops=dict(arrowstyle="->", linestyle="dashed", color="black", linewidth=1.5))
            plt.annotate(f"Δ{trans_acc - xling_acc:.1f}%", xy=(x_positions[i], (trans_acc + xling_acc) / 2),
                         xytext=(-5, 0), textcoords="offset points", ha='right', fontsize=10, color="black", fontweight="bold")

    except IndexError:
        pass

plt.xlabel("Context Length (Tokens)")
plt.ylabel("Accuracy (%)")
plt.title("Name Cloze Task: One-Shot Accuracy vs. Context Length")
plt.legend()
plt.grid(True)
plt.savefig("flare_accuracy_vs_context_length.png", dpi=300, bbox_inches="tight")
plt.show()
