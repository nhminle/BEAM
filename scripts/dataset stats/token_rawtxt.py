import os
import csv
import re
from collections import defaultdict
from transformers import AutoTokenizer

# Load tokenizer (can replace with multilingual if needed)
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def safe_tokenize(text, tokenizer):
    try:
        return len(tokenizer.encode(text))
    except Exception as e:
        print(f"Tokenization error: {e}")
        return 0

def count_words_and_tokens_streamed(root_dir, output_csv="grouped_counts_streamed_2024.csv"):
    lang_suffixes = ['en', 'es', 'tr', 'vi']
    grouped_counts = defaultdict(dict)

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".txt"):
                match = re.match(r"(.+?)_(" + "|".join(lang_suffixes) + r")\.txt", file)

                if match:
                    book_name, lang = match.groups()
                    file_path = os.path.join(dirpath, file)

                    total_words = 0
                    total_tokens = 0

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            for line in f:
                                words = line.strip().split()
                                total_words += len(words)
                                total_tokens += safe_tokenize(line, tokenizer)

                        grouped_counts[book_name][lang] = {
                            "words": total_words,
                            "tokens": total_tokens
                        }
                        print(f"{file_path} → words: {total_words}, tokens: {total_tokens}")

                    except Exception as e:
                        print(f"Failed to process {file_path}: {e}")
                        grouped_counts[book_name][lang] = {"words": "", "tokens": ""}

    # Write to CSV
    with open(output_csv, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        header = ["Book Name"]
        for lang in lang_suffixes:
            header += [f"{lang}_words", f"{lang}_tokens"]
        writer.writerow(header)

        for book in sorted(grouped_counts.keys()):
            row = [book]
            for lang in lang_suffixes:
                counts = grouped_counts[book].get(lang, {"words": "", "tokens": ""})
                row += [counts["words"], counts["tokens"]]
            writer.writerow(row)

    print(f"\nSaved grouped word/token counts to {output_csv}")
    return grouped_counts

directory_path = "/home/ekorukluoglu_umass_edu/beam2/BEAM/alignment/preprocess_books/2024"
count_words_and_tokens_streamed(directory_path)
