import os
import csv
import re
import json
from unidecode import unidecode
from rapidfuzz import fuzz

def tokenize(text):
    """
    Normalizes and tokenizes the input text.
    Uses unidecode to convert unicode text to ASCII, then splits based on non-alphanumeric characters.
    All tokens are converted to lowercase.
    """
    normalized_text = unidecode(text)
    tokens = re.split(r'\W+', normalized_text)
    return [token.lower() for token in tokens if token]

def process_csv_file(file_path, inverted_index):
    """
    Processes a CSV file by reading the 'en' column, tokenizing its content,
    and updating the inverted index with token counts per file.
    """
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if 'en' not in reader.fieldnames:
                print(f"Column 'en' not found in {file_path}. Skipping file.")
                return
            for row in reader:
                text = row.get('en', '')
                for token in tokenize(text):
                    # Update the index by incrementing the count for the token in the given file.
                    if token in inverted_index:
                        if file_path in inverted_index[token]:
                            inverted_index[token][file_path] += 1
                        else:
                            inverted_index[token][file_path] = 1
                    else:
                        inverted_index[token] = {file_path: 1}
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def build_inverted_index(root_folder):
    """
    Recursively walks through root_folder to process all CSV files and build an inverted index
    that records the number of occurrences of each token per file.
    """
    inverted_index = {}
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith('.csv'):
                file_path = os.path.join(dirpath, filename)
                process_csv_file(file_path, inverted_index)
    return inverted_index

def save_inverted_index(index, output_file):
    """
    Saves the inverted index as a JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(index, f, indent=2)
    print(f"Inverted index saved to {output_file}")

def query_index(token, inverted_index, fuzzy_threshold=80):
    """
    Queries the inverted index for a token. If an exact match isn't found,
    applies fuzzy matching to return near matches that exceed the fuzzy_threshold.
    
    :param token: The query token.
    :param inverted_index: The inverted index built earlier.
    :param fuzzy_threshold: The minimum fuzzy matching score (0-100) to consider as a match.
    :return: A dictionary mapping file paths to counts for the token (or its fuzzy match).
    """
    token = token.lower()
    if token in inverted_index:
        return inverted_index[token]
    
    # Fuzzy matching: iterate through keys to find near matches.
    for key in inverted_index.keys():
        if fuzz.ratio(token, key) >= fuzzy_threshold:
            return inverted_index[key]
    return {}

# Example usage:
if __name__ == "__main__":
    main_folder = "/home/ekorukluoglu_umass_edu/beam2/BEAM/scripts/Prompts"  # Update this path to your main folder containing CSV files.
    output_index_file = "/home/ekorukluoglu_umass_edu/beam2/BEAM/olmo-search/reverse_index/inverted_index.json"
    
    # Build the inverted index from CSV files.
    index = build_inverted_index(main_folder)
    
    # Save the index to a file for later use.
    save_inverted_index(index, output_index_file)
    query = "Alice"
    matched_files = query_index(query, index)
    print(f"Files and occurrence counts for token '{query}': {matched_files}")
    with open("results.txt", 'w') as file:
        file.write(f"Files and occurrence counts for token '{query}': {matched_files}")
    # Example query: searching for a token (or fuzzy match if needed).