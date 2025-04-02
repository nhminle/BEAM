import os
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
# import pandas as pd
from fuzzywuzzy import fuzz
from unidecode import unidecode
from collections import defaultdict
import polars as pl
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

# Define a threshold for fuzzy matching (0-100)
FUZZY_THRESHOLD = 70

# Pre-compile the regex to remove punctuation.
PUNCTUATION_RE = re.compile(r'[^\w\s]')

def normalize_text(text):
    """
    Normalize text by converting Unicode characters to ASCII,
    converting to lowercase, and removing punctuation using a pre-compiled regex.
    """
    text = unidecode(text)
    text = text.lower()
    text = PUNCTUATION_RE.sub('', text)
    return text


def tokenize(text):
    """Simple whitespace tokenizer."""
    return text.split()

def build_inverted_index(search_terms):
    """
    Build an inverted index mapping each token to a set of indices into search_terms.
    Each search term already has a precomputed normalized version (norm_term).
    """
    inverted_index = defaultdict(set)
    print("Building inverted index")
    for idx, item in enumerate(search_terms):
        tokens = tokenize(item['norm_term'])
        for token in tokens:
            inverted_index[token].add(idx)
    return inverted_index

def get_candidate_indices(norm_text, inverted_index):
    """
    Given normalized text from a JSONL line, tokenize it and return a set
    of candidate search term indices based on the inverted index.
    """
    candidate_indices = set()
    tokens = tokenize(norm_text)
    for token in tokens:
        if token in inverted_index:
            candidate_indices.update(inverted_index[token])
    return candidate_indices
def process_jsonl(file_path, search_terms, inverted_index):
    """
    Process a single JSONL file:
      - Read the file in a buffered manner.
      - Normalize each JSON line's "text" field.
      - Use the inverted index to fetch candidate search terms.
      - Compare the line's normalized text against candidate search terms using RapidFuzz.
      - Record each occurrence with file details.
    """
    from rapidfuzz import fuzz  # Using RapidFuzz for fuzzy matching
    results = []
    print(f"[Process {os.getpid()}] Processing file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        # Read the entire file content into memory.
        content = f.read()
        # Split the content into lines.
        lines = content.splitlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
            original_text = entry.get('text', '')
            norm_text = normalize_text(original_text)
            candidate_indices = get_candidate_indices(norm_text, inverted_index)
            for idx in candidate_indices:
                candidate = search_terms[idx]
                term = candidate['term']
                norm_term = candidate['norm_term']
                source_csv = candidate['source_csv']
                # Use RapidFuzz's partial_ratio for fuzzy matching.
                if fuzz.partial_ratio(norm_term, norm_text) >= FUZZY_THRESHOLD:
                    print(f"[Process {os.getpid()}] Found a match for term: {term}")
                    results.append({
                        'jsonl_file': file_path,
                        'term': term,
                        'text': original_text,
                        'source_csv': source_csv
                    })
        except json.JSONDecodeError:
            continue
    return results


def process_jsonl_batch(file_batch, search_terms, inverted_index):
    """
    Process a batch of JSONL files.
    """
    pid = os.getpid()
    print(f"[Process {pid}] Started processing a batch of {len(file_batch)} JSONL files.")
    batch_results = []
    for file_path in file_batch:
        batch_results.extend(process_jsonl(file_path, search_terms, inverted_index))
    print(f"[Process {pid}] Finished processing its batch.")
    return batch_results

def chunkify(lst, chunk_size):
    """
    Yield successive chunk_size chunks from lst.
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def load_search_terms(search_csv_directory):
    """
    Load search terms from all CSV files in the provided directory using Polars.
    Pre-compute the normalized version of each search term.
    Only consider CSV files that include "unmasked" or "non_NE" in their filename.
    """
    search_terms = []
    print("Finding terms to search...")
    for root, _, files in os.walk(search_csv_directory):
        for file in files:
            if file.endswith('.csv') and ("unmasked" in file or "non_NE" in file):
                print("Reading:", file)
                csv_path = os.path.join(root, file)
                try:
                    df = pl.read_csv(csv_path)
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")
                    continue

                # Check if "en" column exists; if not, use the first column.
                if "en" in df.columns:
                    terms = df["en"].drop_nulls().to_list()
                else:
                    # Use first column: convert the column to a list.
                    terms = df.select(pl.col(df.columns[0])).drop_nulls().to_series().to_list()
                
                for term in terms:
                    term = term.strip()
                    norm_term = normalize_text(term)
                    search_terms.append({
                        'term': term,
                        'norm_term': norm_term,
                        'source_csv': file
                    })
    return search_terms

def main(jsonl_directory, search_csv_directory, results_csv="results.csv"):
    # Load search terms from CSV files.
    search_terms = load_search_terms(search_csv_directory)
    print(f"Total search terms loaded: {len(search_terms)}")
    
    # Build the inverted index from the search terms.
    inverted_index = build_inverted_index(search_terms)
    
    # Collect all .jsonl file paths.
    jsonl_files = []
    for root, _, files in os.walk(jsonl_directory):
        for file in files:
            if file.endswith('.jsonl'):
                jsonl_files.append(os.path.join(root, file))
    
    print("Total JSONL files found:", len(jsonl_files))
    
    # Split jsonl_files into batches of 10 files each.
    file_batches = list(chunkify(jsonl_files, 5))
    print("Total batches:", len(file_batches))
    
    results_all = []
    # Process batches in parallel using up to 20 processes.
    with ProcessPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_jsonl_batch, batch, search_terms, inverted_index) 
                   for batch in file_batches]
        for future in as_completed(futures):
            results_all.extend(future.result())
    
    # Write results to CSV if any matches are found.
    if results_all:
        results_df = pl.DataFrame(results_all)
        results_df.to_csv(results_csv, index=False, encoding='utf-8')
        print(f"Results written to {results_csv}")
    else:
        print("No matches found.")

if __name__ == "__main__":
    # Update these paths as needed.
    jsonl_directory = "/scratch3/workspace/ekorukluoglu_umass_edu-simple/dclm_dataset/global-shard_01_of_10/local-shard_1_of_10"       # Directory with .jsonl files
    search_csv_directory = "/home/ekorukluoglu_umass_edu/beam2/BEAM/scripts/Prompts/1984"        # Directory containing CSV files with search terms
    main(jsonl_directory, search_csv_directory)
