import os
import pandas as pd
import unidecode
from fuzzywuzzy import fuzz

# --------------------------
# Helper Functions
# --------------------------

def run_exact_match(correct_author, correct_title_list, returned_author, returned_title):
    """
    Compares the returned title and author with the correct ones using fuzzy matching.
    Returns True if both title and author match (with threshold >=90), otherwise False.
    """
    returned_author = str(returned_author) if pd.notna(returned_author) else ''
    correct_author = str(correct_author) if pd.notna(correct_author) else ''

    title_match = any(
        fuzz.ratio(
            unidecode.unidecode(str(returned_title)).lower(),
            unidecode.unidecode(str(correct_title)).lower()
        ) >= 90
        for correct_title in correct_title_list
    ) or any(
        unidecode.unidecode(str(correct_title)).lower() 
        in unidecode.unidecode(str(returned_title)).lower()
        for correct_title in correct_title_list
    )

    author_match = (
        fuzz.ratio(
            unidecode.unidecode(correct_author).lower(),
            unidecode.unidecode(returned_author).lower()
        ) >= 90
        or unidecode.unidecode(correct_author).lower()
        in unidecode.unidecode(returned_author).lower()
    )

    return title_match and author_match

def extract_title_author(results_column):
    """
    Extracts title and author from the 'en' column text using a regular expression.
    For rows that do not match the expected JSON-like pattern, the entire row's text is used for both.
    Returns a DataFrame with two columns: first for title and second for author.
    """
    results_column = results_column.fillna('').astype(str).str.strip()
    extracted = results_column.str.extract(r'"title":\s*"(.*?)",\s*"author":\s*"(.*?)"')
    unmatched_rows = extracted.isnull().all(axis=1)
    extracted.loc[unmatched_rows, 0] = results_column[unmatched_rows]
    extracted.loc[unmatched_rows, 1] = results_column[unmatched_rows]
    # print("tassak")
    # print(extracted)
    return extracted

def get_correct_book_info(book_title, book_names_df):
    """
    Uses fuzzy matching against the "En" column of book_names_df to select the best match
    for the given book_title. Returns the matching row (a Series) if the best score
    is at least 90; otherwise returns None.
    """
    best_match = None
    best_score = 0
    for idx, row in book_names_df.iterrows():
        # print(row)
        candidate = row['En']
        # print(book_title)
        score = fuzz.ratio(book_title.lower(), candidate.lower())
        if score > best_score:
            best_score = score
            best_match = row
    if best_score >= 90:
        return best_match
    else:
        return None

def parse_filename(base_name):
    """
    Parses a filename (without extension) that follows the pattern:
       bookname_directprobe_modelname_promptsetting
    For example, the filename:
       A_Tale_of_Two_Cities_directprobe_ModelOne_one-shot
    will yield:
       book_title: "A Tale of Two Cities"
       model: "ModelOne"
       prompt_setting: "one-shot"
    """
    if '_direct_probe_' not in base_name:
        raise ValueError(f"Expected '_direct_probe_' in filename: {base_name}")
    title_part, remainder = base_name.split('_direct_probe_', 1)
    book_title = title_part.replace('_', ' ').strip()
    if '_' not in remainder:
        raise ValueError(f"Expected '_' to split model and prompt setting in remainder: {remainder}")
    model, prompt_setting = remainder.rsplit('_', 1)
    model = model.strip()
    prompt_setting = prompt_setting.strip()
    return book_title, model, prompt_setting

def evaluate_en(csv_file_path):
    """
    Processes a CSV file and computes the fuzzy-match accuracy for the 'en' column only.
    Steps:
      - Parses the filename to extract book title, model, and prompt setting.
      - Reads the CSV and ensures that an 'en' column exists.
      - Loads the reference book information from book_names.csv.
      - Adjusts known special cases (e.g., "Alice in Wonderland" or "Percy Jackson the Lightning Thief").
      - Uses fuzzy matching (via get_correct_book_info) to locate the correct reference row.
      - Extracts title and author predictions from the 'en' column and compares each row to the reference.
      - Computes accuracy as the percentage of rows that match.
    Returns a tuple: (canonical_book_title, model, accuracy_percentage)
    or None if any error occurs.
    """
    base_name = os.path.splitext(os.path.basename(csv_file_path))[0]
    try:
        book_title, model, prompt_setting = parse_filename(base_name)
    except ValueError as e:
        print(e)
        return None

    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"Error reading {csv_file_path}: {e}")
        return None

    if 'en_results' not in df.columns:
        print(f"'en' column not found in {csv_file_path}. Skipping.")
        return None

    try:
        book_names = pd.read_csv('scripts/Evaluation/dir_probe/book_names.csv')  # adjust path if needed
    except Exception as e:
        print(f"Could not read book_names.csv: {e}")
        return None

    # Adjust book title for known special cases (case-insensitive)
    if book_title.lower() == "alice in wonderland":
        book_title = "Alice s Adventures in Wonderland"
    if book_title.lower() == "percy jackson the lightning thief":
        book_title = "The Lightning Thief"

    correct_info = get_correct_book_info(book_title, book_names)
    if correct_info is None:
        print(f"No matching book found for parsed title: '{book_title}' in {csv_file_path}")
        return None

    correct_author = correct_info['Author']
    correct_title = correct_info['En']

    extracted_df = extract_title_author(df['en_results'])
    # print(extracted_df)
    results = []
    for idx in range(len(extracted_df)):
        returned_title = extracted_df.iat[idx, 0]
        returned_author = extracted_df.iat[idx, 1]
        match = run_exact_match(correct_author, [correct_title], returned_author, returned_title)
        results.append(match)

    if not results:
        print(f"No valid rows in the 'en' column for {csv_file_path}.")
        return None

    accuracy = sum(results) / len(results) * 100
    # print(results)

    return (correct_title, model, accuracy)

def traverse_and_aggregate(results_dir):
    """
    Walks through the results directory (and its subfolders), filtering for CSV files
    that follow the naming pattern and include 'directprobe' in the filename.
    For each file, it extracts the accuracy (computed on the "en" column) and aggregates
    the data into a DataFrame where each row represents a book (using its canonical name
    from the "En" column of book_names.csv) and each column is a model.
    """
    aggregated = {}  # {book_title: {model: accuracy}}

    for root, dirs, files in os.walk(results_dir):
        for file in files:
            # Skip files that are not individual evaluation files.
            if (file.endswith('.csv') and 
                ("aggregate_data" not in file) and 
                ("eval" not in file) and 
                ("direct_probe" in file)):
                full_path = os.path.join(root, file)
                outcome = evaluate_en(full_path)
                if outcome is None:
                    continue
                book_title, model, accuracy = outcome
                if book_title not in aggregated:
                    aggregated[book_title] = {}
                aggregated[book_title][model] = accuracy

    df = pd.DataFrame.from_dict(aggregated, orient='index')
    df.index.name = 'Book Name'
    df.reset_index(inplace=True)
    return df

# --------------------------
# Main Workflow
# --------------------------

def main():
    # Set the top-level directory containing your results CSV files.
    results_dir = 'results/direct_probe'  # Adjust if necessary
    aggregated_df = traverse_and_aggregate(results_dir)

    if aggregated_df.empty:
        print("No accuracy data was aggregated. Check your filenames and folder structure.")
        return

    output_csv = 'aggregated_en_accuracies.csv'
    aggregated_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Aggregated 'en' accuracies saved to {output_csv}")

if __name__ == "__main__":
    main()
