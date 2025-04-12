import pandas as pd
import os
import sys
from rapidfuzz import fuzz

def build_file_dict(project_folder):
    """
    Traverse a directory tree using os.walk and build a mapping
    from filename to its full path.
    """
    file_dict = {}
    for dirpath, _, filenames in os.walk(project_folder):
        for fname in filenames:
            file_dict[fname] = os.path.join(dirpath, fname)
    return file_dict

def split_filename(filename, secondary_suffix="non_NE"):
    """
    Remove the extension from the filename and then separate it into:
    - primary: everything before the suffix (if present)
    - secondary: the specified suffix if found, else fallback to splitting on the first underscore.
    """
    base_name = os.path.splitext(filename)[0]
    suffix_with_sep = "_" + secondary_suffix
    if base_name.endswith(suffix_with_sep):
        primary = base_name[:-len(suffix_with_sep)]
        secondary = secondary_suffix
    else:
        # Fallback: split into two parts (this may not be ideal if the book name itself contains underscores)
        parts = base_name.split("_", 1)
        primary = parts[0]
        secondary = parts[1] if len(parts) > 1 else ""
    return primary, secondary

def find_matching_file(splitted, file_dict):
    """
    Iterate over all files in file_dict. If a file's name (without extension)
    contains the secondary indicator (e.g., "non_NE") or the primary book title,
    return the full file path.
    """
    primary, secondary = splitted
    matching_files = []
    for fname, full_path in file_dict.items():
        # print(full_path)
        base_fname = os.path.splitext(fname)[0]

        if secondary in base_fname and primary in base_fname:
            # print(full_path)
            matching_files.append(full_path)
    # print(matching_files)
    return matching_files
def sliding_window_match(text1, text2, threshold=80):
    """
    Compare two texts using a sliding window approach. The window size is set to the length of the shorter text.
    For each window in the longer text, compute the similarity score with the shorter text using fuzz.ratio.
    If any window's score meets or exceeds the threshold, return True indicating a match.
    """
    text1 = text1.lower()
    text2 = text2.lower()
    # Identify which text is shorter to be the sliding window base.
    if len(text1) < len(text2):
        short_text, long_text = text1, text2
    else:
        short_text, long_text = text2, text1

    window_size = len(short_text)
    if window_size == 0:
        return False  # Avoid division by zero or empty comparisons

    # Slide over the longer text one character at a time.
    for i in range(0, len(long_text) - window_size + 1):
        candidate = long_text[i:i+window_size]
        score = fuzz.ratio(short_text, candidate)
        if score >= threshold:
            return True
    return False
def get_en_results(row, file_data):
    """
    Given a row from the main CSV and a DataFrame loaded from the matched file,
    do an exact match on the passage to retrieve the corresponding en_results.
    """
    passage_to_find = row["passage"]
    match = file_data[file_data["passage"] == passage_to_find]
    if not match.empty:
        return match.iloc[0]["en_results"]
    else:
        return None

# Paths and folder definitions:
input_csv_path = "search-results-(1)-csv.csv"
project_folder = "/home/ekorukluoglu_umass_edu/beam2/BEAM/results/direct_probe"

# Load the main CSV which records occurrences of passages:
df_main = pd.read_csv(input_csv_path)
df_main["en_results"] = None  # New column to hold results

# Build the dictionary mapping filenames to full paths:
file_dict = build_file_dict(project_folder)
print("Built file dictionary.")
result_df = []
# Iterate over each unique filename in df_main:
for fname in df_main["filename"].unique():
    # Split the target filename from the main CSV into primary and secondary parts.
    splitted = split_filename(fname, "non_NE")
    # print(f"Processing {fname}: split into {splitted}")

    # Search over file_dict to find a file that contains either the primary or secondary identifier.
    matching_file_paths = find_matching_file(splitted, file_dict)
    if matching_file_paths:# print(type(matching_file_paths))
        print(f"Found {len(matching_file_paths)} matching files for the book : {fname}")
        for files in matching_file_paths:
            # print(files)
            splittedfilename = files.split('/')
            # print(splittedfilename[7])
            # print(len(matching_file_paths))
            print(f"finding occurences in model : {splittedfilename[7]}")
            try:
                # Load the CSV file corresponding to the matching file.
                file_df = pd.read_csv(files)
                # Update rows in the main CSV that reference this filename.
                for index,rows in df_main.iterrows():
                    if rows['filename'] == fname:
                        passage = rows["passage"]
                        # print(rows['filename'])
                        for index, rows in file_df.iterrows():
                            org_passage = rows["en"]
                            normalized_org = str(org_passage).lower()
                            normalized_passage = str(passage).lower()
                            score = sliding_window_match(normalized_org,normalized_passage)
                            if score:
                                rows['filename'] = fname
                                rows['model'] = splittedfilename[7]
                                result_df.append(rows)

            except Exception as e:
                print(f"Error processing file {files}: {e}")
    else:
        print(f"Matching file not found for: {fname}")
        print(splitted)
        print(matching_file_paths)
# print(result_df)

result_dframe = pd.DataFrame(result_df)
sorted_df = result_dframe.sort_values(by='filename', ascending=True)
# Save the updated main CSV to a new file.
output_csv_path = "updated_search_results.csv"
sorted_df.to_csv(output_csv_path, index=False)
print(f"Updated CSV saved to {output_csv_path}")
