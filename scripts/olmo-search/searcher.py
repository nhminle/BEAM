import os
from multiprocessing import Pool
import pandas as pd
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def search_in_shard_for_text(shard_path, search_text):
    """
    Searches for a specific text in a shard file line by line.
    :param shard_path: Path to the shard file.
    :param search_text: Text to search for.
    :return: List of matching lines.
    """
    results = []
    with open(shard_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            if search_text in line:
                results.append(f"Text: {search_text}, File: {shard_path}, Line: {line_number}")
    return results


def search_in_shard_from_csv(shard_path, csv_file_path, text_columns):
    """
    Searches for all lines in the specified columns of a CSV within a single shard.
    :param shard_path: Path to the shard file.
    :param csv_file_path: Path to the CSV file.
    :param text_columns: List of column names in the CSV that contain text to search.
    :return: List of results for all text entries in the CSV file.
    """
    results = []
    search_df = pd.read_csv(csv_file_path)

    # Filter out columns that do not exist in the CSV
    valid_columns = [col for col in text_columns if col in search_df.columns]

    if not valid_columns:
        logging.info(f"Skipping {csv_file_path}: None of the specified columns exist.")
        return results
    logging.info(f"Searching book {csv_file_path}")
    # Iterate over valid columns
    for column in valid_columns:
        for text in search_df[column].dropna():  # Skip NaN values
            matches = search_in_shard_for_text(shard_path, text)
            results.extend(matches)

    return results


def get_all_csv_files(base_folder):
    """
    Recursively fetches all CSV file paths in the base folder and its subfolders.
    :param base_folder: Path to the base folder.
    :return: List of CSV file paths.
    """
    csv_files = []
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".csv"):
                csv_files.append(os.path.join(root, file))
    return csv_files


def search_csv_and_save_results(shard_path, csv_file, text_columns):
    """
    Searches for text in a single CSV file and saves results to a file named <csv_filename>_results.csv.
    :param shard_path: Path to the shard file.
    :param csv_file: Path to the CSV file.
    :param text_columns: List of column names in the CSV to search for.
    """
    results = search_in_shard_from_csv(shard_path, csv_file, text_columns)
    output_file = f"/home/ekorukluoglu_umass_edu/beam2/BEAM/olmo-search/results/{os.path.splitext(csv_file)[0]}_results.csv"
    if results:
        result_df = pd.DataFrame(results, columns=["Matches"])
        result_df.to_csv(output_file, index=False)
        logging.info(f"Results for {csv_file} saved to {output_file}.")
    else:
        logging.info(f"No matches found for {csv_file}.")


def parallel_search_multiple_csvs_on_single_shard(shard_path, csv_files, text_columns, num_workers=4):
    """
    Parallely searches lines in multiple CSV files on a single shard.
    :param shard_path: Path to the shard file.
    :param csv_files: List of CSV file paths.
    :param text_columns: List of column names in CSV files to search for.
    :param num_workers: Number of parallel processes.
    """
    if not csv_files:
        raise ValueError("No CSV files provided for searching.")

    # Use multiprocessing to process each CSV file in parallel
    with Pool(processes=num_workers) as pool:
        pool.starmap(
            search_csv_and_save_results,
            [(shard_path, csv_file, text_columns) for csv_file in csv_files]
        )


if __name__ == "__main__":
    # Define the shard file and folder containing CSVs
    shard_path = "./shards/shard_00000000_processed.jsonl"  # Replace with the path to your shard file
    base_folder = "/home/ekorukluoglu_umass_edu/beam2/BEAM/scripts/Prompts/"  # Base folder with subfolders containing CSVs
    text_columns = ['en', 'tr', 'es', 'vi']  # Columns in CSV files to search for

    try:
        logging.info("Starting the search")
        # Get all CSV files from the base folder and its subfolders
        csv_files = get_all_csv_files(base_folder)

        # Perform the search and save results for each CSV file
        parallel_search_multiple_csvs_on_single_shard(shard_path, csv_files, text_columns, num_workers=6)

        logging.info("Search completed for all files.")
    except Exception as e:
        logging.error(f"Error: {e}")
