import os
import csv

def count_csv_lines_in_dir(root_dir):
    """
    Count the total number of lines in all CSV files in the given directory and its subdirectories.

    :param root_dir: Root directory to search for CSV files
    :return: A dictionary with subdirectory paths as keys and line counts as values
    """
    result = {}

    for subdir, _, files in os.walk(root_dir):
        csv_line_count = 0
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(subdir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        line_count = sum(1 for _ in reader)
                        csv_line_count += line_count
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

        if csv_line_count > 0:
            result[subdir] = csv_line_count

    return result

if __name__ == "__main__":
    # Replace this with the directory you want to analyze
    directory = input("Enter the directory to search: ")

    if os.path.exists(directory):
        line_counts = count_csv_lines_in_dir(directory)
        if line_counts:
            print("\nCSV Line Counts by Subdirectory:")
            for subdir, count in line_counts.items():
                print(f"{subdir}: {count} lines")
            print(sum([int(i) for i in line_counts.values()]))
        else:
            print("No CSV files found.")
    else:
        print("The specified directory does not exist.")
