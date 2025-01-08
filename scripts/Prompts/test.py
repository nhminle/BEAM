import csv
import os

def count_csv_rows(file_list):
    for file in file_list:
        try:
            with open(file, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                # Skip the header row
                header = next(reader, None)
                row_count = sum(1 for _ in reader)
                print(f"{os.path.basename(file).replace('_filtered_sampled.csv', '')}: {row_count}")
        except FileNotFoundError:
            print(f"File not found: {file}")
        except Exception as e:
            print(f"Error processing file {file}: {e}")

# Example usage
def get_folder_names(directory):
    folder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names

# Main logic
base_path = '/home/nhatminhle_umass_edu/Prompts'
csv_files = [
    os.path.join(base_path, title, f"{title}_filtered_sampled.csv") 
    for title in get_folder_names(base_path)
]

count_csv_rows(csv_files)
