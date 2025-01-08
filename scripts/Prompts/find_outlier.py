import pandas as pd
import tiktoken
import os

# Load the tokenization model
tokenizer = tiktoken.get_encoding("o200k_base")  # Adjust based on your tokenizer

def find_long_passage(csv_files, column_name='vi', target_token_count=600):
    for file in csv_files:
        try:
            # print(f"Processing {file}...")
            csv = pd.read_csv(file)

            # Check if the target column exists
            if column_name not in csv.columns:
                # print(f"Skipping {file} - '{column_name}' column not found.")
                continue

            # Calculate token counts for the target column
            def token_count(text):
                return len(tokenizer.encode(text))
            
            csv['token_count'] = csv[column_name].dropna().apply(token_count)

            # Check for rows with the target token count
            long_passages = csv[csv['token_count'] >= target_token_count]
            if not long_passages.empty:
                print(f"Found passages in {file}:")
                print(long_passages[['token_count', column_name]])
        
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue

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
    os.path.join(base_path, title, f"{title}_non_NE.csv") 
    for title in get_folder_names(base_path)
]

# Find the vi passage with 600 tokens
find_long_passage(csv_files, column_name='vi', target_token_count=800)
