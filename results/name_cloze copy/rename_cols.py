import os
import pandas as pd

def rename_columns_in_csv(file_path, find_phrase, replace_phrase):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Replace the phrase in each column name
    new_columns = [col.replace(find_phrase, replace_phrase) for col in df.columns]
    df.columns = new_columns
    
    # Save the modified DataFrame back to the same file
    df.to_csv(file_path, index=False)
    print(f"Processed: {file_path}")
    
def rename_columns_in_directory(directory, find_phrase, replace_phrase):
    # Get a list of files in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith('.csv'):
            file_path = os.path.join(directory, filename)
            print(file_path)
            rename_columns_in_csv(file_path, find_phrase, replace_phrase)

if __name__ == "__main__":
    dirs_nct = [
        'results/name_cloze/EuroLLM-9B-Instruct',
        'results/name_cloze/gpt-4o-2024-11-20',
        'results/name_cloze/Llama-3.1-8B-Instruct_',
        'results/name_cloze/Llama-3.1-8B-Instruct-quantized.w4a16',
        'results/name_cloze/Llama-3.1-8B-Instruct-quantized.w8a16',
        'results/name_cloze/Llama-3.1-70B-Instruct-quantized.w4a16',
        'results/name_cloze/Llama-3.1-70B-Instruct-quantized.w8a16',
        'results/name_cloze/OLMo-2-1124-7B-Instruct',
        'results/name_cloze/Llama-3.1-70B-Instruct_',
        'results/name_cloze/Llama-3.3-70B-Instruct',
        'results/name_cloze/Llama-3.1-405b',
        'results/name_cloze/OLMo-2-1124-13B-Instruct',
        'results/name_cloze/Qwen2.5-7B-Instruct-1M'
    ]
    for d in dirs_nct: 
        for ps in ['zero-shot', 'one-shot']:
            rename_columns_in_directory(f'{d}/{ps}', '_masked', '')
    
    # def convert_files_without_extension_to_csv(directory):
    #     """
    #     Finds all files in the given directory that do not have an extension
    #     and renames them by appending '.csv' to the filename.
    #     """
    #     # Iterate over all items in the directory
    #     for filename in os.listdir(directory):
    #         file_path = os.path.join(directory, filename)
    #         # Process only files (skip directories)
    #         if os.path.isfile(file_path):
    #             # Split the filename into name and extension
    #             name, ext = os.path.splitext(filename)
    #             # If there's no extension, rename the file
    #             if not ext:
    #                 new_filename = name + ".csv"
    #                 new_file_path = os.path.join(directory, new_filename)
    #                 os.rename(file_path, new_file_path)
    #                 print(f"Renamed '{filename}' to '{new_filename}'")
    # convert_files_without_extension_to_csv('results/name_cloze/gpt-4o-2024-11-20/zero-shot')
