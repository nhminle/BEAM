import os
import csv
import tiktoken

def num_tokens_from_string(string: str, encoding_name: str = "o200k_base") -> int:
    """
    Returns the number of tokens in a text string using the specified encoding.

    Args:
    string (str): The text string to tokenize.
    encoding_name (str): The encoding name. Default is "o200k_base".

    Returns:
    int: The number of tokens in the text string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def count_token_in_file(file_path: str) -> int:
    """
    Reads the content of the file and counts the number of tokens.

    Args:
    file_path (str): The path to the file.

    Returns:
    int: The number of tokens in the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        return num_tokens_from_string(content)

def process_files_in_folder(folder_path: str, output_csv_path: str):
    """
    Processes each file in the specified folder, applies the count_token_in_file function, 
    and writes the results to a CSV file.

    Args:
    folder_path (str): The path to the folder containing the files.
    output_csv_path (str): The path to the output CSV file.
    """
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Filename', 'Token Count'])  # Write the header

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                try:
                    token_count = count_token_in_file(file_path)
                    csvwriter.writerow([filename, token_count])  # Write the filename and token count
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")

if __name__ == '__main__':
    folder_path = '/home/nhatminhle_umass_edu/preprocess_books/raw/tr'
    process_files_in_folder(folder_path, '/home/nhatminhle_umass_edu/preprocess_books/out.csv')