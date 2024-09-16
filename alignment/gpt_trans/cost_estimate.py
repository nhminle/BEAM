import json
import os
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

def extract_content_from_jsonl(jsonl_file_path):
    total_tokens = 0
    total_lines = 0
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line.strip())
            total_tokens += num_tokens_from_string(data.get('body', {}).get('messages', [])[0].get('content', ''))
            total_lines += 1

    return total_tokens, total_lines

def calculate_average_tokens(folder_path, title):
    print(f'~~~///Calculating {title}///~~~')
    total_tokens = 0
    total_lines = 0
    jsonl_files = [f for f in os.listdir(folder_path) if f.endswith('.jsonl')]

    for jsonl_file in jsonl_files:
        file_path = os.path.join(folder_path, jsonl_file)
        file_tokens, file_lines = extract_content_from_jsonl(file_path)
        print(f'{file_path}: {file_tokens} tokens')
        total_tokens += file_tokens
        total_lines += file_lines

    if total_lines == 0:
        print("No lines found in the files.")
        return

    avg_tokens_per_line = total_tokens / total_lines
    print(f"Total tokens in {title}: {total_tokens}")
    print(f"Average tokens per line in {title}: {round(avg_tokens_per_line)}")
    print(f'Projected token output per line in {title}: {round(avg_tokens_per_line/2)}')
    print(f"Total lines int {title}: {total_lines}")
    print(f'Projected cost: {round(total_tokens*5/1000000+(total_tokens-48)/2*15/1000000)/2}')
    print('\n')

def get_folder_names(directory):
    folder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names

if __name__ == '__main__':
    titles = get_folder_names('/home/nhatminhle_umass_edu/gpt_trans/input_json')
    for title in titles:
        input_folder = f'/home/nhatminhle_umass_edu/gpt_trans/input_json/{title}'
        calculate_average_tokens(input_folder, title)
