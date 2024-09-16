import json
import os

def create_prompt(passage):
    return f"""Carefully read and translate the following passage into English, preserving the tags:
        <passage>{passage}</passage>

        Use the following format as output:
        <passage><t#>Your translation</t#></passage> """

def convert(file_path, output_folder):
    tasks = []
    with open(file_path, 'r', encoding='utf-8') as file:
        task_no = 1
        for line in file:
            stripped_line = line.strip()
            # Ignore empty lines
            if stripped_line:
                task = {"custom_id": f"request-{task_no}", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": create_prompt(stripped_line)}],"max_tokens": 4000, "temperature":0.3}}
                task_no += 1
                tasks.append(task)

    outfile_name = os.path.join(output_folder, os.path.basename(file_path).replace('.txt', '.jsonl'))
    print(outfile_name)
    with open(outfile_name, "w", encoding='utf-8') as file:
        for obj in tasks:
            file.write(json.dumps(obj) + '\n')

def convert_folder(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            convert(file_path, output_folder)

def get_folder_names(directory):
    folder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names

if __name__ == "__main__":
    titles = get_folder_names('/home/nhatminhle_umass_edu/gpt_trans/raw_txt')
    for book_title in titles:
        input_folder = f'/home/nhatminhle_umass_edu/gpt_trans/raw_txt/{book_title}'
        output_folder = f'/home/nhatminhle_umass_edu/gpt_trans/input_json/{book_title}'
        convert_folder(input_folder, output_folder)
