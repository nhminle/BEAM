def remove_text_in_brackets(file_path):
    # Read the contents of the file
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Remove text in brackets
    import re
    updated_content = re.sub(r'\[\d+\]', '', content)

    # Write the updated content back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(updated_content)

# Replace 'your_file.txt' with the path to your file
remove_text_in_brackets('/home/nhatminhle_umass_edu/preprocess_books/raw/tr/Of_Mice_and_Men_tr.txt')
