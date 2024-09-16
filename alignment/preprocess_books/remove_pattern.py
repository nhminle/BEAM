import os

def remove_empty_lines(directory):
    # Iterate through all files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):  # Only process .txt files
            file_path = os.path.join(directory, filename)
            
            with open(file_path, 'r') as file:
                lines = file.readlines()  # Read all lines from the file
            
            # Remove empty lines
            cleaned_lines = [line for line in lines if line.strip()]
            
            # Write the cleaned content back to the file
            with open(file_path, 'w') as file:
                file.writelines(cleaned_lines)

# Example usage:
directory_path = 'preprocess_books/raw/en'  # Replace with the path to your directory
remove_empty_lines(directory_path)