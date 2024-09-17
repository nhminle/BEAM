import os
import pandas as pd

def get_folder_names(directory):
    folder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names

def delete_file_in_directory(title):
    # Construct the full file path
    file_path = f'/home/nhatminhle_umass_edu/Prompts/{title}/{title}_en_prompts.csv'
    
    # Check if the file exists
    if os.path.isfile(file_path):
        try:
            # Delete the file
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        except Exception as e:
            print(f"Error deleting file: {file_path}. Error: {e}")
    else:
        print(f"File not found: {file_path}")

def reduce_csv_files(file_path, title):
    try:            
        # Load the CSV file
        df = pd.read_csv(file_path)
        
        # Reduce to the first 50 and last 50 rows
        if len(df) > 100:
            df_reduced = pd.concat([df.head(50), df.tail(50)])
        else:
            df_reduced = df
        
        # Save the reduced CSV file, overwriting the original
        df_reduced.to_csv(f'/home/nhatminhle_umass_edu/Prompts/{title}/{title}_ner.csv', index=False)
        print(f'Reduced {title} to {len(df_reduced)} rows.')
    except Exception:
        print(Exception)

if __name__ == '__main__':
    skip_list = ['raw']
    titles = get_folder_names('/home/nhatminhle_umass_edu/Prompts')
    for title in titles:
        if title not in skip_list:
            reduce_csv_files(f'/home/nhatminhle_umass_edu/Prompts/raw/{title}/{title}_ner.csv', title)