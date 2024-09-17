import pandas as pd
import os

def filter(path):
    try:
        # Load the CSV file
        df = pd.read_csv(path)

        # Filter out rows where the 'Single_ent' column is empty
        filtered_df = df[df['Single_ent'].notna()]

        # Export the filtered data to a new CSV file
        filtered_df.to_csv(path.replace('_para_ner.csv', '_para_ner_filtered.csv'), index=False)

        print("Filtered data exported to 'filtered_output.csv'")
    except Exception as e:
        print(e)

def get_folder_names(directory):
    folder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names

if __name__ == '__main__':
    titles = get_folder_names('/aligner/ner')
    for title in titles:
        filter(path=f'/Filter-par3-alignment/ner/{title}/{title}_para_ner.csv')