import os
import pandas as pd

def get_folder_names(directory):
    folder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names

def main(title):
    # Read the first CSV file
    try: 
        df1 = pd.read_csv(f'/home/nhatminhle_umass_edu/aligner/ner/{title}/{title}_ner.csv')
        print(f'{title}: {len(df1)}\n')
    except:
        print(f'{title} is missing _ner.csv\n')

if __name__ == '__main__':
    titles = get_folder_names('/home/nhatminhle_umass_edu/aligner/ner')
    for title in titles:
        main(title)