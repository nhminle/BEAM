import random
import pandas as pd 
import os

def shuffle_words(prompt):
    words = prompt.split()
    random.shuffle(words)
    return ' '.join(words)

def run_filter(title):
    df1 = pd.read_csv(f'/home/nhatminhle_umass_edu/Prompts/{title}/{title}_filtered.csv')
    df2 = pd.read_csv(f'/home/nhatminhle_umass_edu/Prompts/{title}/{title}_filtered_masked.csv')

    if 'en_prompts_shuffled' not in df1.columns:
        try:
            df1['en_prompts_shuffled'] = df1['en'].apply(lambda x: shuffle_words(x) if isinstance(x, str) else x)
            df1['es_prompts_shuffled'] = df1['es'].apply(lambda x: shuffle_words(x) if isinstance(x, str) else x)
            df1['tr_prompts_shuffled'] = df1['tr'].apply(lambda x: shuffle_words(x) if isinstance(x, str) else x)
            df1['vi_prompts_shuffled'] = df1['vi'].apply(lambda x: shuffle_words(x) if isinstance(x, str) else x)
        except Exception as e:
            print(e)
        df1.to_csv(f'/home/nhatminhle_umass_edu/Prompts/{title}/{title}_filtered.csv', index=False)
        print(f'shuffled {title}_filtered.csv')
    else:
        print(f'{title}_filtered.csv already shuffled')
    
    if 'en_masked_shuffled' not in df2.columns:
        try:
            df2['en_masked_shuffled'] = df2['en_masked'].apply(lambda x: shuffle_words(x) if isinstance(x, str) else x)
            df2['es_masked_shuffled'] = df2['es_masked'].apply(lambda x: shuffle_words(x) if isinstance(x, str) else x)
            df2['tr_masked_shuffled'] = df2['tr_masked'].apply(lambda x: shuffle_words(x) if isinstance(x, str) else x)
            df2['vi_masked_shuffled'] = df2['vi_masked'].apply(lambda x: shuffle_words(x) if isinstance(x, str) else x)
        except Exception as e:
            print(e)
        df2.to_csv(f'/home/nhatminhle_umass_edu/Prompts/{title}/{title}_filtered_masked.csv', index=False)
        print(f'shuffled {title}_filtered_masked.csv')
    else:
        print(f'{title}_filtered_masked.csv already shuffled')

def get_folder_names(directory):
    folder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names

titles = get_folder_names('/home/nhatminhle_umass_edu/Prompts')
for title in titles:
    run_filter(title)