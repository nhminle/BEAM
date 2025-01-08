import os
import pandas as pd
import random

def get_folder_names(directory):
    folder_names = []
    for item in os.listdir(directory):
        print(str(item))
        item_path = os.path.join(directory, item)
        # Check if it's a file (not a folder)
        if os.path.isfile(item_path) and item.endswith('.csv'):  # Only CSV files
            folder_names.append(item)
    return folder_names

def shuffle_words(prompt):
    words = prompt.split()
    random.shuffle(words)
    return ' '.join(words)

def run_filter(title):
    # csv_file = os.path.join(directory, filename)  # Create the full path to the file
    df1 = pd.read_csv(f'/home/nhatminhle_umass_edu/Prompts/{title}.csv')
    # df = pd.read_csv(f'/home/nhatminhle_umass_edu/Filter-par3-alignment/ner/{title}/{title}_para_ner_filtered.csv')
    df1['sentiment'] = df1['sentiment'].reindex(df1.index, fill_value="Not Aligned")

    # Assuming the label is stored in a column named 'choices', filter the dataframe
    aligned_label = "Aligned 😊"

    # Filter the rows where the 'sentiment' column contains 'Aligned 😊'
    filtered_df = df1[df1['sentiment'].apply(lambda x: aligned_label in x if isinstance(x, str) else False)]
    filtered_df = filtered_df[['Single_ent', 'en', 'es', 'tr', 'vi']]
    
    # Shuffle words in the prompt columns
    filtered_df['en_prompts_shuffled'] = filtered_df['en'].apply(lambda x: shuffle_words(x) if isinstance(x, str) else x)
    filtered_df['es_prompts_shuffled'] = filtered_df['es'].apply(lambda x: shuffle_words(x) if isinstance(x, str) else x)
    filtered_df['tr_prompts_shuffled'] = filtered_df['tr'].apply(lambda x: shuffle_words(x) if isinstance(x, str) else x)
    filtered_df['vi_prompts_shuffled'] = filtered_df['vi'].apply(lambda x: shuffle_words(x) if isinstance(x, str) else x)

    filtered_csv_file=f'/home/nhatminhle_umass_edu/Prompts/{title}/{title}_filtered.csv'
    # filtered_df.to_csv(filtered_csv_file, index=False)
    filtered_df.to_csv(f'/home/nhatminhle_umass_edu/Prompts/Fahrenheit_451/Fahrenheit_451_filtered.csv', index=False)
    print(f'Filtered CSV saved to {filtered_csv_file}')

    # Filter the rows where the 'sentiment' column contains 'Aligned 😊'
    filtered_masked_df = df1[df1['sentiment'].apply(lambda x: aligned_label in x if isinstance(x, str) else False)]
    filtered_masked_df = filtered_masked_df[['Single_ent', 'en_masked', 'es_masked', 'tr_masked', 'vi_masked']]
    
    # Shuffle words in the prompt columns
    filtered_masked_df['en_masked_shuffled'] = filtered_masked_df['en_masked'].apply(lambda x: shuffle_words(x) if isinstance(x, str) else x)
    filtered_masked_df['es_masked_shuffled'] = filtered_masked_df['es_masked'].apply(lambda x: shuffle_words(x) if isinstance(x, str) else x)
    filtered_masked_df['tr_masked_shuffled'] = filtered_masked_df['tr_masked'].apply(lambda x: shuffle_words(x) if isinstance(x, str) else x)
    filtered_masked_df['vi_masked_shuffled'] = filtered_masked_df['vi_masked'].apply(lambda x: shuffle_words(x) if isinstance(x, str) else x)

    # filtered_masked_df.to_csv(filtered_csv_file.replace('_filtered.csv', '_filtered_masked.csv'), index=False)
    filtered_masked_df.to_csv('/home/nhatminhle_umass_edu/Prompts/Fahrenheit_451/Fahrenheit_451_filtered_masked.csv', index = False)
    print(f'Filtered CSV saved to {filtered_csv_file.replace('_filtered.csv', '_filtered_masked.csv')}')


if __name__ == "__main__":
    title = 'Fahrenheit_451'
    run_filter(title)

