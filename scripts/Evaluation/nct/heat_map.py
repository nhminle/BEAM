import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def extract_accuracy_from_csv(file_path):
    df = pd.read_csv(file_path)
    required_columns = ['en_correct', 'es_correct', 'vi_correct', 'tr_correct']
    
    if not all(col in df.columns for col in required_columns):
        print(f"Skipping {file_path}: Missing required language results.")
        return None
    
    accuracy = {
        'en': df['en_correct'].value_counts(normalize=True).get('correct', 0) * 100,
        'es': df['es_correct'].value_counts(normalize=True).get('correct', 0) * 100,
        'vi': df['vi_correct'].value_counts(normalize=True).get('correct', 0) * 100,
        'tr': df['tr_correct'].value_counts(normalize=True).get('correct', 0) * 100,
    }
    return accuracy

def create_heatmap(directory, model, release_date_csv, save_path, experiment, shuffled=False):
    release_dates = pd.read_csv(release_date_csv)
    release_dates['Release Date'] = pd.to_datetime(release_dates['Release Date'])  # Ensure datetime format

    all_accuracies = {}
    
    # Extract accuracies for each file in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            accuracies = extract_accuracy_from_csv(file_path)
            if accuracies is not None:
                if shuffled: 
                    book_title = filename.replace(f'_name_cloze_{model}_shuffled.csv', '')
                    book_title = book_title.replace(f'_nct_{model}_shuffled.csv', '')
                else:
                    book_title = filename.replace(f'_name_cloze_{model}.csv', '')
                    book_title = book_title.replace(f'_nct_{model}.csv', '')
                all_accuracies[book_title] = accuracies

    if not all_accuracies:
        print("No valid CSV files found with results for all four languages.")
        return

    accuracy_df = pd.DataFrame.from_dict(all_accuracies, orient='index')
    accuracy_df.reset_index(inplace=True)
    accuracy_df.columns = ['Title'] + list(accuracy_df.columns[1:])  

    print(f"Accuracy DataFrame shape: {accuracy_df.shape}")
    print(f"Release Dates DataFrame shape: {release_dates.shape}")

    # Merge with release dates
    merged_df = pd.merge(accuracy_df, release_dates, on='Title', how='inner')

    print(f"Merged DataFrame shape: {merged_df.shape}")

    # Exit early if no matching data
    if merged_df.empty:
        print("No matching titles between accuracies and release dates.")
        return

    # Sort by release date
    sorted_df = merged_df.sort_values('Release Date')

    print(f"Sorted DataFrame shape: {sorted_df.shape}")

    # Prepare heatmap data
    heatmap_data = sorted_df.set_index('Title').drop(columns=['Release Date'])

    print(f"Heatmap Data shape: {heatmap_data.shape}")

    # Exit early if heatmap data is empty
    if heatmap_data.empty:
        print("Heatmap data is empty. Cannot generate heatmap.")
        return
    
    custom_cmap = LinearSegmentedColormap.from_list(
    'custom_bupu', ['#f7fcfd', '#bfd3e6', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'], N=256
    )

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, cmap=custom_cmap, cbar=True, fmt='.1f', linewidths=.5, vmin=0, vmax=100)
    
    if shuffled:
        plt.title(f'{model}_shuffled_{experiment}', fontsize=16)
    else:
        plt.title(f'{model}_{experiment}', fontsize=16)
    plt.xlabel('Language', fontsize=16)
    plt.ylabel('Books (Sorted by Release Date)', fontsize=16)
    plt.tight_layout()
    if shuffled:
        plt.savefig(f'{save_path}/{model}_shuffled_{experiment}_heatmap.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'{save_path}/{model}_{experiment}_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()


experiment = 'nct' # nct or dir_probe
models = ['OLMo-7B-0724-Instruct-hf', 'Llama-3.1-70B-Instruct', 'Meta-Llama-3.1-8B-Instruct', 'llama405b', 'gpt4o'] # 'OLMo-7B-0724-Instruct-hf', 'Llama-3.1-70B-Instruct', 'Meta-Llama-3.1-8B-Instruct', 'llama405b'
for model in models:
    create_heatmap(directory=f'/Users/minhle/Umass/ersp/Evaluation/nct/eval/csv/{model}/{model}_shuffled', 
                   model=model, 
                   release_date_csv='/Users/minhle/Umass/ersp/Evaluation/nct/eval/csv/release_date.csv',
                   save_path='/Users/minhle/Umass/ersp/Evaluation/nct/eval',
                   experiment=experiment,
                   shuffled=True
                   )
