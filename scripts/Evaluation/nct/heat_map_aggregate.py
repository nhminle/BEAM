import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def extract_total_accuracy_from_dir(directory):
    total_counts = {'en': 0, 'es': 0, 'vi': 0, 'tr': 0}
    correct_counts = {'en': 0, 'es': 0, 'vi': 0, 'tr': 0}
    
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)
            required_columns = ['en_correct', 'es_correct', 'vi_correct', 'tr_correct']
            
            if not all(col in df.columns for col in required_columns):
                print(file_path)
                continue
            
            for lang in ['en', 'es', 'vi', 'tr']:
                total_counts[lang] += len(df[lang + '_correct'])
                correct_counts[lang] += (df[lang + '_correct'] == 'correct').sum()
    
    accuracy = {lang: (correct_counts[lang] / total_counts[lang] * 100) if total_counts[lang] > 0 else 0
                for lang in ['en', 'es', 'vi', 'tr']}
    
    return accuracy

def create_dir_accuracy_heatmap(directories, save_path):
    accuracies = {}
    
    for directory in directories:
        dir_name = os.path.basename(directory.rstrip('/'))
        accuracies[dir_name] = extract_total_accuracy_from_dir(directory)
    
    accuracy_df = pd.DataFrame.from_dict(accuracies, orient='index')
    accuracy_df.reset_index(inplace=True)
    accuracy_df.columns = ['Directory'] + list(accuracy_df.columns[1:])
    
    heatmap_data = accuracy_df.set_index('Directory')
    
    custom_cmap = LinearSegmentedColormap.from_list(
        'custom_bupu', ['#f7fcfd', '#bfd3e6', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'], N=256
    )
    
    plt.figure(figsize=(8, len(directories) * 0.6))
    sns.heatmap(heatmap_data, annot=True, cmap=custom_cmap, cbar=True, fmt='.1f', linewidths=.5, vmin=0, vmax=100)
    plt.title('Name cloze task accuracy aggregated', fontsize=16)
    plt.xlabel('Language', fontsize=16)
    plt.ylabel('Models', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_path}/total_accuracy_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

# Example Usage
directories = [
    '/Users/minhle/Umass/ersp/Evaluation/nct/eval/csv/gpt4o',
    '/Users/minhle/Umass/ersp/Evaluation/nct/eval/csv/llama405b',
    '/Users/minhle/Umass/ersp/Evaluation/nct/eval/csv/Llama-3.1-70B-Instruct',
    '/Users/minhle/Umass/ersp/Evaluation/nct/eval/csv/Meta-Llama-3.1-8B-Instruct',
    '/Users/minhle/Umass/ersp/Evaluation/nct/eval/csv/OLMo-7B-0724-Instruct-hf',
    '/Users/minhle/Umass/ersp/Evaluation/nct/eval/csv/gpt4o/gpt4o_shuffled',
    '/Users/minhle/Umass/ersp/Evaluation/nct/eval/csv/Llama-3.1-70B-Instruct/Llama-3.1-70B-Instruct_shuffled',
    '/Users/minhle/Umass/ersp/Evaluation/nct/eval/csv/llama405b/llama405b_shuffled',
    '/Users/minhle/Umass/ersp/Evaluation/nct/eval/csv/OLMo-7B-0724-Instruct-hf/OLMo-7B-0724-Instruct-hf_shuffled'
]
save_path = '/Users/minhle/Umass/ersp/Evaluation/nct/eval'
create_dir_accuracy_heatmap(directories, save_path)
