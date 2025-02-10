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
            required_columns = ['en_results_both_match', 'es_results_both_match', 'vi_results_both_match', 'tr_results_both_match']
            
            if not all(col in df.columns for col in required_columns):
                print(f"Missing required columns in: {file_path}")
                continue
            
            for lang in ['en', 'es', 'vi', 'tr']:
                total_counts[lang] += len(df[lang + '_results_both_match'])
                correct_counts[lang] += (df[lang + '_results_both_match'] == True).sum()
                
    accuracy = {lang: (correct_counts[lang] / total_counts[lang] * 100) if total_counts[lang] > 0 else 0
                for lang in ['en', 'es', 'vi', 'tr']}
    
    return accuracy

def create_dir_accuracy_heatmap(directories, save_path, y_labels=None):
    accuracies = {}
    
    for i, directory in enumerate(directories):
        # Use the provided y_labels if available; otherwise, use the directory's basename.
        if y_labels is not None:
            dir_label = y_labels[i]
        else:
            dir_label = os.path.basename(directory.rstrip('/'))
        accuracies[dir_label] = extract_total_accuracy_from_dir(directory)
    
    accuracy_df = pd.DataFrame.from_dict(accuracies, orient='index')
    accuracy_df.reset_index(inplace=True)
    accuracy_df.columns = ['Model'] + list(accuracy_df.columns[1:])
    
    heatmap_data = accuracy_df.set_index('Model')
    
    custom_cmap = LinearSegmentedColormap.from_list(
        'custom_bupu', ['#f7fcfd', '#bfd3e6', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'], N=256
    )
    
    plt.figure(figsize=(8, len(heatmap_data) * 0.6))
    sns.heatmap(heatmap_data, annot=True, cmap=custom_cmap, cbar=True, fmt='.1f', linewidths=.5, vmin=0, vmax=100)
    plt.title('Direct Probe Accuracy Aggregated', fontsize=16)
    plt.xlabel('Language', fontsize=16)
    plt.ylabel('Models', fontsize=16)
    plt.tight_layout()
    save_file = os.path.join(save_path, 'total_accuracy_heatmap.png')
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Heatmap saved to {save_file}")

directories = [
    '/Users/minhle/Umass/ersp/Evaluation/dir_probe/eval/csv/Llama-3.1-70B-Instruct',
    '/Users/minhle/Umass/ersp/Evaluation/dir_probe/eval/csv/Meta-Llama-3.1-8B-Instruct',
    '/Users/minhle/Umass/ersp/Evaluation/dir_probe/eval/csv/OLMo-7B-0724-Instruct-hf'
]

# Custom labels for the y-axis (should match the order/length of `directories`)
custom_labels = ['Llama 3.1 70B', 'Meta Llama 3.1 8B', 'OLMo 7B 0724']

save_path = '/Users/minhle/Umass/ersp/Evaluation/dir_probe/eval'
create_dir_accuracy_heatmap(directories, save_path, y_labels=custom_labels)
