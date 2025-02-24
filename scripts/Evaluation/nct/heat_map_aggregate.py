import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def extract_total_accuracy(directory, prompt_setting):
    """Extract accuracy for regular and shuffled results."""
    total_counts = {'en': 0, 'es': 0, 'vi': 0, 'tr': 0, 
                    'en_shuffled': 0, 'es_shuffled': 0, 'vi_shuffled': 0, 'tr_shuffled': 0}
    correct_counts = {'en': 0, 'es': 0, 'vi': 0, 'tr': 0, 
                      'en_shuffled': 0, 'es_shuffled': 0, 'vi_shuffled': 0, 'tr_shuffled': 0}

    # List all files in directory
    files = os.listdir(directory)

    # Filter only files matching the prompt setting
    csv_files = [f for f in files if f.endswith('.csv') and prompt_setting in f]

    for filename in csv_files:
        # Identify corresponding shuffled file
        shuffled_filename = filename.replace('.csv', '_shuffled.csv')
        
        # Read unshuffled data
        file_path = os.path.join(directory, filename)
        df = pd.read_csv(file_path)

        # Try to read shuffled data if exists
        shuffled_path = os.path.join(directory, shuffled_filename)
        df_shuffled = pd.read_csv(shuffled_path) if os.path.exists(shuffled_path) else pd.DataFrame()

        required_columns = [f'{lang}_correct' for lang in ['en', 'es', 'vi', 'tr']]
        if not all(col in df.columns for col in required_columns):
            print(f"⚠️ Missing required columns in {file_path}")
            continue

        # Process regular data
        for lang in ['en', 'es', 'vi', 'tr']:
            total_counts[lang] += len(df[f'{lang}_correct'])
            correct_counts[lang] += (df[f'{lang}_correct'] == 'correct').sum()

        # Process shuffled data if available
        if not df_shuffled.empty:
            for lang in ['en', 'es', 'vi', 'tr']:
                if f'{lang}_correct' in df_shuffled.columns:
                    total_counts[f'{lang}_shuffled'] += len(df_shuffled[f'{lang}_correct'])
                    correct_counts[f'{lang}_shuffled'] += (df_shuffled[f'{lang}_correct'] == 'correct').sum()

    # Calculate accuracy percentages
    accuracy = {lang: (correct_counts[lang] / total_counts[lang] * 100) if total_counts[lang] > 0 else 0
                for lang in total_counts.keys()}
    
    return accuracy

def create_dir_accuracy_heatmap(directories, save_path, prompt_setting):
    accuracies = {}

    for directory in directories:
        dir_name = directory.split('/')[-4]  # Extract model name from path
        accuracies[dir_name] = extract_total_accuracy(directory, prompt_setting)

    accuracy_df = pd.DataFrame.from_dict(accuracies, orient='index')
    accuracy_df.reset_index(inplace=True)
    accuracy_df.columns = ['Model'] + list(accuracy_df.columns[1:])

    heatmap_data = accuracy_df.set_index('Model')

    custom_cmap = LinearSegmentedColormap.from_list(
        'custom_bupu', ['#f7fcfd', '#bfd3e6', '#8c96c6', '#8c6bb1', '#88419d', '#810f7c', '#4d004b'], N=256
    )

    plt.figure(figsize=(10, len(directories) * 0.5))
    sns.heatmap(
        heatmap_data,
        annot=True,
        cmap=custom_cmap,
        cbar=True,
        fmt='.1f',
        linewidths=.5,
        vmin=0,
        vmax=100
    )

    shot_type = 'One-Shot' if 'one-shot' in prompt_setting else 'Zero-Shot'
    plt.title(f'Name Cloze Task Accuracy ({shot_type})', fontsize=14)
    plt.xlabel('Language', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()

    save_filename = f"{save_path}/total_accuracy_{prompt_setting}.png"
    plt.savefig(save_filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Heatmap saved to {save_filename}")


# Example Usage
directories = [
    '/Users/emir/Downloads/asd/BEAM/results/name_cloze/EuroLLM-9B-Instruct/eval/csv/',
    '/Users/emir/Downloads/asd/BEAM/results/name_cloze/gpt-4o-2024-11-20/eval/csv/',
    '/Users/emir/Downloads/asd/BEAM/results/name_cloze/Llama-3.1-8B-Instruct/eval/csv/',
    '/Users/emir/Downloads/asd/BEAM/results/name_cloze/Llama-3.1-8B-Instruct-quantized.w4a16/eval/csv/',
    '/Users/emir/Downloads/asd/BEAM/results/name_cloze/Llama-3.1-8B-Instruct-quantized.w8a16/eval/csv/',
    '/Users/emir/Downloads/asd/BEAM/results/name_cloze/Llama-3.1-70B-Instruct/eval/csv/',
    '/Users/emir/Downloads/asd/BEAM/results/name_cloze/Llama-3.1-70B-Instruct-quantized.w4a16/eval/csv/',
    '/Users/emir/Downloads/asd/BEAM/results/name_cloze/Llama-3.1-70B-Instruct-quantized.w8a16/eval/csv/',
    '/Users/emir/Downloads/asd/BEAM/results/name_cloze/Llama-3.1-405b/eval/csv/',
    '/Users/emir/Downloads/asd/BEAM/results/name_cloze/Llama-3.3-70B-Instruct/eval/csv/',
    '/Users/emir/Downloads/asd/BEAM/results/name_cloze/OLMo-2-1124-7B-Instruct/eval/csv/',
    '/Users/emir/Downloads/asd/BEAM/results/name_cloze/OLMo-2-1124-13B-Instruct/eval/csv/',
    '/Users/emir/Downloads/asd/BEAM/results/name_cloze/Qwen2.5-7B-Instruct-1M/eval/csv/',
]

save_path = '/Users/emir/Downloads/asd/BEAM/results/visualizations/score_table_overall'

# Run for both one-shot and zero-shot
for prompt_setting in ['one-shot', 'zero-shot']:
    create_dir_accuracy_heatmap(directories, save_path, prompt_setting)
