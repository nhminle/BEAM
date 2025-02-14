import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def extract_total_accuracy_from_dir(directory, prompt_setting, shuffled=False):
    total_counts = {'en': 0, 'es': 0, 'vi': 0, 'tr': 0}
    correct_counts = {'en': 0, 'es': 0, 'vi': 0, 'tr': 0}

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            # Check if filename matches prompt setting and shuffled/unshuffled
            if prompt_setting not in filename:
                continue
            if shuffled and 'shuffled' not in filename:
                continue
            if not shuffled and 'shuffled' in filename:
                continue

            file_path = os.path.join(directory, filename)
            df = pd.read_csv(file_path)

            required_columns = ['en_correct', 'es_correct', 'vi_correct', 'tr_correct']
            if not all(col in df.columns for col in required_columns):
                print(f"⚠️ Missing required columns in {file_path}")
                continue

            for lang in ['en', 'es', 'vi', 'tr']:
                total_counts[lang] += len(df[lang + '_correct'])
                correct_counts[lang] += (df[lang + '_correct'] == 'correct').sum()

    accuracy = {lang: (correct_counts[lang] / total_counts[lang] * 100) if total_counts[lang] > 0 else 0
                for lang in ['en', 'es', 'vi', 'tr']}
    
    return accuracy

def create_dir_accuracy_heatmap(directories, save_path, prompt_setting, shuffled):
    accuracies = {}

    for directory in directories:
        dir_name = directory.split('/')[-4]  # Extract model name from path
        accuracies[dir_name] = extract_total_accuracy_from_dir(directory, prompt_setting, shuffled)

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
    shuffle_type = 'Shuffled' if shuffled else 'Unshuffled'
    plt.title(f'Name Cloze Task Accuracy ({shot_type}, {shuffle_type})', fontsize=14)
    plt.xlabel('Language', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.tight_layout()

    save_filename = f"{save_path}/total_accuracy_{prompt_setting}_{'shuffled' if shuffled else 'unshuffled'}.png"
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

# Run for both one-shot and zero-shot, and shuffled/unshuffled combinations
for prompt_setting in ['one-shot', 'zero-shot']:
    for shuffled in [False, True]:
        create_dir_accuracy_heatmap(directories, save_path, prompt_setting, shuffled)
