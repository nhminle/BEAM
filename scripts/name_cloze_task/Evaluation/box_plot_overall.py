import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# stack CSV files from a folder into one DataFrame for aggregate analysis
def stack_and_visualize(folder_path):
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    df_list = [pd.read_csv(file) for file in all_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    #specifying cols, makes life easier
    columns_to_plot = [
        'en_highest_fuzzy_match',
        'es_highest_fuzzy_match',
        'vi_highest_fuzzy_match',
        'tr_highest_fuzzy_match'
    ]
    combined_df = combined_df[columns_to_plot]
    sns.set(style="whitegrid")
    flare_palette = sns.color_palette("flare", len(columns_to_plot))

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=combined_df, palette=flare_palette, linewidth=1.5, width=0.6)
    plt.title("OLMo-7B-0724-Instruct Box Plot of Fuzzy Match Scores by Language", fontsize=16, fontweight='bold')
    plt.ylabel("Fuzzy Match Score", fontsize=12)
    plt.xlabel("Language", fontsize=12)
    plt.xticks(range(len(columns_to_plot)), ["en", "es", "vi", "tr"], fontsize=10, color='black')
    plt.yticks(fontsize=10, color='black')

    plt.tight_layout()
    plt.savefig("/Users/alishasrivastava/BEAM-scripts/BEAM/scripts/name_cloze_task/Evaluation/eval/OLMo-7B-0724-Instruct-hf_overall_nct_box_plot.png") #TODO: enter path
    plt.show()

if __name__ == '__main__':
    folder_path = "/Users/alishasrivastava/BEAM-scripts/BEAM/scripts/name_cloze_task/Evaluation/eval/csv/OLMo-7B-0724-Instruct-hf" #TODO: enter path
    stack_and_visualize(folder_path)
