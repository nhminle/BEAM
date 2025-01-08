import pandas as pd
import matplotlib.pyplot as plt
import tiktoken
import os
import numpy as np
from scipy.stats import ks_2samp, mannwhitneyu

# Load the tokenization model
tokenizer = tiktoken.get_encoding("o200k_base")  # Adjust based on your tokenizer

# Function to calculate token distribution, plot horizontal box plots, and display stats outside
def plot_horizontal_tok_distribution(csv_files_list1, csv_files_list2, columns_to_analyze, title, output_plot='boxplot_with_stats_horizontal.png'):
    all_token_counts_list1 = {col: [] for col in columns_to_analyze}
    all_token_counts_list2 = {col: [] for col in columns_to_analyze}

    def process_files(csv_files, all_token_counts):
        for file in csv_files:
            try:
                print(f"Processing {file}...")
                csv = pd.read_csv(file)

                # Check if all required columns exist
                if not all(col in csv.columns for col in columns_to_analyze):
                    print(f"Skipping {file} - Missing one or more required columns: {columns_to_analyze}")
                    continue

                # Calculate token counts for each column
                for column_name in columns_to_analyze:
                    def token_count(text):
                        return len(tokenizer.encode(text))
                    
                    token_counts = csv[column_name].dropna().apply(token_count).tolist()
                    all_token_counts[column_name].extend(token_counts)
            
            except Exception as e:
                print(f"Error processing {file}: {e}")
                continue

    # Process both lists of files
    process_files(csv_files_list1, all_token_counts_list1)
    process_files(csv_files_list2, all_token_counts_list2)

    # Generate horizontal box plot with statistical tests
    fig, axes = plt.subplots(len(columns_to_analyze), 1, figsize=(14, 20), sharex=True)
    for idx, column_name in enumerate(columns_to_analyze):
        data1 = all_token_counts_list1[column_name]
        data2 = all_token_counts_list2[column_name]
        
        # Compute basic statistics
        mean1, median1, std1 = np.mean(data1), np.median(data1), np.std(data1)
        mean2, median2, std2 = np.mean(data2), np.median(data2), np.std(data2)
        
        # Perform statistical tests
        ks_stat, ks_p = ks_2samp(data1, data2)
        mw_stat, mw_p = mannwhitneyu(data1, data2, alternative='two-sided')
        
        # Prepare stats text
        stats_text = (
            f"{column_name}:\n"
            f"NE - Mean: {mean1:.2f}, Median: {median1:.2f}, Std Dev: {std1:.2f}\n"
            f"non NE - Mean: {mean2:.2f}, Median: {median2:.2f}, Std Dev: {std2:.2f}\n"
            f"KS Test: Stat={ks_stat:.3f}, P={ks_p:.3f} {'(Significant)' if ks_p < 0.05 else '(Not Significant)'}\n"
            f"MW U Test: Stat={mw_stat:.3f}, P={mw_p:.3f} {'(Significant)' if mw_p < 0.05 else '(Not Significant)'}"
        )
        
        # Create horizontal box plots
        ax = axes[idx]
        ax.boxplot([data1, data2], vert=False, labels=[f'NE - {column_name}', f'non NE - {column_name}'])
        ax.set_title(f'Comparison of Token Counts for Column: {column_name}', fontsize=14)
        ax.set_xlabel('Token Count', fontsize=12)

        # Add stats text to the right of the plot
        fig.text(0.75, (len(columns_to_analyze) - idx - 0.5) / len(columns_to_analyze), stats_text, fontsize=10, verticalalignment='center')

    plt.suptitle(title, fontsize=18, x=0.4)
    plt.tight_layout(rect=[0, 0, 0.7, 0.96])  # Leave space for stats on the right

    # Save the plot
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Box plot with stats saved as '{output_plot}'.")

    # Show the plot
    plt.show()

# Function to get folder names from a specified directory
def get_folder_names(directory):
    folder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names

# Main logic
base_path = 'Prompts'

csv_files_list1 = [
    os.path.join(base_path, title, f"{title}_filtered_sampled.csv") 
    for title in get_folder_names(base_path)
]

csv_files_list2 = [
    os.path.join(base_path, title, f"{title}_non_NE.csv") 
    for title in get_folder_names(base_path)
]

columns_to_analyze = ['en', 'es', 'vi', 'tr']  # List of columns to analyze
plot_horizontal_tok_distribution(
    csv_files_list1,
    csv_files_list2,
    columns_to_analyze,
    title='Token Count Distribution Comparison with Statistical Tests',
    output_plot='Prompts/boxplot_with_stats_horizontal.png'
)
