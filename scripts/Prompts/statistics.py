# import pandas as pd
# import os
# from collections import Counter
# import ast

# def analyze_csv_files(file_paths):
#     # Initialize variables to collect data
#     all_names = []
#     total_entries = 0

#     # Process each CSV file
#     for file_path in file_paths:
#         try:
#             # Read the CSV file
#             df = pd.read_csv(file_path)
            
#             # Check if 'Single_ent' column exists
#             if 'Single_ent' not in df.columns:
#                 print(f"Skipping {file_path}: 'Single_ent' column not found.")
#                 continue

#             # Process each row in the 'Single_ent' column
#             for entry in df['Single_ent']:
#                 try:
#                     # Parse the string representation of the list
#                     name_list = ast.literal_eval(entry)

#                     # Ensure it's a list
#                     if isinstance(name_list, list):
#                         all_names.extend(name_list)
#                         total_entries += 1
#                 except (ValueError, SyntaxError):
#                     print(f"Skipping malformed entry in {file_path}: {entry}")

#         except Exception as e:
#             print(f"Error reading {file_path}: {e}")

#     # Calculate statistics
#     name_counts = Counter(all_names)
#     unique_names = len(name_counts)
#     most_common_names = name_counts.most_common(5)

#     # Print statistics
#     print(f"Total entries (lists) across all files: {total_entries}")
#     print(f"Total unique names: {unique_names}")
#     print("Top 5 most popular names and their counts:")
#     for name, count in most_common_names:
#         print(f"  {name}: {count}")

#     # Additional statistics
#     average_names_per_entry = len(all_names) / total_entries if total_entries > 0 else 0
#     print(f"Average number of names per entry: {average_names_per_entry:.2f}")

#     # Return the raw data and stats for further use if needed
#     return {
#         "total_entries": total_entries,
#         "unique_names": unique_names,
#         "most_common_names": most_common_names,
#         "average_names_per_entry": average_names_per_entry,
#     }

# def get_folder_names(directory):
#     folder_names = []
#     for item in os.listdir(directory):
#         item_path = os.path.join(directory, item)
#         if os.path.isdir(item_path):
#             folder_names.append(item)
#     return folder_names

# # Example usage
# if __name__ == "__main__":
#     # List of CSV file paths
#     file_paths = [f'/home/nhatminhle_umass_edu/Prompts/{name}/{name}_filtered.csv' for name in get_folder_names('/home/nhatminhle_umass_edu/Prompts')]

#     # Analyze the files
#     stats = analyze_csv_files(file_paths)

# import pandas as pd
# import matplotlib.pyplot as plt
# import tiktoken
# import os

# # Load the tokenization model
# tokenizer = tiktoken.get_encoding("o200k_base")  # Adjust based on your tokenizer

# # Function to calculate token distribution and plot histogram for multiple columns
# def plot_tok_distribution(csv_files, columns_to_analyze, title,
#                           output_plot='combined_token_distribution_histogram.png', bins=100):
#     all_token_counts = {col: [] for col in columns_to_analyze}

#     for file in csv_files:
#         try:
#             print(f"Processing {file}...")
#             csv = pd.read_csv(file)
            
#             # Check if all required columns exist
#             if not all(col in csv.columns for col in columns_to_analyze):
#                 print(f"Skipping {file} - Missing one or more required columns: {columns_to_analyze}")
#                 continue
            
#             # Calculate token counts for each column
#             for column_name in columns_to_analyze:
#                 def token_count(text):
#                     return len(tokenizer.encode(text))
                
#                 csv[f'{column_name}_token_count'] = csv[column_name].dropna().apply(token_count)
#                 all_token_counts[column_name].extend(csv[f'{column_name}_token_count'].tolist())
        
#         except Exception as e:
#             print(f"Error processing {file}: {e}")
#             continue

#     if not any(all_token_counts.values()):
#         print("No token counts calculated. Exiting.")
#         return

#     plt.figure(figsize=(12, 8))
#     for column_name, token_counts in all_token_counts.items():
#         plt.hist(token_counts, bins=bins, alpha=0.8, label=f'{column_name} Tokens', edgecolor='black')
    
#     plt.title(title, fontsize=16)
#     plt.xlabel('Number of Tokens', fontsize=14)
#     plt.ylabel('Frequency', fontsize=14)
#     plt.legend(loc='upper right')
#     plt.xlim(0, 650)

#     # Save the plot
#     plt.savefig(output_plot, dpi=300, bbox_inches='tight')
#     print(f"Histogram saved as '{output_plot}'.")

#     # Show the plot
#     plt.show()

# # Function to get folder names from a specified directory
# def get_folder_names(directory):
#     folder_names = []
#     for item in os.listdir(directory):
#         item_path = os.path.join(directory, item)
#         if os.path.isdir(item_path):
#             folder_names.append(item)
#     return folder_names

# # Main logic
# base_path = '/home/nhatminhle_umass_edu/Prompts'
# csv_files = [
#     os.path.join(base_path, title, f"{title}_filtered_sampled.csv") 
#     for title in get_folder_names(base_path)
# ]

# columns_to_analyze = ['vi', 'tr', 'es', 'en']  # List of columns to analyze
# plot_tok_distribution(
#     csv_files,
#     columns_to_analyze,
#     title = 'Token Count Distribution Across Aligned Passages WITH NE (excluding 2024 books)',
#     output_plot='/home/nhatminhle_umass_edu/Prompts/combined_token_distribution_histogram_NE.png',
#     bins=100
# )

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
base_path = '/home/nhatminhle_umass_edu/Prompts'

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
    output_plot='/home/nhatminhle_umass_edu/Prompts/boxplot_with_stats_horizontal.png'
)
