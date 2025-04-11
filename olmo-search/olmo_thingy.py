import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def clean_book_name(filename):
    """
    Remove known substrings from the filename to extract the clean book name.
    It removes:
      - the '.csv' extension,
      - '_non_NE'
      - '_unmasked_passages'
    """
    # Remove the file extension if present.
    if filename.endswith('.csv'):
        name = filename[:-4]
    else:
        name = filename
    # Remove undesired substrings.
    name = name.replace("_non_NE", "").replace("_unmasked_passages", "")
    return name

def create_book_model_heatmap(occurrence_csv, accuracy_csv, save_path):
    # -------------------------------------------------------------
    # 1. Load occurrence CSV and create a sorted ordering of books
    # -------------------------------------------------------------
    df_occ = pd.read_csv(occurrence_csv)
    
    # Make sure the occurrence CSV contains a "Filename" column.
    if "filename" not in df_occ.columns:
        raise ValueError("Occurrence CSV must have a 'filename' column.")
    
    # Create a new column with the cleaned book name
    df_occ['Book_Name'] = df_occ['filename'].apply(clean_book_name)
    
    # Determine which column to use as the occurrence metric.
    # (Possible names include 'Occurrences', 'Count', or 'Percentage'.)
    if 'Occurrences' in df_occ.columns:
        metric = 'Occurrences'
    elif 'Count' in df_occ.columns:
        metric = 'Count'
    elif 'Percentage' in df_occ.columns:
        metric = 'Percentage'
    else:
        # Fallback: use the second column.
        metric = df_occ.columns[1]
    
    # Sort the occurrence DataFrame by the chosen metric (descending).
    df_occ_sorted = df_occ.sort_values(by=metric, ascending=False)
    sorted_books = df_occ_sorted['Book_Name'].tolist()
    
    # -------------------------------------------------------------
    # 2. Load Aggregated Accuracy CSV and reorder rows using the occurrence data
    # -------------------------------------------------------------
    df_acc = pd.read_csv(accuracy_csv)
    
    # Verify the aggregated CSV has a "Book Name" column; then set it as the index.
    if "Book Name" not in df_acc.columns:
        raise ValueError("Aggregated CSV must contain a 'Book Name' column.")
    df_acc.set_index("Book Name", inplace=True)
    
    # Now, filter and order the rows (books) to match the ordering from the occurrence data.
    # Only include books that appear in the aggregated accuracy CSV.
    ordered_books = [book for book in sorted_books if book in df_acc.index]
    df_acc = df_acc.loc[ordered_books]
    
    # -------------------------------------------------------------
    # 3. Create the heatmap (x-axis: model names, y-axis: book names)
    # -------------------------------------------------------------
    # Create a custom colormap (this is inspired by your snippet)
    custom_cmap = LinearSegmentedColormap.from_list(
        'custom_cmap', 
        ['#f7fcfd', '#ccece6', '#66c2a4', '#238b45', '#005824'], 
        N=256
    )
    
    plt.figure(figsize=(12, len(ordered_books)*0.5 + 2))
    ax = sns.heatmap(
        df_acc,
        annot=True,
        cmap=custom_cmap,
        cbar=True,
        fmt='.1f',
        linewidths=0.5,
        vmin=0,
        vmax=100
    )
    
    # Set axis labels: x-axis as model names (i.e. the remaining columns) and y-axis as book names.
    plt.xlabel('Model Names', fontsize=14)
    plt.ylabel('Book Names', fontsize=14)
    plt.title('Aggregated Accuracy Heatmap (Books Sorted by Occurrence)', fontsize=16)
    
    # Improve readability: rotate x tick labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save and show the heatmap
    save_file = f"{save_path}/book_model_accuracy_heatmap.png"
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Heatmap saved to {save_file}")

# -------------------------------
# EXAMPLE USAGE: Adjust these paths as needed.
# -------------------------------
occurrence_csv = '/path/to/Percentage_of_Distinct_Passages_per_Filename.csv'
accuracy_csv   = '/path/to/aggregated_en_accuracies.csv'
save_path      = '/desired/path/to/save'

create_book_model_heatmap(occurrence_csv, accuracy_csv, save_path)
