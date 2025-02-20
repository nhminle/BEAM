import pandas as pd
import re
import os

def clean_text(text):
    return re.sub(r'\[MASK\]|@@PLACEHOLDER@@', '', str(text)).strip()

def compare_en_masked(csv_path1, csv_path2, book_name):
    # Load both CSV files with explicit encoding
    df1 = pd.read_csv(csv_path1, encoding="utf-8")
    df2 = pd.read_csv(csv_path2, encoding="utf-8")
    
    # Check if 'en_masked' column exists in both files
    if 'en_masked' not in df1.columns or 'en_masked' not in df2.columns:
        print(f"Error: 'en_masked' column not found in one or both CSV files for book: {book_name}")
        return
    
    # Reset index to avoid misalignment issues
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    
    # Check if both files have the same number of rows
    if len(df1) != len(df2):
        print(f"Warning: CSV files for book '{book_name}' have different number of rows. {len(df1)} vs {len(df2)}")
    
    # Normalize 'en_masked' column to avoid encoding, whitespace, and newline issues
    df1['en_masked'] = df1['en_masked'].astype(str).str.strip().str.replace("\r\n", "\n").str.lstrip("\ufeff").apply(clean_text)
    df2['en_masked'] = df2['en_masked'].astype(str).str.strip().str.replace("\r\n", "\n").str.lstrip("\ufeff").apply(clean_text)
    
    # Compare the 'en_masked' column row by row
    min_rows = min(len(df1), len(df2))
    mismatches = []
    
    for i in range(min_rows):
        if df1.loc[i, 'en_masked'] != df2.loc[i, 'en_masked']:
            mismatches.append(i)
    
    # Print results
    if mismatches:
        print(f"Mismatches found in book '{book_name}':")
        for row in mismatches:
            print(f"Row {row}: CSV1='{df1.loc[row, 'en_masked']}' | CSV2='{df2.loc[row, 'en_masked']}'")
    else:
        print(f"No mismatches found in 'en_masked' column for book '{book_name}'.")

def add_single_ent_column(csv1_path, csv2_path, book_name):
    df1 = pd.read_csv(csv1_path, encoding="utf-8")
    df2 = pd.read_csv(csv2_path, encoding="utf-8")
    
    if 'Single_ent' not in df2.columns:
        print(f"Skipping '{book_name}': 'Single_ent' column not found in stored CSV.")
        return
    
    # Ensure length match before merging
    if len(df1) != len(df2):
        print(f"Skipping '{book_name}': Row mismatch between CSVs ({len(df1)} vs {len(df2)}).")
        return
    
    # Add 'Single_ent' column
    df1['Single_ent'] = df2['Single_ent']
    
    # Overwrite the _masked_passages.csv file with the new column
    df1.to_csv(csv1_path, index=False, encoding="utf-8")
    print(f"Updated '{csv1_path}' with 'Single_ent' column.")

def process_all_books(prompts_folder):
    for book_name in os.listdir(prompts_folder):
        book_path = os.path.join(prompts_folder, book_name)
        
        if not os.path.isdir(book_path) or book_name == "2024":
            continue
        
        csv1_path = os.path.join(book_path, f"{book_name}_masked_passages.csv")
        stored_folder = os.path.join(book_path, "stored")
        csv2_path = os.path.join(stored_folder, f"{book_name}.csv")
        
        if os.path.exists(csv1_path) and os.path.exists(csv2_path):
            print(f"Processing book: {book_name}")
            compare_en_masked(csv1_path, csv2_path, book_name)
            add_single_ent_column(csv1_path, csv2_path, book_name)
        else:
            print(f"Skipping book '{book_name}' due to missing CSV files.")

if __name__ == "__main__":
    prompts_folder = "/Users/alishasrivastava/BEAM-scripts/BEAM/scripts/Prompts"  # Replace with actual path
    process_all_books(prompts_folder)
