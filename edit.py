import pandas as pd
from google.cloud import translate_v2 as translate

def translate_text(text, target_language):
    """Translate text using Google Translate."""
    client = translate.Client()
    result = client.translate(text, target_language=target_language)
    return result['translatedText']

def process_csv(file_path):
    # Load the CSV
    df = pd.read_csv(file_path)
    
    # Identify columns ending with '_match'
    match_cols = [col for col in df.columns if col.endswith('_match')]
    print("Detected _match columns:", match_cols)  # Debugging step
    
    # Iterate over each '_match' column
    for match_col in match_cols:
        lang_prefix = match_col.split('_')[0]  # Extract language prefix (e.g., 'st' from 'st_match')
        gt_col = f"{lang_prefix}_gt"  # Define the corresponding '_gt' column
        
        # Remove the column if it already exists
        if gt_col in df.columns:
            df.drop(columns=[gt_col], inplace=True)
        
        # Initialize the new `_gt` column with None
        df[gt_col] = None
        
        # Process rows where the '_match' column is FALSE
        for idx, row in df.iterrows():
            match_value = row[match_col]
            if pd.notnull(match_value) and str(match_value).strip().upper() == "FALSE":
                print(f"FALSE found at row {idx} in column {match_col}. Translation to {lang_prefix} started.")
                text_to_translate = row['en_masked']
                try:
                    translated_text = translate_text(text_to_translate, lang_prefix)
                    df.at[idx, gt_col] = translated_text
                    print(f"Translation completed for row {idx} in column {match_col}.")
                except Exception as e:
                    print(f"Translation failed for row {idx}, text: {text_to_translate}. Error: {e}")
    
    # Overwrite the original CSV
    df.to_csv(file_path, index=False)
    print(f"CSV file successfully updated: {file_path}")

file_path = "/Users/alishasrivastava/BEAM-scripts/BEAM/scripts/Prompts/1984/1984.csv"
process_csv(file_path)
