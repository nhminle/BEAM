import pandas as pd
import ast
import os

def normalize_single_ent(single_ent):
    """Normalize a single entity to its canonical form."""
    try:
        name_list = ast.literal_eval(single_ent)
        if isinstance(name_list, list):
            return sorted(name_list)[0].lower().strip()
    except Exception as e:
        print(f"Error processing entry: {single_ent} -> {e}")
    return single_ent

def stratified_sample(csv_file, masked_csv_file, output_file, masked_output_file, column_name='Single_ent', sample_size=100):
    """Perform stratified sampling on both original and masked CSV files."""
    try:
        # Read both CSV files
        df = pd.read_csv(csv_file)
        masked_df = pd.read_csv(masked_csv_file)
        
        if len(df) != len(masked_df):
            raise ValueError("The original and masked CSV files must have the same number of rows.")
        
        # Normalize the 'Single_ent' column
        df['canonical_single_ent'] = df[column_name].apply(normalize_single_ent)
        
        if len(df) < sample_size:
            print(f"CSV file has only {len(df)} rows, less than the required {sample_size}. Saving the entire file.")
            
            # Drop the 'canonical_single_ent' column before saving
            df = df.drop(columns=['canonical_single_ent'], errors='ignore')
            masked_df = masked_df.drop(columns=['canonical_single_ent'], errors='ignore')
            
            # Save the entire files
            df.to_csv(output_file, index=False)
            masked_df.to_csv(masked_output_file, index=False)
            return
        
        # Step 1: Calculate group sizes proportionally for exactly 100 rows
        group_counts = df['canonical_single_ent'].value_counts()
        group_proportions = group_counts / group_counts.sum()  # Proportions of each group
        group_sizes = (group_proportions * 100).round().astype(int)  # Scale to 100 rows

        # Step 2: Perform stratified sampling while preserving order
        sampled_rows = []
        for group, group_df in df.groupby('canonical_single_ent', sort=False):
            num_rows_to_sample = group_sizes.get(group, 0)
            sampled_rows.append(group_df.iloc[:num_rows_to_sample])

        # Step 3: Concatenate sampled rows
        stratified_sample = pd.concat(sampled_rows)

        # Step 4: If the sampled DataFrame has more than 100 rows, truncate to 100
        # If less than 100 rows, add rows back proportionally to reach 100
        if len(stratified_sample) > 100:
            stratified_sample = stratified_sample.iloc[:100]
        elif len(stratified_sample) < 100:
            remaining_rows_needed = 100 - len(stratified_sample)
            additional_rows = df.loc[~df.index.isin(stratified_sample.index)].head(remaining_rows_needed)
            stratified_sample = pd.concat([stratified_sample, additional_rows])

        # Step 5: Ensure the final output has exactly 100 rows in original order
        stratified_sample = stratified_sample.sort_index()
        sampled_indices = stratified_sample.index  # Save the sampled indices
        
        # Save the sampled rows from both files
        stratified_sample.drop(columns=['canonical_single_ent'], inplace=True)
        stratified_sample.to_csv(output_file, index=False)
        
        # Use the same indices to sample from the masked file
        masked_sample = masked_df.loc[sampled_indices]
        masked_sample.to_csv(masked_output_file, index=False)
        
        print(f"Stratified sample of {sample_size} rows saved to {output_file} and {masked_output_file}.")
    
    except Exception as e:
        print(f"Error: {e}")

def get_folder_names(directory):
    """Get a list of folder names in the given directory."""
    return [item for item in os.listdir(directory) if os.path.isdir(os.path.join(directory, item))]

if __name__ == "__main__":
    base_path = '/home/nhatminhle_umass_edu/Prompts'
    
    for title in get_folder_names(base_path):
        input_csv = os.path.join(base_path, title, f"{title}_filtered.csv")
        masked_csv = os.path.join(base_path, title, f"{title}_filtered_masked.csv")
        output_csv = os.path.join(base_path, title, f"{title}_filtered_sampled.csv")
        masked_output_csv = os.path.join(base_path, title, f"{title}_filtered_masked_sampled.csv")
        
        if not os.path.isfile(input_csv) or not os.path.isfile(masked_csv):
            print(f"File not found: {input_csv} or {masked_csv}")
            continue
        
        stratified_sample(input_csv, masked_csv, output_csv, masked_output_csv, column_name='Single_ent', sample_size=100)
