import os
import pandas as pd
import numpy as np

def guess_accuracy(data):
    """
    Compute per-column accuracy (percentage) from a DataFrame of evaluation results.
    Each column is assumed to be a binary indicator (0/1) per test instance.
    """
    results = {}
    # If the DataFrame is empty or has zero rows, return an empty dict.
    if len(data) == 0:
        return results
    for column in data.columns:
        column_acc = float(data[column].sum() / len(data) * 100)
        results[column] = column_acc
    return results

def average_accuracy(accuracy_dict):
    """
    Compute the average accuracy from the accuracy_dict.
    If no values exist, return NaN.
    """
    if not accuracy_dict:
        return np.nan
    return np.mean(list(accuracy_dict.values()))

def list_eval_csv_files(directory):
    """
    List files in the given directory that are evaluation CSV files.
    We assume these files end with '_eval.csv'.
    """
    try:
        files = [f for f in os.listdir(directory) if f.endswith('_eval.csv')]
        return files
    except FileNotFoundError:
        print(f"Error: Directory {directory} does not exist.")
        return []
    except PermissionError:
        print(f"Error: Permission denied for directory {directory}.")
        return []

def main():
    # List your models here.
    # Replace with the actual model names that you use.
    models = ['Model1', 'Model2', 'Model3']
    # Set your base directory where the results are saved.
    base_dir = 'results/direct_probe'
    # If you are using a specific prompt setting folder (e.g., "ne_one_shot" or "zero_shot")
    prompt_setting = 'ne_one_shot'
    
    # This dictionary will hold the combined accuracies.
    # Structure: {book_name: {model: avg_accuracy}}
    combined_accuracies = {}
    
    # Iterate over each model
    for model in models:
        # Construct the evaluation directory path for each model.
        eval_dir = os.path.join(base_dir, model, prompt_setting, 'evaluation')
        csv_files = list_eval_csv_files(eval_dir)
        print(f"Processing {len(csv_files)} CSV files for model {model} in directory {eval_dir}.")
        
        for csv_file in csv_files:
            # Assume CSV files are named like: BookName_eval.csv
            # Remove the '_eval.csv' suffix to get the book name.
            book_name = csv_file.replace('_eval.csv', '')
            file_path = os.path.join(eval_dir, csv_file)
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"Could not read {file_path}: {e}")
                continue

            # Compute accuracy values per column and take the average as the overall accuracy for the book.
            acc_dict = guess_accuracy(df)
            avg_acc = average_accuracy(acc_dict)
            
            # Add the result to the combined accuracies dictionary.
            if book_name not in combined_accuracies:
                combined_accuracies[book_name] = {}
            combined_accuracies[book_name][model] = avg_acc

    # Convert the dictionary to a DataFrame.
    # This DataFrame will have one row per book and columns for each model.
    summary_df = pd.DataFrame.from_dict(combined_accuracies, orient='index')
    summary_df.index.name = 'Book Name'
    summary_df.reset_index(inplace=True)
    
    # Save the summary CSV file.
    output_csv = 'combined_accuracies.csv'
    summary_df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Combined accuracies saved to {output_csv}")

if __name__ == '__main__':
    main()
