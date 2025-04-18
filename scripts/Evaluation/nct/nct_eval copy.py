import os
import re
import pandas as pd
import unidecode
from fuzzywuzzy import fuzz
import ast

def is_2024_book(book_title):
    """
    Check if the title is one of the 2024 set.
    """
    print(book_title)
    return book_title in [
        'Below_Zero', 'Bride', 'First_Lie_Wins', 'Funny_Story',
        'If_Only_I_Had_Told_Her', 'Just_for_the_Summer', 'Lies_and_Weddings',
        'The_Ministry_of_Time', 'The_Paradise_Problem', 'You_Like_It_Darker_Stories'
    ]

def extract_output_names(text):
    """
    Extracts text between <output> and </output>.
    If no matches are found, returns the original text.
    If matches are found, joins them with a space.
    """
    matches = re.findall(r'<output>(.*?)</output>', text)
    return " ".join(matches) if matches else text

def preprocess_result(val):
    """
    Preprocess a result string.
    First applies extract_output_names, then unidecodes,
    strips extra whitespace, and converts to lowercase.
    """
    if isinstance(val, str):
        # Apply extract_output_names
        processed = extract_output_names(val)
        # Then normalize the string
        return unidecode.unidecode(processed).strip().lower()
    return val

def main(model_folder, prompt_setting, filename, book_title):
    # Construct input file path
    input_filepath = os.path.join(model_folder, prompt_setting, filename + '.csv')
    df1 = pd.read_csv(input_filepath)
    
    # Load ground truth file based on book_title
    if is_2024_book(book_title):
        print("ADASDSADASDASD")
        # /home/ekorukluoglu_umass_edu/beam2/BEAM/scripts/Prompts/2024/Below_Zero/Below_Zero_unmasked_passages.csv
        df2 = pd.read_csv(f'/home/ekorukluoglu_umass_edu/beam2/BEAM/scripts/Prompts/2024/{book_title}/{book_title}_filtered_masked.csv')
    else:
        df2 = pd.read_csv(f'/home/ekorukluoglu_umass_edu/beam2/BEAM/scripts/Prompts/2024/{book_title}/{book_title}_filtered_masked.csv')
    
    # Merge ground truth into df1 using the common 'en' column.
    df1['Single_ent'] = df2['Single_ent']
    
    # Check that all ground truth values are present and in list form.
    def ensure_list(val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        diagnostic_path = os.path.join(os.getcwd(), "diagnostic_df1.csv")
        df1.to_csv(diagnostic_path, index=False)
        raise ValueError(f"Ground truth must be present and in list form. Got: {val}. "
                         f"Saved diagnostic file to {diagnostic_path}")
    
    df1['Single_ent'] = df1['Single_ent'].apply(ensure_list)
    
    # Preprocess the result columns (both unshuffled and shuffled) using preprocess_result.
    for col in df1.columns:
        if col.endswith('_results') or col.endswith('_shuffled_results'):
            df1[col] = df1[col].apply(preprocess_result)
    
    # Expand entity tokens in the ground truth.
    def granular_ents(val):
        expanded_list = []
        for item in val:
            if isinstance(item, str):
                expanded_list.append(item)
                expanded_list.extend(item.split())
        return list(set(expanded_list))
    
    df1['Single_ent'] = df1['Single_ent'].apply(granular_ents)
    
    # Extract available languages from the columns ending in '_results'
    available_langs = [col.split('_results')[0] for col in df1.columns if col.endswith('_results')]
    
    # Define a helper function to calculate evaluation scores.
    def calculate_match_scores(ents_list, en_result):
        en_result_normalized = unidecode.unidecode(str(en_result)).lower().strip()
        exact_match = 0
        highest_fuzzy_score = 0
        for ent in ents_list:
            ent_normalized = unidecode.unidecode(ent).lower().strip()
            # Use strict equality for exact match.
            if ent_normalized == en_result_normalized:
                exact_match = 1
            fuzzy_score = fuzz.ratio(ent_normalized, en_result_normalized) / 100
            highest_fuzzy_score = max(highest_fuzzy_score, fuzzy_score)
        return exact_match, highest_fuzzy_score
    
    # For each available language, compute new evaluation columns.
    for lang in available_langs:
        # Process unshuffled results.
        col_unshuffled = f"{lang}_results"
        if col_unshuffled in df1.columns:
            df1[f'{lang}_exact_match'], df1[f'{lang}_highest_fuzzy_match'] = zip(*df1.apply(
                lambda row: calculate_match_scores(row['Single_ent'], row[col_unshuffled]), axis=1))
            df1[f'{lang}_correct'] = df1.apply(
                lambda row, lang=lang: 'correct' if (
                    row[f'{lang}_exact_match'] == 1
                    or (row[f'{lang}_exact_match'] == 0 and row[f'{lang}_highest_fuzzy_match'] >= 0.7)
                ) else 'incorrect',
                axis=1
            )
        
        # Process shuffled results.
        col_shuffled = f"{lang}_shuffled_results"
        if col_shuffled in df1.columns:
            df1[f'{lang}_shuffled_exact_match'], df1[f'{lang}_shuffled_highest_fuzzy_match'] = zip(*df1.apply(
                lambda row: calculate_match_scores(row['Single_ent'], row[col_shuffled]), axis=1))
            df1[f'{lang}_shuffled_correct'] = df1.apply(
                lambda row, lang=lang: 'correct' if (
                    row[f'{lang}_shuffled_exact_match'] == 1
                    or (row[f'{lang}_shuffled_exact_match'] == 0 and row[f'{lang}_shuffled_highest_fuzzy_match'] >= 0.7)
                ) else 'incorrect',
                axis=1
            )
    
    # Define the new columns to include in the output CSV.
    # This includes the ground truth, original result columns, and evaluation columns.
    new_cols = ['Single_ent']
    for lang in available_langs:
        # Unshuffled columns.
        unshuffled_cols = [f"{lang}_results", f'{lang}_exact_match', f'{lang}_highest_fuzzy_match', f'{lang}_correct']
        # Shuffled columns.
        shuffled_cols = [f"{lang}_shuffled_results", f'{lang}_shuffled_exact_match', f'{lang}_shuffled_highest_fuzzy_match', f'{lang}_shuffled_correct']
        for col in unshuffled_cols:
            if col in df1.columns:
                new_cols.append(col)
        for col in shuffled_cols:
            if col in df1.columns:
                new_cols.append(col)
    
    # Decide output folder. If it's a 2024 book, go to /evaluation/2024, else /evaluation.
    eval_dir_base = os.path.join(model_folder, prompt_setting, "evaluation")
    os.makedirs(eval_dir_base, exist_ok=True)
    
    if is_2024_book(book_title):
        eval_dir_2024 = os.path.join(eval_dir_base, "2024")
        os.makedirs(eval_dir_2024, exist_ok=True)
        output_csv = os.path.join(eval_dir_2024, f'{filename}.csv')
    else:
        output_csv = os.path.join(eval_dir_base, f'{filename}.csv')
    
    # Save the output CSV with only the specified columns.
    df1[new_cols].to_csv(output_csv, index=False, encoding='utf-8')
    print(f"Saved evaluation for {filename} under {output_csv}")

def list_csv_files(directory):
    try:
        files = os.listdir(directory)
        return [file.replace('.csv', '') for file in files if file.endswith('.csv')]
    except FileNotFoundError:
        print(f"Error: The directory '{directory}' does not exist.")
        return []
    except PermissionError:
        print(f"Error: Permission denied for accessing '{directory}'.")
        return []
    
def parse_filename(base_name):
    experiments = ["name_cloze"]
    experiment = None
    experiment_index = None
    for exp in experiments:
        index = base_name.find(exp)
        if index != -1:
            experiment = exp
            experiment_index = index
            break
    if experiment is None:
        raise ValueError(f"Experiment not found in filename: {base_name}")
    title_part = base_name[:experiment_index]
    if title_part.endswith('_'):
        title_part = title_part[:-1]
    title = title_part.strip()
    remainder = base_name[experiment_index + len(experiment):]
    if remainder.startswith('_'):
        remainder = remainder[1:]
    prompt_setting = None
    for ps in ["one_shot", "zero_shot"]:
        if ps in remainder:
            prompt_setting = ps
            break
    if prompt_setting is None:
        raise ValueError(f"Prompt setting ('one-shot' or 'zero-shot') not found in filename: {base_name}")
    parts = remainder.rsplit('_', 1)
    if len(parts) != 2:
        raise ValueError(f"Unable to parse model and prompt setting from remainder: {remainder}")
    return title

def run_evaluation(model_folder):
    prompt_settings = ['one-shot', 'zero-shot']
    for prompt in prompt_settings:
        folder_path = os.path.join(model_folder, prompt)
        titles = list_csv_files(folder_path)
        for t in titles:
            print(f'---------------- Running {t} for prompt setting: {prompt} ----------------')
            main(model_folder, prompt, t, parse_filename(t))

# Example usage:
dirs_nct = [
    '/home/ekorukluoglu_umass_edu/beam2/BEAM/results/name_cloze/Llama-3.1-405b/2024'
]
for d in dirs_nct:
    run_evaluation(d)
# /home/ekorukluoglu_umass_edu/beam2/BEAM/results/name_cloze/Llama-3.1-405b/2024/one-shot
# /home/ekorukluoglu_umass_edu/beam2/BEAM/results/name_cloze/Llama-3.1-405b/2024/one-shot