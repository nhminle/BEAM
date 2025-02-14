import pandas as pd
import unidecode
from fuzzywuzzy import fuzz
import os
import matplotlib.pyplot as plt
import ast
import seaborn as sns
import re 
import sys
def main(title,model,prompt_setting):
    print(title,model,prompt_setting)
    # _name_cloze_Llama-3.1-405b_one-shot
    replace_str = '_name_cloze_{model}_{prompt_setting}'
    print(replace_str)
    book_name = title.replace(f'_name_cloze_{model}_{prompt_setting}', '')
    print(f'Processing {book_name}...')
    df1 = pd.read_csv(f'/Users/emir/Downloads/asd/BEAM/results/name_cloze/{model}/{prompt_setting}/{title}.csv')
    ent_df = pd.read_csv(f"/Users/emir/Downloads/asd/BEAM/scripts/Prompts/{book_name}/stored/{book_name}_filtered_sampled.csv")
    # print(df1['en_masked_shuffled_results'])
    # return
    #append the "Single_ent" column from ent_df to df1
    df1['Single_ent'] = ent_df['Single_ent']
    title = title.replace('nct', 'name_cloze')
    df = pd.DataFrame()
    df_shuffled = pd.DataFrame()

    available_langs = [col.split('_')[0] for col in df1.columns if col.endswith('_masked_results')]

    for col in [f"{lang}_masked_results" for lang in available_langs]:
        df1[col] = df1[col].str.replace(r'<output>(.*?)</output>', r'\1', regex=True)
    for col in [f"{lang}_masked_shuffled_results" for lang in available_langs]:
        df1[col] = df1[col].str.replace(r'<output>(.*?)</output>', r'\1', regex=True)

    df1['Single_ent'] = df1['Single_ent'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['Single_ent'] = df1['Single_ent']
    df_shuffled['Single_ent'] = df1['Single_ent']

    for col in [f"{lang}_masked_results" for lang in available_langs]:
        df1[col] = df1[col].str.lower()
        df[col] = df1[col].str.lower()

    for col in [f"{lang}_masked_shuffled_results" for lang in available_langs]:
        df1[col] = df1[col].str.lower()
        df_shuffled[col] = df1[col].str.lower()
    print(df_shuffled)
    def granular_ents(row):
        expanded_list = []
        for item in row:
            expanded_list.append(item)
            expanded_list.extend(item.split())
        # print("row: ", row)
        # print("gran:", list(set(expanded_list)))
        return list(set(expanded_list))

    df1['Single_ent'] = df1['Single_ent'].apply(granular_ents)
    print(df1.iloc[39])

    def is_string(value):
        if isinstance(value, str):
            normalized_value = unidecode.unidecode(value)
            return normalized_value.replace(" ", "").isalpha()
        return False

    for col in [f"{lang}_masked_results" for lang in available_langs]:
        for value in df1[col]:
            if not is_string(value):
                print(type(value))
                value = str(value)
                print(f"Invalid '{col}': {value}")
    
    def extract_output_names(text):
        return re.findall(r'<output>(.*?)</output>', text)
    
    def calculate_match_scores(ents_list, en_result):
        en_result_normalized = unidecode.unidecode(str(en_result)).lower().strip()
        
        exact_match = 0
        highest_fuzzy_score = 0
        
        for ent in ents_list:
            ent_normalized = unidecode.unidecode(ent).lower().strip()
            # ent_extracted = extract_output_names(ent_normalized)
            # ent_normalized = ent_extracted[0] if ent_extracted else ent_normalized
            if ent_normalized == en_result_normalized:
                exact_match = 1
            
            fuzzy_score = fuzz.ratio(ent_normalized, en_result_normalized)/ 100
            highest_fuzzy_score = max(highest_fuzzy_score, fuzzy_score)
        
        return exact_match, highest_fuzzy_score

    for lang in available_langs:
        try:
            df[f'{lang}_exact_match'], df[f'{lang}_highest_fuzzy_match'] = zip(
                *df1.apply(lambda row: calculate_match_scores(row['Single_ent'], row[f'{lang}_masked_results']), axis=1)
            )
            df_shuffled[f'{lang}_exact_match'], df_shuffled[f'{lang}_highest_fuzzy_match'] = zip(
                *df1.apply(lambda row: calculate_match_scores(row['Single_ent'], row[f'{lang}_masked_shuffled_results']), axis=1)
            )
        except Exception as e:
            print(f"Error processing {lang}: {e}")
    # print(df_shuffled.columns)
    # print(df_shuffled)
    # return
    # print(df)

    def assess(language, df, threshold=0.7):
        fuzzy_col = f'{language}_highest_fuzzy_match'
        exact_col = f'{language}_exact_match'
        result_col = f'{language}_correct'
        
        df[result_col] = df.apply(
            lambda row: 'correct' if (
                row[exact_col] == 1 or
                (row[exact_col] == 0 and row[fuzzy_col] >= threshold)
            ) else 'incorrect',
            axis=1
        )

        return df

    for lang in available_langs:
        assess(lang, df)
        assess(lang, df_shuffled)
    # print(df)
    # models = ['OLMo-7B-0724-Instruct-hf', 'Llama-3.1-70B-Instruct', 'Meta-Llama-3.1-8B-Instruct', 'gpt4o']
    # cur_model = ''
    # for model in models:
    #     if model in title:
    #         cur_model = model
    df.to_csv(f'/Users/emir/Downloads/asd/BEAM/results/name_cloze/{model}/eval/csv/{title}_{prompt_setting}.csv', index=False, encoding='utf-8')
    df_shuffled.to_csv(f'/Users/emir/Downloads/asd/BEAM/results/name_cloze/{model}/eval/csv/{title}_{prompt_setting}_shuffled.csv', index=False, encoding='utf-8')
    # ------------------------------------------ plotting unshuffled data ------------------------------------------
    unshuffled_guess_accuracy = {
        lang: df[f'{lang}_correct'].value_counts(normalize=True).get('correct', 0) * 100
        for lang in available_langs if f'{lang}_correct' in df
    }
    # print(unshuffled_guess_accuracy)
    languages = list(unshuffled_guess_accuracy.keys())
    accuracy_values = list(unshuffled_guess_accuracy.values())
    # print(languages)
    # print(accuracy_values)
    plt.figure(figsize=(10, 6))
    colors = ['#4E79A7', '#F28E2B', '#76B7B2', '#E15759']
    bars = plt.bar(languages, accuracy_values, color=colors)
    plt.rcParams.update({'font.size': 14}) 

    plt.xlabel('Language', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=16)
    plt.title(f'{title.replace(f'_', ' ')}', fontsize=16)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, 5, f'{height:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')  # Bold formatting added here

    plt.ylim(0, 100)
    plt.savefig(f'/Users/emir/Downloads/asd/BEAM/results/name_cloze/{model}/eval/{title}.png', dpi=300, bbox_inches='tight')
    
    # ------------------------------------------ plotting shuffled data ------------------------------------------
    shuffled_guess_accuracy = {
        lang: df_shuffled[f'{lang}_correct'].value_counts(normalize=True).get('correct', 0) * 100
        for lang in available_langs if f'{lang}_correct' in df
    }
    # print(df_shuffled)
    languages = list(shuffled_guess_accuracy.keys())
    accuracy_values = list(shuffled_guess_accuracy.values())
    plt.figure(figsize=(10, 6))
    colors = ['#4E79A7', '#F28E2B', '#76B7B2', '#E15759']
    bars = plt.bar(languages, accuracy_values, color=colors)
    plt.rcParams.update({'font.size': 14}) 

    plt.xlabel('Language', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=16)
    plt.title(f'{title.replace(f'_', ' ')} SHUFFLED', fontsize=16)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, 5, f'{height:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')  # Bold formatting added here

    plt.ylim(0, 100)
    plt.savefig(f'/Users/emir/Downloads/asd/BEAM/results/name_cloze/{model}/eval/{title}_shuffled.png', dpi=300, bbox_inches='tight')
    

def list_csv_files(directory):
    try:
        files = os.listdir(directory)
        
        csv_files = [file.replace('.csv', '') for file in files if file.endswith('.csv')]
        
        return csv_files
    except FileNotFoundError:
        print(f"Error: The directory '{directory}' does not exist.")
        return []
    except PermissionError:
        print(f"Error: Permission denied for accessing '{directory}'.")
        return []

# titles = list_csv_files('/Users/minhle/Umass/ersp/Evaluation/nct/llm_out')

# # for t in titles:
# #     print(f'----------------running {t}----------------')
# #     main(t)

# main('The_Boy_in_the_Striped_Pyjamas_name_cloze_Llama-3.1-70B-Instruct')
#'EuroLLM-9B-Instruct',
models = [ 'OLMo-7B-0724-Instruct-hf', 'Llama-3.1-70B-Instruct', 'Llama-3.3-70B-Instruct', 'Meta-Llama-3.1-8B-Instruct', 'OLMo-2-1124-13B-Instruct','Llama-3.1-405b','gpt-4o-2024-11-20'] # add more models here
#models = ['Llama-3.1-405b']
prompt_setting = 'zero-shot' # one-shot || zero-shot

for model in models:
    titles = list_csv_files(f'/Users/emir/Downloads/asd/BEAM/results/name_cloze/{model}/{prompt_setting}/')
    for title in titles:
        if 'Below_Zero' not in title and 'The_Ministry_of_Time' not in title and 'Funny_Story' not in title:
            print(f'----------------running {title}----------------')
            main(title, model, prompt_setting)