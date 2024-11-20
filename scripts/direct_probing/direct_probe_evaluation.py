import pandas as pd
import unidecode
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
import os


def run_exact_match(correct_author, correct_title_list, returned_author, returned_title,lang):
    returned_author = str(returned_author) if pd.notna(returned_author) else ''
    correct_author = str(correct_author) if pd.notna(correct_author) else ''

    # Check if the returned title matches any of the titles in the correct_title_list
    title_match = any(unidecode.unidecode(str(returned_title)).lower() == unidecode.unidecode(str(title)).lower() for title in correct_title_list)
    # Check if the authors match
    author_match = True if unidecode.unidecode(correct_author).lower() == unidecode.unidecode(returned_author).lower() else False
    # Check if both title and author match
    both_match = True if title_match == True and author_match == True else False
    result = {
        f"{lang}_title_match": title_match,
        f"{lang}_author_match": author_match,
        f"{lang}_both_match": both_match
    }
    return result


def extract_title_author(results_column):
    # Extract title and author from the results column using regex
    results_column = results_column.fillna('').astype(str).str.strip()
    return results_column.str.extract(r'"title":\s*"(.*?)",\s*"author":\s*"(.*?)"')


def evaluate(csv_file_name, book_title, book_author):
    # Load book names and CSV data
    book_names = pd.read_csv('book_names.csv')
    df = pd.read_csv(csv_file_name)
    
    # Filter columns containing 'results'
    filtered_df = df.loc[:, df.columns.str.contains('results', case=False)]

    # Find the matching row for the given book title
    matching_row = book_names[book_names.isin([book_title]).any(axis=1)].values.flatten().tolist()
    print(f"Matching row titles: {matching_row}")
    results_all = []  
    # Iterate through each filtered column
    for column in filtered_df.columns:
        print(f"Running: {column}")

        lang_results = []
        # Extract titles and authors from the column data
        filtered_column = extract_title_author(filtered_df[column])

        # Iterate through the extracted title and author pairs
        for i in range(len(filtered_column)):
            returned_title = filtered_column[0].iloc[i]  # Title is in index 0
            returned_author = filtered_column[1].iloc[i]  # Author is in index 1

            # Run exact match evaluation
            eval_result = run_exact_match(book_author, matching_row, returned_author, returned_title,column.split('_')[0])
            # print(f"Evaluation result for passage {i}: {eval_result}")

            # Store the result
            lang_results.append(eval_result)
        
        lang_results_df = pd.DataFrame(lang_results)
        results_all.append(lang_results_df)
        # Add any further logic for processing `results` if needed
    final_results_df = pd.concat(results_all, axis=1)

    # results_df = pd.DataFrame(results)
    final_results_df.to_csv('testing.csv', index=False, encoding='utf-8')
    return final_results_df

def plotting(results_df):
    total_observations = len(results_df)
    author_accuracy = {
        'en': df['en_author_correct'].value_counts(normalize=True)['correct'] * 100,
        'es': df['es_author_correct'].value_counts(normalize=True)['correct'] * 100,
        'tr': df['tr_author_correct'].value_counts(normalize=True)['correct'] * 100,
        'vi': df['vi_author_correct'].value_counts(normalize=True)['correct'] * 100
    }
    languages = list(author_accuracy.keys())
    accuracy_values = list(author_accuracy.values())
    plt.figure(figsize=(10, 6))
    bars = plt.bar(languages, accuracy_values, color=['blue', 'green', 'red', 'purple'])

    plt.bar(languages, accuracy_values, color=['blue', 'orange', 'purple', 'pink'])
    plt.rcParams.update({'font.size': 14}) 

    plt.xlabel('Language', fontsize=16)
    plt.ylabel('Accuracy (%)', fontsize=16)
    plt.title('Author Prediction Accuracy by Language  - GPT4o', fontsize=16)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, 5, f'{height:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')  # Bold formatting added here

    plt.ylim(0, 100)
    plt.show()


def get_folder_names(directory):
    # Get all folder names from a given directory
    folder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        folder_names.append(item)
        
    return folder_names


def read_txt_file(file_path):
    # Read text file content
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content.strip()


if __name__ == "__main__":
    title = '1984'
    print(f'----------------- Running {title} -----------------')
    a =evaluate(csv_file_name="1984.csv", book_title=title, book_author='George Orwell')
