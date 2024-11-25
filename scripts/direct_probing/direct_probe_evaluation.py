import pandas as pd
import unidecode
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
import os
import numpy as np

def run_exact_match(correct_author, correct_title_list, returned_author, returned_title, lang):
    returned_author = str(returned_author) if pd.notna(returned_author) else ''
    correct_author = str(correct_author) if pd.notna(correct_author) else ''
    
    # print("Returned Author:", returned_author)
    # print("Correct Author:", correct_author)

    # Check if the returned title matches any of the titles in the correct_title_list using fuzzy matching
    title_match = any(
        fuzz.ratio(unidecode.unidecode(str(returned_title)).lower(), unidecode.unidecode(str(title)).lower()) >= 70
        for title in correct_title_list
    )

    # Check if the authors match using fuzzy matching
    author_match = fuzz.ratio(unidecode.unidecode(correct_author).lower(), unidecode.unidecode(returned_author).lower()) >= 70

    # Check if both title and author match
    both_match = title_match and author_match

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


def evaluate(csv_file_name, book_title):
    # Load book names and CSV data
    print(book_title)
    book_names = pd.read_csv('book_names.csv')
    df = pd.read_csv(csv_file_name)
    available_langs = [col.split('_')[0] for col in df.columns if col.endswith('_masked_results')]
    # Filter columns containing 'results'
    filtered_df = df.loc[:, df.columns.str.contains('results', case=False)]
    book_title = book_title.replace('_direct_probe_gpt4o', '')
    book_title = book_title.replace('_',' ')
    print(book_title)
    # Find the matching row for the given book title
    matching_row = book_names[book_names.isin([book_title]).any(axis=1)].values.flatten().tolist()
    author = matching_row[0]
    print(author)
    print(f"Matching row titles: {matching_row}")
    results_all = pd.DataFrame()
    # Iterate through each filtered column
    for column in filtered_df.columns:
        # print(f"Running: {column}")

        lang_results = []
        # Extract titles and authors from the column data
        filtered_column = extract_title_author(filtered_df[column])

        # Iterate through the extracted title and author pairs
        for i in range(len(filtered_column)):
            returned_title = filtered_column[0].iloc[i]  # Title is in index 0
            returned_author = filtered_column[1].iloc[i]  # Author is in index 1

            # Run exact match evaluation
            eval_result = run_exact_match(author, matching_row, returned_author, returned_title,column)
            #print(f"Evaluation result for passage {i}: {eval_result}")

            # Store the result
            lang_results.append(eval_result)
        #add results to the results_all   
        lang_results_df = pd.DataFrame(lang_results)
        results_all =pd.concat([results_all,lang_results_df],axis =1)
    
    results_all.to_csv('testing.csv', index=False, encoding='utf-8')
    return results_all

def split_data(data): #this function splits our results to shuffled and unshuffled
    shuffled = pd.DataFrame()
    unshuffled = pd.DataFrame()

    for column in data.columns:
        if 'shuffled' in column:
            shuffled = pd.concat([shuffled,data[column]],axis =1)
        else:
            unshuffled = pd.concat([unshuffled,data[column]],axis =1)

    return shuffled, unshuffled

def save_data(title,data,shuffled):
    if shuffled:
        data.to_csv(f'./Evaluation/shuffled/{title}_eval.csv', index= False, encoding='utf-8')
    else:
        data.to_csv(f'./Evaluation/unshuffled/{title}_eval.csv', index= False, encoding='utf-8')  


def guess_accuracy(data):
    results = {}
    for column in data.columns:
        column_acc = float(data[column].sum()/len(data)*100)
       
       #print("Accuracy for " +column + " is "+str(column_acc))
        results[f'{str(column)}']= column_acc

    return results

    # lang: df[f'{lang}_correct'].value_counts(normalize=True).get('correct', 0) * 100
    # for lang in available_langs if f'{lang}_correct' in df
    
def get_folder_names(directory):
    # Get all folder names from a given directory
    folder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        folder_names.append(item)
        
    return folder_names

def plot(accuracy_data, title, shuffled):
    languages = ['en', 'es', 'tr', 'vi']
    categories = ['title_match', 'author_match', 'both_match']
    data = {}

    for lang in languages:
        data[lang] = []
        for cat in categories:
            try:
                if shuffled:
                    # Try to access the shuffled key
                    value = accuracy_data[f"{lang}_prompts_shuffled_results_{cat}"]
                else:
                    # Try to access the unshuffled key
                    value = accuracy_data[f"{lang}_results_{cat}"]
                data[lang].append(value)  # Append the value if it exists
            except KeyError:
                # Skip if the key is missing
                print(f"Missing data for {lang} - {cat}, skipping...")
                data[lang].append(None)

    # Plotting
    x = np.arange(len(categories))  # Category indices
    bar_width = 0.2
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, lang in enumerate(languages):
        # Filter out None values and their corresponding categories
        filtered_data = [val for val in data[lang] if val is not None]
        filtered_categories = [cat for val, cat in zip(data[lang], categories) if val is not None]

        if filtered_data:  # Only plot if there is valid data
            ax.bar(
                x[:len(filtered_data)] + i * bar_width,
                filtered_data,
                bar_width,
                label=lang.capitalize()
            )

    # Adding labels, title, and legend
    ax.set_xlabel('Categories', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title(
        f'Accuracy per Language and Category, {title.replace("_direct_probe_gpt4o.csv", "").replace("_", " ")}',
        fontsize=14
    )
    ax.set_xticks(x + bar_width * 1.5)  # Centering the group labels
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(title='Language', fontsize=10)

    # Display the plot
    plt.tight_layout()
    if shuffled:
        plt.savefig(f'./Evaluation/plots/shuffled/{title}.png', dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f'./Evaluation/plots/unshuffled/{title}.png', dpi=300, bbox_inches='tight')

def read_txt_file(file_path):
    # Read text file content
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content.strip()


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

def create_heatmap(df,title,shuffled,release_date_csv):
    release_dates = pd.read_csv(release_date_csv)
    release_dates['Release Date'] = pd.to_datetime(release_dates['Release Date'])  # Ensure datetime format



if __name__ == "__main__":
    titles = list_csv_files('./Evaluation/gpt4o/')
    unshuffled_accuracy_list = {}
    shuffled_accuracy_list = {}
    
    for title in titles:
        # if title == 'A_thousand_splendid_suns_direct_probe_gpt4o':
        print(f'----------------- Running {title} -----------------')    
        results_evaluated =evaluate(csv_file_name=f'./Evaluation/gpt4o/{title}.csv', book_title=title)
        shuffled, unshuffled = split_data(results_evaluated)
        save_data(title,shuffled,True)
        save_data(title,unshuffled,False)
        unshuffled_acc_df = guess_accuracy(unshuffled)
        shuffled_acc_df = guess_accuracy(shuffled)
        print(unshuffled_acc_df.keys)
        # unshuffled_accuracy_list.append(unshuffled_acc_df)
        # shuffled_accuracy_list.apend(shuffled_acc_df)
        plot(unshuffled_acc_df,title,False) 
        plot(shuffled_acc_df,title,True)    

