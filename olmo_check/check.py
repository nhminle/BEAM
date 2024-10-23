import requests
import pandas
import os

def get_folder_names(directory):
    folder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names

def split_string_into_substrings(text, parts=4):
    # Split the text into words
    words = text.split()
    
    # Calculate the approximate number of words per part
    total_words = len(words)
    words_per_part = total_words // parts
    remainder = total_words % parts
    
    substrings = []
    start = 0
    
    # Split the words into the required number of parts
    for i in range(parts):
        # Distribute the remainder across parts
        end = start + words_per_part + (1 if i < remainder else 0)
        substrings.append(' '.join(words[start:end]))
        start = end
    
    return substrings

def olmo_check(df,title):
    results = []
    i = 0
    for entry in df['en']:
        char_length = len(entry)
        subs = split_string_into_substrings(entry)
        i = i+1
        for item in subs:
            # print("Payloading item :" + item)
            payload = {
                'index': 'v4_rpj_llama_s4',
                'query_type': 'infgram_prob',
                'query': item,
            }
            result = requests.post('https://api.infini-gram.io/', json=payload).json()
            results.append(result)
    print(str(i))
    result_df = pandas.DataFrame(results)
    result_df.to_csv(f'BEAM/olmo_check/{title}_olmochecked.csv',index=False)



if __name__ == "__main__":
    titles = get_folder_names('scripts/Prompts')
    skip_list = ['Adventures_of_Huckleberry_Finn']
    for title in titles:
        if title in skip_list:
            print(f'----------------- running {title} -----------------')
            df=pandas.read_csv(f"scripts/Prompts/{title}/{title}_filtered_masked.csv")
            olmo_check(df,title)
      