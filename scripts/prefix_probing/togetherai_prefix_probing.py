import pandas as pd
from bs4 import BeautifulSoup
from openai import OpenAI
import os
from together import Together

client = Together()

def model_call(prompt, temp=0.0): 
    """
   Function to test models
    """
    ## Fireworks
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", #change to correct model!!!!!!!!
        temperature=temp,
        # top_p=1.0,
        max_tokens=800,
        messages=[{
        "role": "user",
        "content": prompt,
        }],
    )

    response = response.choices[0].message.content

    print(response, flush=True)

    return response


def extract_output(text):
    soup = BeautifulSoup(text, 'html.parser')
    passages = soup.find_all('continuation')

    if passages:
        return max([passage.get_text() for passage in passages])
    else:
        return None


def predict(passage, bookname, lang, word_count):

    few_shot_examples = {
        "es": {
            "example_passage": "Ciertamente, si había algo de lo que estaban completamente seguros los animales, era de no",
            "example_output": "Ciertamente, si había algo de lo que estaban completamente seguros los animales, era de no querer la vuelta de Jones. Cuando se les presentaba de esta forma, no sabían qué decir."
        },
        "tr": {
            "example_passage": "Hayvanların en küçük bir kuşku duymadıkları tek bir",
            "example_output": "Hayvanların en küçük bir kuşku duymadıkları tek bir şey varsa, o da Jones'un geri dönmesini istemedikleriydi."
        },
        "vi": {
            "example_passage": "Rõ ràng đây là điều rất bình thường, chẳng có ai ăn bánh mà lại bị to ra hay bé đi cả, nhưng Alice đang quá trông chờ",
            "example_output": "Rõ ràng đây là điều rất bình thường, chẳng có ai ăn bánh mà lại bị to ra hay bé đi cả, nhưng Alice đang quá trông chờ vào những điều bất thường nên khi cái điều bình thường đó xảy ra đã khiến cô cảm thấy cuộc đời sao mà ảm đạm và đáng chán"
        },
        "en": {
            "example_passage": "As Gregor Samsa awoke one morning from uneasy dreams",
            "example_output": "As Gregor Samsa awoke one morning from uneasy dreams he found himself transformed in his bed into a gigantic insect."
        }
    }

    example = few_shot_examples.get(lang)


    text_template = """
        You are provided with an incomplete passage from {book_name}. Your task is to complete the passage in the SAME LANGUAGE with {word_count} words. Return the entire completed passage, including the original input passage.
        Here is an example:
        <passage>{example_passage}</passage>
        <passage>{example_output}</passage>

        Here is the passage:
        <passage>{passage}</passage>

        You must format your output exactly as follows:
       <passage>Completed passage here</passage>
    """
    messages=text_template.format(book_name=bookname,word_count=word_count,example_passage=example["example_passage"],example_output=example["example_output"],passage=passage)
    completion = model_call(messages)
    extract = extract_output(completion)
    if extract:
        return extract
    else:
        print(completion)
    return completion


def longest_common_subsequence(str1, str2):
    dp = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]

    # Build the dp array
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    lcs = []
    i, j = len(str1), len(str2)
    while i > 0 and j > 0:
        if str1[i-1] == str2[j-1]:
            lcs.append(str1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] >= dp[i][j-1]:
            i -= 1
        else:
            j -= 1

    res = ''.join(reversed(lcs))
    if res:
        return res
    else:
        return ''


def split_sentence_in_half(sentence):
    words = sentence.split()  
    midpoint = len(words) // 2  

    first_half = ' '.join(words[:midpoint])
    second_half = ' '.join(words[midpoint:])

    return first_half, second_half, len(words)



def trim_starting_similarity(string1, string2):
    # Find the length of the common prefix
    min_len = min(len(string1), len(string2))
    i = 0
    while i < min_len and string1[i] == string2[i]:
        i += 1

    # Return the remaining part of the second string
    return string2[i:]


def slice_full_words(text, limit):
    # If the text length is within the limit, return the whole string
    if len(text) <= limit:
        return text
    # Check if the last character in the slice is a space
    elif text[limit] == ' ':
        return text[:limit]
    else:
        # Find the next space after the limit
        next_space = text.find(' ', limit)
        if next_space == -1:  # No space found, return the whole string
            return text
        else:
            return text[:next_space]


def prefixProbe(csv_file_name, book_title):
    try:
        df = pd.read_csv(csv_file_name)

        languages = ["en", "vi", "es", "tr"]
        #for lang in languages:
        for lang in df.columns:
            if lang not in ['Single_ent','Unnamed: 0']:
                print(f'///running {lang}///')
                output = []
                for i in range(len(df)):
                    full_passage = df[lang].iloc[i]
                    first_half, second_half, word_count = split_sentence_in_half(full_passage)
                    try:
                        print(f"Running prompt for {lang}: {first_half}")
                        completion = predict(first_half, book_title, lang.split('_')[0], word_count)
                        #trimmed_completion = slice_full_words(trim_starting_similarity(first_half, completion), len(second_half))
                        output.append([first_half, second_half,completion])
                    except Exception as e:
                        output.append([first_half, str(e), False])
                        print(e) 
                output_df = pd.DataFrame(output, columns=[f'{lang}_first_half', f'{lang}_second_half', f'{lang}_Completion'])
                df = pd.concat([df, output_df], axis=1)

        df.to_csv(f"/home/ekorukluoglu/beam3/BEAM/scripts/prefix_probing/together_out/{book_title}_prefix_probe_llama405b.csv", index=False, encoding='utf-8')
    except Exception as e:
        print(e)


def get_folder_names(directory):
    folder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names

def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content.strip()

if __name__ == "__main__":
    titles = get_folder_names('/home/ekorukluoglu/beam3/BEAM/scripts/Prompts')
    skip_list = ['raw','Adventures_of_Huckleberry_Finn']
    for title in titles:
        if title in skip_list:
            print(f'----------------- running {title} -----------------')
            prefixProbe(csv_file_name=f"/home/ekorukluoglu/beam3/BEAM/scripts/Prompts/{title}/{title}_filtered.csv", book_title=title.replace('_', ' '))
