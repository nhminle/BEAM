from bs4 import BeautifulSoup
from openai import OpenAI
import pandas as pd
import os
import re
from unidecode import unidecode
from fireworks.client import Fireworks

def extract_output(html):
    soup = BeautifulSoup(html, 'html.parser')
    translation_tag = soup.find('output')
    if translation_tag:
        return translation_tag.decode_contents()                        
    
    return None


def model_call(prompt, temp=0.0):
    response = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3p1-405b-instruct", #change to correct model!!!!!!!!
        temperature=temp,                                    # top_p=1.0,
        max_tokens=800,
        messages = prompt
    )
    response = response.choices[0].message.content
#    print(response, flush=True)
    return response


def predict(lang, passage):
    few_shot_examples = {
        "es": {
            "example_passage": "Lord Henry alzó las cejas y lo miró con asombro a través de las delgadas volutas de humo que, al salir de su cigarrillo con mezcla de opio, se retorcían adoptando extrañas formas.",
            "example_output": '"title": "The Picture of Dorian Gray","author": "Oscar Wilde"'
        },
        "tr": {
            "example_passage": "Kendimden pek çok şey kattım buna.” Lord Henry divana boylu boyunca uzanarak güldü.",
            "example_output": '"title": "The Picture of Dorian Gray","author": "Oscar Wilde"'
        },
        "vi": {
            "example_passage": "Oliver không hề thích thú kiểu đùa giỡn này và chuẩn bị thổ lộ sự bất bình của mình với hai anh bạn",
            "example_output": '"title": "Oliver Twist","author": "Charles Dickens"'
        },
        "en": {
            "example_passage": "I am afraid I must be going, Basil.",
            "example_output": '"title": "The Picture of Dorian Gray", "author": "Oscar Wilde"'
        }
    }

    example = few_shot_examples.get(lang)

    text_template = """
        You are provided with a passage in {lang}. Your task is to carefully read and determine which book this passage originates from and who the author is. You must make a guess, even if you are uncertain.

        Here is an example:
        <passage>{example_passage}</passage>
        <output>{example_output}</output>

        Here is the passage:
        <passage>{passage}</passage>

        Use the following format as output:
       <output>"title": "Book name","author": "author name"</output>
    """
    messages=[{"role": "user", "content": text_template.format(lang=lang,example_passage=example["example_passage"],example_output=example["example_output"],passage=passage)}]
    completion = model_call(messages)
    extract = extract_output(completion)
    if extract:
        return extract
    else:
        print(completion)
    return completion


def direct_probe(csv_file_name, book_title):
    try:
        df = pd.read_csv(csv_file_name)

        for language in df.columns:
            if language not in ['Single_ent','Unnamed: 0']:
                print(f'Running {language}')
                output = []
                passage_no = 0
                for i in range(len(df)):
                    passage = df[language].iloc[i]
                    content = predict(language.split('_')[0],passage)
                    output.append(content)
                    print(f'{passage_no}: {content}')
                    passage_no += 1
                index_of_language = df.columns.get_loc(language)
                output_col = pd.Series(output)
                df.insert(index_of_language + 1, f"{language}_results", output_col)

        df.to_csv(f"/home/ekorukluoglu/beam3/BEAM/scripts/direct_probing/fireworks_out/{book_title}_direct_probe_llama405b.csv", index=False, encoding='utf-8')
    except Exception as e:
       print(f'Error: {e}')

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
    
    for title in titles:
        if title in [  "Adventures_of_Huckleberry_Finn",  "Bride",  "Dune",  "Harry_Potter_and_the_Deathly_Hallows",  "Sense_and_sensibility",  "The_Picture_of_Dorian_Gray",  "You_Like_It_Darker_Stories","Adventures_of_Sherlock_Holmes"]:
            print(f'----------------- running {title} -----------------')
            direct_probe(csv_file_name=f"/home/ekorukluoglu/beam3/BEAM/scripts/Prompts/{title}/{title}_filtered.csv", book_title=title)
                
