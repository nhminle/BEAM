from bs4 import BeautifulSoup
from openai import OpenAI
import pandas as pd
import os
from unidecode import unidecode
from fireworks.client import Fireworks

def model_call(prompt, temp=0.0): 
    """
   Function to test models
    """
    ## Fireworks
    response = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3p1-405b-instruct", #change to correct model!!!!!!!!
        temperature=temp,
        # top_p=1.0,
        max_tokens=800,
       


        messages = prompt
    )

    response = response.choices[0].message.content

    print(response, flush=True)

    return response

def extract_output(html):
    soup = BeautifulSoup(html, 'html.parser')

    name_tag = soup.find('name')

    if name_tag:
        return name_tag.decode_contents()

    return None

def predict(lang, passage):
    few_shot_examples = {
        "es": {
            "example_passage": "Creo que incluso [MASK] estaría de acuerdo con tu opinión",
            "example_name": "Kafka"
        },
        "tr": {
            "example_passage": "Sadece [MASK] değil tüm grup hayduttu ve diğer insanlardan ayrı yaşıyorlardı",
            "example_name": "Robin"
        },
        "vi": {
            "example_passage": "Rõ ràng đây là điều rất bình thường, chẳng có ai ăn bánh mà lại bị to ra hay bé đi cả, nhưng [MASK] đang quá trông chờ vào những điều bất thường nên khi cái điều bình thường đó xảy ra đã khiến cô cảm thấy cuộc đời sao mà ảm đạm và đáng chán.",
            "example_name": "Alice"
        },
        "en": {
            "example_passage": "Stay gold, [MASK], stay gold.",
            "example_name": "Ponyboy"
        }
    }

    example = few_shot_examples.get(lang)

    if example is None:
        raise ValueError(f"Invalid language code: {lang}")

    text_template = f"""
        You are provided with a passage from a book. Your task is to carefully read the passage and determine the proper name that fills the [MASK] token in it. This name is exactly one word long, and is a proper name (not a pronoun or any other word). You must make a guess, even if you are uncertain:

        Here is an example:
        <passage>{example['example_passage']}</passage>
        <name>{example['example_name']}</name>

        Here is the passage:
        <passage>{passage}</passage>

        Use the following format as output:
        <name>Name</name>
    """

    
    messages=[            {"role": "user", "content": text_template.format(
                example_passage=example["example_passage"],
                example_output=example["example_name"],
                passage=passage
            )}
        ]
    completion = model_call(messages)
    extract = extract_output(completion)
    if extract:
        return extract
    else:
        print(completion)
    return completion

def name_cloze_task(csv_file_name, book_title):
    # try:
        df = pd.read_csv(csv_file_name)

        for col in df.columns:
            if col not in ['Single_ent','Unnamed: 0']:
                print(f'///running {col}///')
                output = []
                for i in range(len(df)):
                    masked_passage = df[col].iloc[i]
                    # print(col.split('_')[0])
                    content = predict(col.split('_')[0], masked_passage)
                    print(i, content)
                    output.append(content)
                index_of_language = df.columns.get_loc(col)
                guess_results = pd.Series(output)
                df.insert(index_of_language + 1, f"{col}_results", guess_results)
        df.to_csv(f"/home/ekorukluoglu/beam3/BEAM/scripts/name_cloze_task/fireworks_out/{book_title}_nct_llama405b.csv", index=False, encoding='utf-8')
    # except:
    #     print(f'{csv_file_name} is missing')


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
    
    skip_list = ['raw', 'Adventures_of_Huckleberry_Finn', 'A_Tale_of_Two_Cities', 'A_thousand_splendid_suns', 'Below_Zero', 'Bride', 'Dracula', 'Dune', 'Fahrenheit_451', 'Harry_Potter_and_the_Deathly_Hallows', 'If_Only_I_Had_Told_Her', 'Just_for_the_Summer', 'Lies_and_Weddings', 'Paper_Towns', 'Percy_Jackson_The_Lightning_Thief', 'Sense_and_sensibility', 'The_Boy_in_the_Striped_Pyjamas', 'The_Handmaid_s_Tale', 'The_Ministry_of_Time', 'The_Paradise_Problem', 'The_Picture_of_Dorian_Gray','Adventures_of_Sherlock_Holmes']
    
    for title in titles:
        if title not in skip_list:
            print(f'----------------- running {title} -----------------')
            name_cloze_task(csv_file_name=f"/home/ekorukluoglu/beam3/BEAM/scripts/Prompts/{title}/{title}_filtered_masked.csv", book_title=title)
