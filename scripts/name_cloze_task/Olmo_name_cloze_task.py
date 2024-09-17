import os
import openai
import pandas as pd
import re
from bs4 import BeautifulSoup
from unidecode import unidecode

client = openai.OpenAI(
    api_key=os.environ['TOGETHER_API_KEY'],
    base_url="https://api.together.xyz/v1",
)

def extract_output(text):
    # Use regular expression to find the text between square brackets
    return re.search(r'"(.*?)"', text).group(1)


def predict(lang, passage):
    if lang == "es":
        text = f"""
            You are provided with a passage in {lang}. Your task is to carefully read the passage and determine what proper name goes into the [MASK]. This name is a proper name (not a pronoun or any other word). You must make a guess, even if you are uncertain.

            Example input:
            <passage>Creo que incluso [MASK] estaría de acuerdo con tu opinión</passage>
            Example output:
            The proper name that fills in the [MASK] is "Kafka"

            Read the following passage and determine what proper name fills in the [MASK]:
            <passage>{passage}</passage>
            You must format your output exactly as follows:
            The proper name that fills in the [MASK] is "..." 
        """
    elif lang == "tr":
        text = f"""
            You are provided with a passage in {lang}. Your task is to carefully read the passage and determine what proper name goes into the [MASK]. This name is a proper name (not a pronoun or any other word). You must make a guess, even if you are uncertain.

            Example input:
            <passage>Sadece [MASK] değil tüm grup hayduttu ve diğer insanlardan ayrı yaşıyorlardı</passage>
            Example output:
            The proper name that fills in the [MASK] is "Robin"

            Read the following passage and determine what proper name fills in the [MASK]:
            <passage>{passage}</passage>
            You must format your output exactly as follows:
            The proper name that fills in the [MASK] is "..." 
            """
    elif lang == "vi":
        text = f"""
            You are provided with a passage in {lang}. Your task is to carefully read the passage and determine what proper name goes into the [MASK]. This name is a proper name (not a pronoun or any other word). You must make a guess, even if you are uncertain.

            Example input:
            <passage>"Rõ ràng đây là điều rất bình thường, chẳng có ai ăn bánh mà lại bị to ra hay bé đi cả, nhưng [MASK] đang quá trông chờ vào những điều bất thường nên khi cái điều bình thường đó xảy ra đã khiến cô cảm thấy cuộc đời sao mà ảm đạm và đáng chán."</passage>
            Example output:
            The proper name that fills in the [MASK] is "Alice"

            Read the following passage and determine what proper name fills in the [MASK]:
            <passage>{passage}</passage>
            You must format your output exactly as follows:
            The proper name that fills in the [MASK] is "..." 
        """
    else:
        text = f"""
            You are provided with a passage in {lang}. Your task is to carefully read the passage and determine what proper name goes into the [MASK]. This name is a proper name (not a pronoun or any other word). You must make a guess, even if you are uncertain.

            Example input:
            <passage>Stay gold, [MASK], stay gold.</passage>
            Example output:
            The proper name that fills in the [MASK] is "Ponyboy"

            Read the following passage and determine what proper name fills in the [MASK]:
            <passage>{passage}</passage>
            You must format your output exactly as follows:
            The proper name that fills in the [MASK] is "..."            
            """

    completion = client.chat.completions.create(
        model="allenai/OLMo-7B-Instruct",

        max_tokens=50,

        messages=[
            {"role": "user", "content": text}
        ],
        temperature=0.0
    )
    print(completion.choices[0].message.content)
    extract = extract_output(completion.choices[0].message.content)
    if extract:
        return extract
    # else:
    #     print(completion.choices[0].message.content)
    return completion.choices[0].message.content


def eval_guess(ent, guess):
    print(f'ent: {ent} <-> guess: {guess}')
    ent = ent.strip().lower()
    guess = unidecode(guess.strip().lower())
    if ent in guess or guess in ent:
        print('correct')
        return "Correct"
    print('incorrect')
    return "Incorrect"

def name_cloze_task(csv_file_name, book_title):
    try:
        df = pd.read_csv(csv_file_name)

        for col in df.columns:
            if col != 'Entity':
                print(f'///running {col}///')
                output = []
                correct_guess = []
                for i in range(len(df)):
                    masked_passage = df[col].iloc[i]
                    content = predict(col, masked_passage)
                    print(i, content)
                    output.append(content)
                    ent = df['Entity'].iloc[i]
                    correct_guess.append(eval_guess(ent, content))
                index_of_language = df.columns.get_loc(col)
                guess_results = pd.Series(output)
                df.insert(index_of_language + 1, f"{col}_results", guess_results)
                correctness = pd.Series(correct_guess)
                df.insert(index_of_language + 2, f"{col}_correctness", correctness)
        df.to_csv(f"/name_cloze_task/out/{book_title.replace(' ', '_')}_nct_olmo.csv", index=False, encoding='utf-8')
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
    # titles = get_folder_names('/Prompts')
    # skip_list = ['raw']
    # for title in titles:
    #     if title not in skip_list:
    #         print(f'----------------- running {title} -----------------')
    #         name_cloze_task(csv_file_name=f"/Prompts/{title}/{title}_ner_masked.csv", book_title=title.replace('_', ' '))
    title = 'Sense_and_sensibility'
    name_cloze_task(csv_file_name=f"/Prompts/{title}/{title}_ner_masked.csv", book_title=title.replace('_', ' '))