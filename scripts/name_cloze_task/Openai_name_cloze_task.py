from bs4 import BeautifulSoup
from openai import OpenAI
import pandas as pd
import os
from unidecode import unidecode

client = OpenAI(
    api_key=os.environ['OpenAI_API_KEY'],
)

def extract_output(html):
    # Parse the HTML
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

    text_template = """
        You are provided with a passage from a book. Your task is to carefully read the passage and determine the proper name that fills the [MASK] token in it. This name is exactly one word long, and is a proper name (not a pronoun or any other word). You must make a guess, even if you are uncertain:

        Here is an example:
        <passage>{example_passage}</passage>
        <name>{example_name}</name>

        Here is the passage:
        <passage>{passage}</passage>

        Use the following format as output:
       <name>Name</name>
    """

    completion = client.chat.completions.create(
        model="gpt-4o",

        max_tokens=100,

        messages=[
            {"role": "user", "content": text_template.format(
                example_passage=example["example_passage"],
                example_output=example["example_output"],
                passage=passage
            )}
        ],
        temperature=0.0
    )

    extract = extract_output(completion.choices[0].message.content)
    if extract:
        return extract
    else:
        print(completion.choices[0].message.content)
    return completion.choices[0].message.content

def name_cloze_task(csv_file_name, book_title):
    try:
        df = pd.read_csv(csv_file_name)

        for col in df.columns:
            if col != 'Entity':
                print(f'///running {col}///')
                output = []
                for i in range(len(df)):
                    masked_passage = df[col].iloc[i]
                    content = predict(col, masked_passage)
                    print(i, content)
                    output.append(content)
                index_of_language = df.columns.get_loc(col)
                guess_results = pd.Series(output)
                df.insert(index_of_language + 1, f"{col}_results", guess_results)
        df.to_csv(f"{book_title}_nct_gpt4o.csv", index=False, encoding='utf-8')
    except:
        print(f'{csv_file_name} is missing')


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
    titles = get_folder_names('/Prompts')
    for title in titles:
        print(f'----------------- running {title} -----------------')
        name_cloze_task(csv_file_name=f"/Prompts/{title}/{title}_filtered_masked.csv", book_title=title)
