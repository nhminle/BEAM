import os
from openai import OpenAI
import pandas as pd
import re
from unidecode import unidecode
from bs4 import BeautifulSoup


client = OpenAI(
    api_key=os.environ['TOGETHER_API_KEY'],
    base_url="https://api.together.xyz/v1",
)


def extract_output(html):
    # Parse the HTML
    soup = BeautifulSoup(html, 'html.parser')

    # Find the <translation> tag and extract its text
    translation_tag = soup.find('output')

    # If the <translation> tag is found, return its text
    if translation_tag:
        return translation_tag.decode_contents()

    # If the <translation> tag is not found, return None or an appropriate message
    return None


def predict(lang, passage):
    if lang == "es":
        text = f"""
            You are provided with a passage in {lang}. Your task is to carefully read the passage and identify the book it originated from and the book's author.

            Example:
            <passage>Lord Henry alzó las cejas y lo miró con asombro a través de las delgadas volutas de humo que, al salir de su cigarrillo con mezcla de opio, se retorcían adoptando extrañas formas.</passage>
            <output>"title": "The Picture of Dorian Gray","author": "Oscar Wilde"</output>

            Now, here is the passage. Please provide the book title and its author:
            <passage>{passage}</passage>
            You must format your output exactly as follows:
            <output>"title": "Book title","author": "author name"</output>
        """
    elif lang == "tr":
        text = f"""
            You are provided with a passage in {lang}. Your task is to carefully read the passage and identify the book it originated from and the book's author.

            Example:
            <passage>Kendimden pek çok şey kattım buna.” Lord Henry divana boylu boyunca uzanarak güldü.</passage>
            <output>"title": "The Picture of Dorian Gray","author": "Oscar Wilde"</output>

            Now, here is the passage. Please provide the book title and its author:
            <passage>{passage}</passage>
            You must format your output exactly as follows:
            <output>"title": "Book title","author": "author name"</output>
        """
    elif lang == "vi":
        text = f"""
            You are provided with a passage in {lang}. Your task is to carefully read the passage and identify the book it originated from and the book's author.

            Example:
            <passage>Oliver không hề thích thú kiểu đùa giỡn này và chuẩn bị thổ lộ sự bất bình của mình với hai anh bạn</passage>
            <output>"title": "Oliver Twist","author": "Charles Dickens"</output>

            Now, here is the passage. Please provide the book title and its author:
            <passage>{passage}</passage>
            You must format your output exactly as follows:
            <output>"title": "Book title","author": "author name"</output>
        """
    else:
        text = f"""
            You are provided with a passage in {lang}. Your task is to carefully read the passage and identify the book it originated from and the book's author.

            Example:
            <passage>I am afraid I must be going, Basil.</passage>
            <output>"title": "The Picture of Dorian Gray", "author": "Oscar Wilde"</output>

            Now, here is the passage. Please provide the book title and its author:
            <passage>{passage}</passage>
            You must format your output exactly as follows:
            <output>"title": "Book title","author": "author name"</output>
        """

    completion = client.chat.completions.create(

        model="meta-llama/Llama-3-8b-chat-hf",

        max_tokens=100,

        messages=[
            {"role": "user", "content": text}
        ],
        temperature=0.0
    )
    
    extract = extract_output(completion.choices[0].message.content)
    if extract:
        return extract
    else:
        print(completion.choices[0].message.content)
    return completion.choices[0].message.content


def eval_guess(correct, guess):
    correct = correct.strip().lower()
    guess = unidecode(guess.strip().lower())

    # Use regex to check for whole word match
    correct_pattern = re.compile(rf'\b{re.escape(correct)}\b', re.IGNORECASE)
    guess_pattern = re.compile(rf'\b{re.escape(guess)}\b', re.IGNORECASE)

    if correct_pattern.search(guess) or guess_pattern.search(correct):
        return "Correct"
    return "Incorrect"


def direct_probe(csv_file_name, book_title, book_author):
    try:
        df = pd.read_csv(csv_file_name)

        for language in df.columns:
            if language != 'Entity':
                print(f'Running {language}')
                output = []
                eval_title = []
                eval_author = []
                passage_no = 0
                for i in range(len(df)):
                    passage = df[language].iloc[i]
                    content = predict(language, passage)
                    output.append(content)
                    title_eval = eval_guess(book_title, content)
                    author_eval = eval_guess(book_author, content)
                    eval_title.append(title_eval)
                    eval_author.append(author_eval)
                    print(f'{passage_no}: {content}')
                    print(f'title eval: {title_eval}, author eval: {author_eval}')
                    passage_no += 1
                index_of_language = df.columns.get_loc(language)
                output_col = pd.Series(output)
                df.insert(index_of_language + 1, f"{language}_results", output_col)
                title_col = pd.Series(eval_title)
                df.insert(index_of_language + 2, f"{language}_title", title_col)
                author_col = pd.Series(eval_author)
                df.insert(index_of_language + 3, f"{language}_author", author_col)

        df.to_csv(f"/direct_probing/out/{title.replace(' ', '_')}_direct-probe_llama3", index=False, encoding='utf-8')
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
    skip_list = ['raw']
    for title in titles:
        if title not in skip_list:
            print(f'----------------- running {title} -----------------')
            direct_probe(csv_file_name=f"/Prompts/{title}/{title}_ner.csv", book_title=title.replace('_', ' '), book_author=read_txt_file(f'/Prompts/{title}/author.txt'))
            