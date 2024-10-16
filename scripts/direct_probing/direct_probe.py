import os
import pandas as pd
from bs4 import BeautifulSoup
from vllm import LLM, SamplingParams
from argparse import ArgumentParser

def extract_output(llm_output):
    soup = BeautifulSoup(llm_output, 'html.parser')
    translation_tag = soup.find('output')
    if translation_tag:
        return translation_tag.decode_contents()
    
    return None

def predict(lang, passages, llm):

    SYSTEM_PROMPT = "You are a helpful assistant. You follow instructions carefully."

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

    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100)

    prompts = tokenizer.apply_chat_template(
        [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text_template.format(
                    lang=lang,
                    example_passage=example["example_passage"],
                    example_output=example["example_output"],
                    passage=passage
                )},
            ] for passage in passages
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    batch_results = []
    for output in outputs:
        extract = extract_output(output.outputs[0].text)
        if not extract:
            extract = output.outputs[0].text
        extract = extract.replace('\n', ' ')
        batch_results.append(extract)

    return batch_results

def direct_probe(csv_file_name, book_title, llm, model_name):
    try:
        df = pd.read_csv(csv_file_name)

        for language in df.columns:
            if language != 'Single_ent':
                print(f'Running {language}')
                passages = df[language].tolist()  
                output = predict(language, passages, llm)

                index_of_language = df.columns.get_loc(language)
                df.insert(index_of_language + 1, f"{language}_results", pd.Series(output))

        df.to_csv(f"{book_title}_direct_probe_{model_name}.csv", index=False, encoding='utf-8')
    except Exception as e:
        print(f'Error: {e}')

def get_folder_names(directory):
    folder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names

if __name__ == "__main__":
    titles = get_folder_names('/Prompts')
    
    parser = ArgumentParser()
    parser.add_argument("model", type=str, help="Name of the model to use")
    args = parser.parse_args()

    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=2048)
    
    for title in titles:
        print(f'----------------- running {title} -----------------')
        direct_probe(csv_file_name=f"/Prompts/{title}/{title}_filtered.csv", book_title=title, llm=llm, model_name=args.model.split('/')[1])
