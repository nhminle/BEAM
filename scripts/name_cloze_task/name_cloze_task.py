import os
import pandas as pd
from bs4 import BeautifulSoup
from vllm import LLM, SamplingParams
from argparse import ArgumentParser

def extract_output(llm_output):
    soup = BeautifulSoup(llm_output, 'html.parser')
    translation_tag = soup.find('name')
    if translation_tag:
        return translation_tag.decode_contents()
    return None

def predict(lang, passages, llm):

    SYSTEM_PROMPT = "You are a helpful assistant. You follow instructions carefully."

    example_passages = {
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

    details = example_passages.get(lang)

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

    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=100)

    prompts = tokenizer.apply_chat_template(
        [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text_template.format(
                    example_passage=details["example_passage"],
                    example_name=details["example_name"],
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
                output = predict(language.replace('_masked', ''), passages, llm)

                index_of_language = df.columns.get_loc(language)
                df.insert(index_of_language + 1, f"{language}_results", pd.Series(output))

        df.to_csv(f"/home/nhatminhle_umass_edu/Tasks/out/name_cloze/{book_title}_name_cloze_{model_name}.csv", index=False, encoding='utf-8')
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
    titles = get_folder_names('/home/nhatminhle_umass_edu/Prompts')
    
    parser = ArgumentParser()
    parser.add_argument("model", type=str, help="Name of the model to use")
    args = parser.parse_args()

    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=2048)
    
    skip_list = ['raw']
    for title in titles:
        if title not in skip_list:
            print(f'----------------- running {title} -----------------')
            direct_probe(csv_file_name=f"/home/nhatminhle_umass_edu/Prompts/{title}/{title}_filtered_masked.csv", book_title=title, llm=llm, model_name=args.model.split('/')[1])