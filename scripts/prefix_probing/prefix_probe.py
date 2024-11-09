import os
import pandas as pd
import re
from vllm import LLM, SamplingParams
from argparse import ArgumentParser


def predict(passages, lang, bookname, word_counts, llm):

    SYSTEM_PROMPT = "You are a helpful assistant. You follow every single instruction carefully."

    few_shot_examples = {
        "es": {
            "example_passage": "Ciertamente, si había algo de lo que estaban completamente seguros los animales, era de no",
            "example_output": "querer la vuelta de Jones. Cuando se les presentaba de esta forma, no sabían qué decir."
        },
        "tr": {
            "example_passage": "Hayvanların en küçük bir kuşku duymadıkları tek bir",
            "example_output": "şey varsa, o da Jones'un geri dönmesini istemedikleriydi."
        },
        "vi": {
            "example_passage": "Rõ ràng đây là điều rất bình thường, chẳng có ai ăn bánh mà lại bị to ra hay bé đi cả, nhưng Alice đang quá trông chờ",
            "example_output": "vào những điều bất thường nên khi cái điều bình thường đó xảy ra đã khiến cô cảm thấy cuộc đời sao mà ảm đạm và đáng chán"
        },
        "en": {
            "example_passage": "As Gregor Samsa awoke one morning from uneasy dreams",
            "example_output": "he found himself transformed in his bed into a gigantic insect."
        }
    }

    example = few_shot_examples.get(lang)


    text_template = """
        You are provided with an incomplete passage from a book. Your task is to complete the passage in the SAME LANGUAGE with more than {word_count} words. Return the continuation as directly quoted from the book, do not include the original input passage.

        Here is an example:
        input: <passage>{example_passage}</passage>
        output: <passage>{example_output}</passage>

        Here is the passage:
        <passage>{passage}</passage>

        You must format your output exactly as follows:
        output: ...
    """
    print(text_template)

    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1000)

    prompts = tokenizer.apply_chat_template(
        [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text_template.format(
                book_name=bookname,
                word_count=word_count,
                example_passage=example["example_passage"],
                example_output=example["example_output"],
                passage=passage
            )},
            ] for passage, word_count in zip(passages, word_counts)
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)

    batch_results = []
    for output in outputs:
        extract = output.outputs[0].text
        extract = extract.replace('\n', ' ')

        batch_results.append(extract)

    return batch_results


def split_sentence_in_half(sentence):
    words = sentence.split()  
    midpoint = len(words) // 2  

    first_half = ' '.join(words[:midpoint])
    second_half = ' '.join(words[midpoint:])

    return first_half, second_half, len(words)


def trim_common_prefix_suffix(string1, string2):
    string2 = re.sub(r'^[^a-zA-Z]+', '', string2)

    # Helper function to remove non-word characters for flexible comparison
    def clean_text(text):
        return re.sub(r'\W+', '', text)  # Keep only word characters (letters and numbers)

    # Cleaned versions for comparison, ignoring non-word characters
    cleaned_string1 = clean_text(string1)
    cleaned_string2 = clean_text(string2)

    # Find the longest matching suffix of `cleaned_string1` that aligns with the start of `cleaned_string2`
    for i in range(len(cleaned_string1)):
        suffix = cleaned_string1[i:]
        if cleaned_string2.startswith(suffix):
            # Rebuild the matched prefix from the original `string2`, up to the length of the matched suffix
            match_position = 0
            count = 0
            for char in string2:
                if re.match(r'\w', char):  # Only count word characters
                    count += 1
                match_position += 1
                if count == len(suffix):  # Stop once we've matched the suffix length
                    break
            return string2[match_position:].strip()

    # If no match is found, return `string2` unchanged
    return string2


def remove_extra_suffix(text, limit):
    if len(text) <= limit:
        return text
    elif text[limit] == ' ':
        return text[:limit]
    else:
        next_space = text.find(' ', limit)
        if next_space == -1:  
            return text
        else:
            return text[:next_space]


def prefixProbe(csv_file_name, book_title, llm, model_name):
    try:
        df = pd.read_csv(csv_file_name)
        languages = ["en", "vi", "es", "tr"]
        df.drop('Single_ent', axis=1, inplace=True)
        for lang in languages:
            df[[f"{lang}_first_half", f"{lang}_second_half", f"{lang}_word_count"]] = df[lang].apply(
                lambda x: pd.Series(split_sentence_in_half(x)) if pd.notnull(x) else pd.Series([x, x, 0])
            )
            df.drop(lang, axis=1, inplace=True)
            print(f'Running {lang}')
            passages = df[f"{lang}_first_half"].tolist()
            word_counts = df[f"{lang}_word_count"].tolist()
            output = predict(passages, lang, book_title.replace('_', ' '), word_counts, llm)

            index_of_lang = df.columns.get_loc(f"{lang}_word_count")
            df.insert(index_of_lang + 1, f"{lang}_results_raw", pd.Series(output))
            # trim of the first half
            df.insert(index_of_lang + 1, f"{lang}_results", df.apply(
                lambda row: trim_common_prefix_suffix(row[f"{lang}_first_half"], row[f"{lang}_results_raw"]), 
                axis=1
            ))

        df.to_csv(f"/home/nhatminhle_umass_edu/Tasks/out/prefix_probe/{book_title}_prefix_probe_{model_name}.csv", index=False, encoding='utf-8')
        print(f'saved to /home/nhatminhle_umass_edu/Tasks/out/prefix_probe/{book_title}_prefix_probe_{model_name}.csv')

    except Exception as e:
        print(e)


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
    
    # titles = get_folder_names('/home/nhatminhle_umass_edu/Prompts')
    # skip_list = ['raw']
    # for title in titles:
    #     if title not in skip_list:
    #         print(f'----------------- running {title} -----------------')
    #         prefixProbe(csv_file_name=f"/home/nhatminhle_umass_edu/Prompts/{title}/{title}_filtered.csv", book_title=title, llm=llm, model_name=args.model.split('/')[1])
    title = 'A_Tale_of_Two_Cities'
    prefixProbe(csv_file_name=f"/home/nhatminhle_umass_edu/Prompts/{title}/{title}_filtered.csv", book_title=title, llm=llm, model_name=args.model.split('/')[1])