import pandas as pd
import tiktoken
import os


def count_token_in_string(string: str, encoding_name: str = "o200k_base") -> int:
    """
    Returns the number of tokens in a text string using the specified encoding.

    Args:
    string (str): The text string to tokenize.
    encoding_name (str): The encoding name. Default is "o200k_base". New for gpt-4o.

    Returns:
    int: The number of tokens in the text string.
    """
    # Load the encoding
    encoding = tiktoken.get_encoding(encoding_name)

    # Convert the string into tokens and count them
    num_tokens = len(encoding.encode(string))
    return num_tokens


def predict(lang, passage, bookname, word_count, prompt_setting="zero-shot"):
    # Demonstrations dictionary to provide an example passage and its continuation if not zero-shot
    demonstrations = {
        "es": {
            "first_half": "Ciertamente, si había algo de lo que estaban completamente seguros los animales, era de no",
            "second_half": "querer la vuelta de Jones. Cuando se les presentaba de esta forma, no sabían qué decir."
        },
        "tr": {
            "first_half": "Hayvanların en küçük bir kuşku duymadıkları tek bir",
            "second_half": "şey varsa, o da Jones'un geri dönmesini istemedikleriydi."
        },
        "vi": {
            "first_half": "Rõ ràng đây là điều rất bình thường, chẳng có ai ăn bánh mà lại bị to ra hay bé đi cả, nhưng Alice đang quá trông chờ",
            "second_half": "vào những điều bất thường nên khi cái điều bình thường đó xảy ra đã khiến cô cảm thấy cuộc đời sao mà ảm đạm và đáng chán."
        },
        "en": {
            "first_half": "As Gregor Samsa awoke one morning from uneasy dreams",
            "second_half": "he found himself transformed in his bed into a gigantic insect."
        }
    }

    demo_passage = ""
    # If prompt_setting is not zero-shot, include an example demonstration
    if prompt_setting != "zero-shot":
        demo = demonstrations.get(lang)
        if demo:
            demo_passage = f"""
            Here is an example:
            <passage>{demo['first_half']}</passage>
            <continuation>{demo['second_half']}</continuation>
            """

    text = f"""
       You are provided with an incomplete passage from a book. Your task is to continue the passage *in the SAME LANGUAGE* with {word_count}. Your response should ONLY be the continuation, directly quoted from the book. Do not repeat or rephrase the input.
        {demo_passage}
        Here is the passage:
        <passage>{passage}</passage>

        Use the following format as output:
       <continuation>Passage continuation</continuation>
    """

    return count_token_in_string(text), count_token_in_string(passage)//2


def direct_probe(csv_file_name, book_title, prompt_setting="zero-shot"):
    try:
        df = pd.read_csv(csv_file_name)
        total_token = 0
        total_lines = 0
        total_output = 0

        for language in df.columns:
            if language != 'Entity':
                for i in range(len(df)):
                    passage = df[language].iloc[i]
                    text, output = predict(language, passage, book_title, 50, prompt_setting=prompt_setting)
                    total_token += text
                    total_output += output
                    total_lines += 1
        
        avg_tokens_per_line = total_token / total_lines
        avg_tokens_per_output = round(total_output / total_lines)
        print(f"Total tokens in {book_title}: {total_token}")
        print(f"Average tokens per line in {book_title}: {round(avg_tokens_per_line)}")
        print(f'Projected token output per line in {book_title}: {avg_tokens_per_output}')
        print(f"Total lines in {book_title}: {total_lines}")
        print(f'Projected cost: {(total_token*5/1000000+(avg_tokens_per_output)*15/1000000)/2}')
        print('\n')
        return (total_token*5/1000000+(20)*15/1000000)/2
    except Exception as e:
        print(f"Error processing {book_title}: {e}")
        return 0

def get_folder_names(directory):
    folder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names

if __name__ == "__main__":
    prompt_setting = "zero-shot"
    titles = get_folder_names('/Users/alishasrivastava/BEAM/scripts/Prompts')
    total = 0
    for title in titles:
        if title != 'raw':
            print(f'----------------- running {title} -----------------')
            total += direct_probe(f"/Users/alishasrivastava/BEAM/scripts/Prompts/{title}/{title}_filtered.csv", title, prompt_setting)
    print(f'\ntotal amount for all books: {total}')
