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


# replace the prompt with the correct language when running experiments
def predict(lang, passage, bookname, word_count):
    if lang == "es":
        text = f"""
            You are provided with a passage in {lang} from {bookname}. Your task is to complete this passage according to {bookname} within {word_count} words.

            Example:
            <passage>Ciertamente, si había algo de lo que estaban completamente seguros los animales, era de no</passage>
            <continuation>querer la vuelta de Jones. Cuando se les presentaba de esta forma, no sabían qué decir.</continuation>

            Here is the passage. Please provide its continuation in {lang} according to {bookname}:
            <passage>{passage}</passage>
            You must format your output exactly as follows:
            <continuation>your continuation</continuation>
        """
    elif lang == "tr":
        text = f"""
            You are provided with a passage in {lang} from {bookname}. Your task is to complete this passage according to {bookname} within {word_count} words.

            Example:
            <passage>Hayvanların en küçük bir kuşku duymadıkları tek bir</passage>
            <continuation>şey varsa, o da Jones'un geri dönmesini istemedikleriydi.</continuation>

            Here is the passage. Please provide its continuation in {lang} according to {bookname}:
            <passage>{passage}</passage>
            You must format your output exactly as follows:
            <continuation>your continuation</continuation>
        """
    elif lang == "vi":
        text = f"""
            You are provided with a passage in {lang} from {bookname}. Your task is to complete this passage according to {bookname} within {word_count} words.

            Example:
            <passage>Rõ ràng đây là điều rất bình thường, chẳng có ai ăn bánh mà lại bị to ra hay bé đi cả, nhưng Alice đang quá trông chờ</passage>
            <continuation>vào những điều bất thường nên khi cái điều bình thường đó xảy ra đã khiến cô cảm thấy cuộc đời sao mà ảm đạm và đáng chán.</continuation>

            Here is the passage. Please provide its continuation in {lang} according to {bookname}:
            <passage>{passage}</passage>
            You must format your output exactly as follows:
            <continuation>your continuation</continuation>
        """
    else:
        text = f"""
            You are provided with a passage in {lang} from {bookname}. Your task is to complete this passage according to {bookname} within {word_count} words.
            
            Example:
            <passage>As Gregor Samsa awoke one morning from uneasy dreams</passage>
            <continuation>he found himself transformed in his bed into a gigantic insect.</continuation>
            
            Here is the passage. Please provide its continuation in {lang} according to {bookname}:
            <passage>{passage}</passage>
            You must format your output exactly as follows:
            <continuation>your continuation</continuation>
        """
    return count_token_in_string(text), count_token_in_string(passage)//2


def direct_probe(csv_file_name, book_title):
    try:
        df = pd.read_csv(csv_file_name)
        total_token = 0
        total_lines = 0
        total_output = 0

        for language in df.columns:
            if language != 'Entity':
                for i in range(len(df)):
                    passage = df[language].iloc[i]
                    text, output = predict(language, passage, book_title, 50)
                    total_token += text
                    total_output += output
                    total_lines += 1
        
        avg_tokens_per_line = total_token / total_lines
        avg_tokens_per_output = round(total_output / total_lines)
        print(f"Total tokens in {title}: {total_token}")
        print(f"Average tokens per line in {title}: {round(avg_tokens_per_line)}")
        print(f'Projected token output per line in {title}: {avg_tokens_per_output}')
        print(f"Total lines in {title}: {total_lines}")
        print(f'Projected cost: {(total_token*5/1000000+(avg_tokens_per_output)*15/1000000)/2}')
        print('\n')
        return (total_token*5/1000000+(20)*15/1000000)/2
    except Exception as e:
        print(e)
        return 0

def get_folder_names(directory):
    folder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names

if __name__ == "__main__":
    titles = get_folder_names('/home/nhatminhle_umass_edu/Prompts')
    total = 0
    for title in titles:
        if title != 'raw':
            print(f'----------------- running {title} -----------------')
            total += direct_probe(f"/home/nhatminhle_umass_edu/Prompts/{title}/{title}_ner.csv", title)
    print(f'\ntotal amount for all books: {total}')