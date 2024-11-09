import csv
from bs4 import BeautifulSoup
from openai import OpenAI
import os
import evaluate


client = OpenAI(
    api_key=os.environ['TOGETHER_API_KEY'],
    base_url="https://api.together.xyz/v1",
)


def longest_common_subsequence(str1, str2):
    # Create a 2D array to store lengths of longest common subsequence.
    dp = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]

    # Build the dp array
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # Now, reconstruct the longest common subsequence from the dp array.
    lcs = []
    i, j = len(str1), len(str2)
    while i > 0 and j > 0:
        if str1[i - 1] == str2[j - 1]:
            lcs.append(str1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    # Since we've built lcs from the end, we need to reverse it before returning.
    return ''.join(reversed(lcs))


def token_overlap(model_completion, gold_completion):
    # Tokenize both completions. Here we simply use split for simplicity.
    model_tokens = model_completion.split()
    gold_tokens = gold_completion.split()

    # Count the matching tokens from the start
    matching_count = 0
    for m_token, g_token in zip(model_tokens, gold_tokens):
        if m_token == g_token:
            matching_count += 1
        else:
            break

    return matching_count


def predict(passage, bookname, lang, word_count):
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
            You are provided with a passage from {bookname} translated to {lang}. Your task is to complete this passage according to {bookname} within {word_count} words.

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

    completion = client.chat.completions.create(

        model="meta-llama/Llama-3-8b-chat-hf",

        max_tokens=250,

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


def split_sentence_in_half(sentence):
    words = sentence.split()  # Split the sentence into words
    midpoint = len(words) // 2  # Find the midpoint

    # Split the list of words into two halves
    first_half = ' '.join(words[:midpoint])
    second_half = ' '.join(words[midpoint:])

    return first_half, second_half, len(words)


def extract_output(text):
    soup = BeautifulSoup(text, 'html.parser')
    passages = soup.find_all('continuation')

    if passages:
        return max([passage.get_text() for passage in passages])
    else:
        return None


def trim_starting_similarity(string1, string2):
    # Find the length of the common prefix
    min_len = min(len(string1), len(string2))
    i = 0
    while i < min_len and string1[i] == string2[i]:
        i += 1
    
    # Return the remaining part of the second string
    return string2[i:]

def slice_full_words(text, limit):
    # If the text length is within the limit, return the whole string
    if len(text) <= limit:
        return text
    # Check if the last character in the slice is a space
    elif text[limit] == ' ':
        return text[:limit]
    else:
        # Find the next space after the limit
        next_space = text.find(' ', limit)
        if next_space == -1:  # No space found, return the whole string
            return text
        else:
            return text[:next_space]


def prefixProbe(csv_file_name, book_title, rouge, bleurt):
    try:
        languages = ["en", "vi", "es", "tr"]
        full_passages_by_language = {lang: [] for lang in languages}

        with open(csv_file_name, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                for lang in languages:
                    full_passages_by_language[lang].append(row[lang])

        
        results = []

        num_passages = len(full_passages_by_language["en"])  
        for i in range(num_passages):
            print(f'---- {i} ----')
            row = []
            for language in languages:
                full_passage = full_passages_by_language[language][i]
                first_half, second_half, word_count = split_sentence_in_half(full_passage)
                try:
                    print(f"Running prompt for {language}: {first_half}")
                    completion = predict(first_half, book_title, language, word_count)
                    trimmed_completion = slice_full_words(trim_starting_similarity(first_half, completion), len(second_half))
                    print(trimmed_completion)
                    rougel = rouge.compute(predictions=[trimmed_completion], references=[second_half], rouge_types=['rougeL'])['rougeL']
                    if not rougel: rougel = 0
                    print(rougel)
                    bleurt_score = bleurt.compute(predictions=[trimmed_completion], references=[second_half])['scores'][0]
                    if not bleurt_score: bleurt_score = -1
                    print(bleurt_score)
                    row.extend([first_half, second_half, trimmed_completion, longest_common_subsequence(second_half, trimmed_completion), rougel, bleurt_score])
                except Exception as e:
                    row.extend([first_half, str(e), False])
            results.append(row)

        with open(f"/prefix_probing/out/{book_title.replace(' ', '_')}_prefix_probe_llama3.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = []
            for lang in languages:
                header.extend([f'{lang} first half', f'{lang} second half', f'{lang} Completion', f'{lang} lcs', f'{lang} rougeL score', f'{lang} bleurt_score'])
            writer.writerow(header)
            writer.writerows(results)

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
    rouge = evaluate.load('rouge')
    bleurt = evaluate.load("bleurt", module_type="metric")
    titles = get_folder_names('/Prompts')
    skip_list = ['raw']
    for title in titles:
        if title not in skip_list:
            print(f'----------------- running {title} -----------------')
            prefixProbe(csv_file_name=f"/Prompts/{title}/{title}_ner.csv", book_title=title.replace('_', ' '), rouge=rouge, bleurt=bleurt)
    