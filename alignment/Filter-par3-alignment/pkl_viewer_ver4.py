import re
import pickle
import pandas as pd
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import os
import string
import sacrebleu


def check_intersection(str1, str2):
    words1 = re.findall(r'\w+|[^\w\s]', str1)
    words2 = re.findall(r'\w+|[^\w\s]', str2)

    for i in range(len(words1)):
        if ' '.join(words1[i:]) == ' '.join(words2[:len(words1) - i]):
            return True

    for i in range(len(words1), 0, -1):
        if ' '.join(words1[:i]) == ' '.join(words2[-i:]):
            return True

    return False


def find_tags_for_sentence(full_text):
        pattern = re.compile(r'(<t\d+>.*?</t\d+>)', re.DOTALL)
        matches = pattern.findall(full_text)

        tag_contents = {}
        for match in matches:
            tag = re.search(r'<(t\d+)>', match).group(1)
            content = re.sub(r'<.*?>', '', match).strip()
            tag_contents[tag] = content

        return list(tag_contents.items())


def extract_tags(tag_contents, sentence, start_index=0):
        tags = []
        if sentence:
            sentence = re.sub(r'^[\W\s]+|[\W\s]+$', '', sentence)
            for tag, content in tag_contents[start_index:]:
                content = re.sub(r'^[\W\s]+|[\W\s]+$', '', content)
                if content:
                    if content in sentence:
                        tags.append(tag)
                    elif sentence in content:
                        tags.append(tag)
                        break
                    elif tags:
                        break
                    elif check_intersection(content, sentence) or check_intersection(sentence, content):
                        tags.append(tag)
        return tags


def extract_passages(soup, tags):
        passages = []
        for tag in tags:
            elements = soup.find_all(tag)
            for element in elements:
                text = element.get_text(strip=True)
                if text:
                    passages.append(text)
        return passages


def process_book_data(pkl_file_path, langs, gpttrans_folder, og_lang_folder, output_raw, output_processed, title):
    start = time.time()

    with open(pkl_file_path, 'rb') as f:
        data = pickle.load(f)

    para_data = {'line_no': list(range(len(data['gt_paras']))),
                 'en': data['gt_paras']}
    for lang in langs:
        para_data.update({lang: data['translator_data'][f'{title}_{lang}_processed_gpttrans']['translator_paras']})
    df = pd.DataFrame(para_data)
    print(df)

    for lang in langs:
        sacrebleu_scores1 = []
        sacrebleu_scores2 = []

        for index, row in tqdm(df.iterrows(), desc=f"Computing sacrebleu Scores for {lang}", total=df.shape[0]):
            addK = sacrebleu.sentence_bleu(row[lang], [row['en']], lowercase=True, smooth_method='add-k', smooth_value=1)
            sacrebleu_score = round(addK.score,3)
            sacrebleu_scores1.append(sacrebleu_score)

            exp = sacrebleu.sentence_bleu(row[lang], [row['en']], lowercase=True, smooth_method='exp')
            sacrebleu_score2 = round(exp.score,3)
            sacrebleu_scores2.append(sacrebleu_score2)

        col_indx = df.columns.get_loc(lang)

        df.insert(col_indx + 1, f'{lang}_sc_add1', sacrebleu_scores1)
        df.insert(col_indx + 2, f'{lang}_sc_exp', sacrebleu_scores2)

    print(df)

    df.to_csv(output_raw, index=False, encoding='utf-8')    

    df['vi_translation'] = df['vi']
    df['tr_translation'] = df['tr']
    df['es_translation'] = df['es']

    def extract_tags_dataframe(tag_contents, sentence):
        nonlocal start_line_number
        tags = extract_tags(tag_contents, sentence, start_line_number)
        if tags:
            for index, tag_content in enumerate(tag_contents):
                if tag_content[0] == tags[-1]:
                    start_line_number = index
        return tags

    soup_dict_gpttrans = {}
    for lang in langs:
        with open(f'{gpttrans_folder}/{title}_{lang}_processed_gpttrans.txt', 'r', encoding='utf-8') as file:
            html = file.read()
        soup_dict_gpttrans[lang] = find_tags_for_sentence(html)

    for lang in langs:
        start_line_number = 0
        df[f'{lang}'] = df[f'{lang}'].apply(lambda col: extract_tags_dataframe(soup_dict_gpttrans[lang], col))

    print(df)

    soup_dict = {}
    for lang in langs:
        with open(f'{og_lang_folder}/{title}_{lang}_processed.txt', 'r', encoding='utf-8') as file:
            html = file.read()
        soup_dict[lang] = BeautifulSoup(html, 'html.parser')

    for lang in tqdm(langs, desc="Processing languages"):
        passages = []
        for tags in tqdm(df[f'{lang}'], desc=f"Extracting passages for {lang}", leave=False):
            passages.append(extract_passages(soup_dict[lang], tags))
        df[f'{lang}'] = passages

    for lang in langs:
        df[lang] = df[lang].apply(lambda x: ' '.join(x))

    min_score = 5
    max_score = 100
    smoothing = 'add1'

    # mask to grab only the row within the range
    mask = ((df[f'vi_sc_{smoothing}'] >= min_score) & (df[f'vi_sc_{smoothing}'] <= max_score)) & \
           ((df[f'tr_sc_{smoothing}'] >= min_score) & (df[f'tr_sc_{smoothing}'] <= max_score)) & \
           ((df[f'es_sc_{smoothing}'] >= min_score) & (df[f'es_sc_{smoothing}'] <= max_score))

    df = df[mask]

    # clean csv by removing the sacrebleu score 
    for lang in langs:
        df.drop(f'{lang}_sc_add1', axis=1, inplace=True)
        df.drop(f'{lang}_sc_exp', axis=1, inplace=True)

    # filter out alignment by lenght 
    mask = (df['en'].str.len() * 3 >= df['vi'].str.len()) & \
           (df['en'].str.len() * 3 >= df['es'].str.len()) & \
           (df['en'].str.len() * 3 >= df['tr'].str.len())

    df = df[mask]

    df.to_csv(output_processed, index=False, encoding='utf-8')

    end = time.time()
    print(f"The time of execution of the function is : {end-start} s")
    print(f"The time of execution of the function is : {(end-start)/60} mins")

def get_file_names(directory):
    txt_file_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) and item.endswith('.pkl'):
            txt_file_names.append(item.replace('_aligned.pkl', '').replace('_para', ''))
    return txt_file_names


if __name__ == '__main__':
    alignment_lvl= 'para' # para / sent
    titles = get_file_names(f'/Filter-par3-alignment/pkl/{alignment_lvl}')
    print(titles)
    for title in titles:
        print(f'processing {title}')
        process_book_data(
            pkl_file_path=f'/Filter-par3-alignment/pkl/{alignment_lvl}/{title}_{alignment_lvl}_aligned.pkl',
            langs=['vi', 'tr', 'es'],
            gpttrans_folder=f'/Filter-par3-alignment/gpttrans/{title}',
            og_lang_folder=f'/Filter-par3-alignment/og_lang/{title}',
            output_raw=f'/Filter-par3-alignment/aligned_raw/{alignment_lvl}/{title}_aligned_raw.csv',
            output_processed=f'/Filter-par3-alignment/aligned_filtered/{alignment_lvl}/{title}_aligned.csv',
            title= title,
        )


