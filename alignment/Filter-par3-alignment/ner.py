import stanza
import pandas as pd
import os
from tqdm import tqdm
import re

def get_folder_names(directory):
    folder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names

def extract_named_entities(text, model):
    doc = model(text)
    return [ent.text.lower() for ent in doc.ents if ent.type == "PERSON" or ent.type == 'PER']

def process_and_filter_csv(en, vi, tr, es, title, alignment_lvl):
    try:
        df = pd.read_csv(f'/Filter-par3-alignment/aligned_filtered/{alignment_lvl}/{title}_aligned.csv')
        models = {'en': en, 'vi': vi, 'tr': tr, 'es': es}

        one_ent = []
        many_ents = []

        for index, row in tqdm(df.iterrows(), desc=f"Processing {title}", total=df.shape[0]):
            has_ent = {'en': 0, 'vi': 0, 'tr': 0, 'es': 0}
            ents = set()

            for col, model in models.items():
                named_entities = extract_named_entities(row[col], model)
                if len(named_entities) == 1:
                    has_ent[col] += 1
                    e = re.sub(r"[.,!?:;\’\"'(){}\[\]/\\|^&@#$%+=-].*$", "", named_entities[0]).strip()
                    ents.add(e) 
                elif len(named_entities) > 1:
                    e = set([re.sub(r"[.,!?:;\’\"'(){}\[\]/\\|^&@#$%+=-].*$", "", n).strip() for n in named_entities])
                    has_ent[col] += len(e)
                    ents.update(e)
                else:
                    break

            if all([x == 1 for x in has_ent.values()]): # Check if all language models found one named entity
                one_ent.append((index, list(ents))) # Store the results for later DataFrame update
            else:
                many_ents.append((index, list(ents)))

        df.insert(1, 'Single_ent', pd.Series(dtype='object'))
        for res in one_ent:
            index, ents = res
            df.at[index, 'Single_ent'] = ents
        df.insert(2, 'Multiple_ents', pd.Series(dtype='object'))
        for res in many_ents:
            index, ents = res
            df.at[index, 'Multiple_ents'] = ents
        
        if alignment_lvl == 'para':
            df.to_csv(f'/Filter-par3-alignment/ner/{title}/{title}_para_ner.csv', index=False)
        else:
            df.to_csv(f'/Filter-par3-alignment/ner/{title}/{title}_ner.csv', index=False)

    except Exception as e:
        print(e)




if __name__ == '__main__':
    en_model = stanza.Pipeline('en', processors='tokenize,ner', use_gpu=True)
    es_model = stanza.Pipeline('es', processors='tokenize,ner', use_gpu=True)
    tr_model = stanza.Pipeline('tr', processors='tokenize,ner', use_gpu=True)
    vi_model = stanza.Pipeline('vi', processors='tokenize,ner', use_gpu=True)

    skip_list = []
    titles = get_folder_names('/gpt_trans/raw_txt')
    for title in titles:
        if title not in skip_list:
            process_and_filter_csv(
                en=en_model,
                vi=vi_model,
                es=es_model,
                tr=tr_model,
                title=title,
                alignment_lvl='para'
            )
