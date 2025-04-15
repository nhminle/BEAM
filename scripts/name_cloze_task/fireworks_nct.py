from fireworks.client import Fireworks
import logging 
import sys
import io
import time  ### ADDED OR MODIFIED ###
import pandas as pd
from bs4 import BeautifulSoup
from openai import OpenAI
import os
import re

client = Fireworks(api_key="fw_3Zn92C3stccuXKPXACNmnEYV")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logging.getLogger('fireworks.client').setLevel(logging.WARNING)
logging.getLogger('fireworks').setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
# Reconfigure stdout to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def extract_output(llm_output):
    soup = BeautifulSoup(llm_output, 'html.parser')
    name_tag = soup.find('name')
    if name_tag:
        return name_tag.decode_contents()

    return None


def predict(lang, passage, mode="unshuffled", prompt_setting="zero_shot"):
    
    demonstrations = {
         "es": {
            "unshuffled": "Hemos de agregar que quemaba tan hondamente el pecho de [MASK], que quizá había mayor verdad en el rumor que lo que nuestra moderna incredulidad nos permite aceptar.",
            "shuffled": "lo Hemos quemaba de verdad nos moderna rumor hondamente que que el quizá tan en el mayor había que agregar pecho [MASK], que aceptar. de incredulidad permite nuestra"
        },
        "tr": {
            "unshuffled": "Ve [MASK]'ın göğsünü o kadar derinden yaktı ki, belki de modern şüphemizin kabul etmeye meyilli olmadığı söylentide daha fazla gerçeklik vardı.",
            "shuffled": "ki, yaktı göğsünü gerçeklik vardı. meyilli söylentide belki fazla [MASK]'ın derinden olmadığı Ve kadar şüphemizin de kabul modern etmeye daha o"
        },
        "vi": {
            "unshuffled": "Và chúng ta tất phải thuật lại rằng nó đã nung đốt thành dấu hằn vào ngực [MASK] sâu đến nỗi có lẽ trong lời đồn kia có nhiều phần sự thực hơn là đầu óc đa nghi của chúng ta trong thời hiện đại có thể sẵn sàng thừa nhận.",
            "shuffled": "ta phải thuật trong ta trong lẽ thể đại nỗi có nhận. nung đa hằn nghi đốt đồn lời vào dấu sâu Và hơn có sự hiện [MASK] của có phần thực kia ngực sẵn chúng tất thời nhiều sàng chúng đầu rằng đến là lại thừa đã óc nó thành"
        },
        "en": {
            "unshuffled": "And we must needs say, it seared [MASK]'s bosom so deeply, that perhaps there was more truth in the rumor than our modern incredulity may be inclined to admit.",
            "shuffled": "admit. say, to inclined that the be more must so than it may needs modern we in rumor was deeply, incredulity perhaps our seared bosom there [MASK]'s And truth"
        },
        "st": {
            "unshuffled": "'Me re tlameha ho re, earared [MASK] bosom e tebileng haholo, hore mohlomong ho na le 'nete e ngata ka menyenyetsi ho feta ho se lumele ha rona ea kajeno e ka ba tšekamelo ea ho lumela.",
            "shuffled": "earared e re 'Me kajeno e re, ho lumele ea mohlomong e hore menyenyetsi ngata ha ka rona 'nete ba na ka ea le haholo, tšekamelo feta ho tebileng se bosom ho ho lumela. ho tlameha [MASK]"
        },
        "yo": {
            "unshuffled": "Àti pé a gbọ́dọ̀ nílò láti sọ pé, ó mú àyà [MASK] jinlẹ̀, pé bóyá òtítọ́ púpọ̀ wà nínú àròsọ ju àìgbàgbọ́ ìgbàlódé wa lọ lè fẹ́ láti gbà.",
            "shuffled": "nínú a jinlẹ̀, ìgbàlódé [MASK] òtítọ́ ó púpọ̀ pé, pé mú wà láti lọ àròsọ àìgbàgbọ́ sọ wa láti àyà gbọ́dọ̀ lè fẹ́ pé nílò ju Àti gbà. bóyá"
        },
        "tn": {
            "unshuffled": "Mme re tshwanetse ra re, [MASK] le fa go ntse jalo, go ne go na le boammaaruri jo bogolo go feta mo tumelong ya rona ya gompieno.",
            "shuffled": "tumelong le gompieno. re, jalo, mo ya ne feta jo bogolo boammaaruri go go le [MASK] ra ya ntse go na fa Mme rona tshwanetse go re"
        },
        "ty": {
            "unshuffled": "E e ti'a ia tatou ia parau e, ua î roa te ouma o [MASK] i te reira, e peneia'e ua rahi a'e te parau mau i roto i te parau i to tatou ti'aturi-ore-raa no teie tau.",
            "shuffled": "tau. mau E te tatou ti'a teie e parau rahi te e, parau î i ua ia reira, [MASK] te to tatou i parau ua ti'aturi-ore-raa ouma roto te roa i peneia'e a'e ia e no o i"
        },
        "mai": {
            "unshuffled": "आ हमरासभकेँ ई कहबाक आवश्यकता अछि जे ई [MASK] छातीकेँ एतेक गहराई सँ प्रभावित कयलक, जे शायद अफवाहमे ओहिसँ बेसी सत्य छल जतेक हमर आधुनिक अविश्वास स्वीकार करय लेल इच्छुक भऽ सकैत अछि।",
            "shuffled": "अछि आवश्यकता अफवाहमे प्रभावित गहराई [MASK] सँ हमर ओहिसँ अछि। ई लेल बेसी छल जे जतेक हमरासभकेँ अविश्वास करय आ कयलक, स्वीकार कहबाक आधुनिक ई छातीकेँ शायद सकैत एतेक इच्छुक सत्य जे भऽ"
        },
        "mg": {
            "unshuffled": "Ary tsy maintsy mila miteny isika hoe, nampivoaka lalina ny tratran'i [MASK] izany, ka angamba nisy fahamarinana bebe kokoa tao anatin'ilay tsaho fa tsy mety ho mora miaiky ny tsy finoana maoderina ananantsika.",
            "shuffled": "miteny tsy miaiky mora maintsy mila ny tsy kokoa [MASK] tsy fa ka bebe mety tratran'i izany, anatin'ilay fahamarinana ho maoderina lalina tsaho finoana angamba Ary isika tao ananantsika. hoe, nisy nampivoaka ny"
        }

    }

    demo = demonstrations.get(lang)[mode]
    
    demo_passage = ""
    if prompt_setting != "zero_shot":
        demo_passage = f"""
        
        Here is an example:
        <passage>{demo}</passage>
        <output>Hester</output>
        
        """
    
    prompt = f"""
       You are provided with a passage in {lang}. Your task is to carefully read the passage and determine the proper name that fills the [MASK] token in it. This name is a proper name (not a pronoun or any other word). You must make a guess IN ENGLISH, even if you are uncertain.
        {demo_passage}
        Here is the passage:
        <passage>{passage}</passage>

        Use the following format as output:
       <output>Name</output>
    """
    try:
        completion = client.chat.completions.create(
            model="accounts/fireworks/models/llama-v3p1-405b-instruct",

            max_tokens=100,

            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
    except Exception as e:
        logging.error(f"Error calling Fireworks API: {e}")
        time.sleep(1)  # Sleep for 1s
        return f"ERROR: {e}"

    extract = extract_output(completion.choices[0].message.content)
    if extract:
        return extract
    else:
        logging.info(completion.choices[0].message.content)
    return completion.choices[0].message.content

def name_cloze_task(csv_file_name, book_title, prompt_setting="zero-shot"):
    try:
        df = pd.read_csv(csv_file_name)
        # Adjust the languages list to suit your needs
        languages = ["en", "vi", "es", "tr", "mg", "mai", "ty", "tn", "yo", "st"]
        for language in df.columns:
            if language not in['Multiple_ents','Single_ent']:
                # print(f'Running {language}')
                output = []
                mode = "shuffled" if "shuffled" in language.lower() else "unshuffled"
                for i in range(len(df)):
                    masked_passage = df[language].iloc[i]
                    base_language = language.split('_')[0]
                    content = predict(base_language, masked_passage, mode, prompt_setting)
                    #print(f'{i}: {content}')
                    output.append(content)
                index_of_language = df.columns.get_loc(language)
                guess_results = pd.Series(output)
                df.insert(index_of_language + 1, f"{language}_results", guess_results)
                
        output_file_name = (
            f"/home/ekorukluoglu_umass_edu/beam2/BEAM/results/name_cloze/"
            f"Llama-3.1-405b/2024/{book_title}_name_cloze_Llama-3.1-405b_"
            f"{prompt_setting}_masked_passages.csv"
        )
        df.to_csv(output_file_name, index=False, encoding='utf-8')
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
    titles = get_folder_names('/home/ekorukluoglu_umass_edu/beam2/BEAM/scripts/Prompts/2024')
    
    for title in titles:
        if "1984" not in title:
            logging.info(f'----------------- running {title} -----------------')
            name_cloze_task(csv_file_name=f"/home/ekorukluoglu_umass_edu/beam2/BEAM/scripts/Prompts/2024/{title}/{title}_filtered_masked.csv", book_title=title, prompt_setting="zero_shot") # modify the prompt setting here
            name_cloze_task(csv_file_name=f"/home/ekorukluoglu_umass_edu/beam2/BEAM/scripts/Prompts/2024/{title}/{title}_filtered_masked.csv", book_title=title, prompt_setting="one_shot") # modify the prompt setting here