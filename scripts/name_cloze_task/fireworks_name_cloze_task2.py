from bs4 import BeautifulSoup
from openai import OpenAI
import pandas as pd
import os
from fireworks.client import Fireworks
import re
client = Fireworks(api_key=os.environ['FIREWORKS_API_KEY'])

def model_call(prompt, temp=0.0): 
    """
   Function to test models
    """
    ## Fireworks
    response = client.chat.completions.create(
        model="accounts/fireworks/models/llama-v3p1-405b-instruct", #change to correct model!!!!!!!!
        temperature=temp,
        # top_p=1.0,
        max_tokens=800,
       


        messages = prompt
    )

    response = response.choices[0].message.content

    print(response, flush=True)

    return response


def extract_output(html):
    soup = BeautifulSoup(html, 'html.parser')
    name_tag = soup.find('name')
    if name_tag:
        return name_tag.decode_contents()
        # Regex pattern to extract the title and author
# Regex pattern to extract 'Name' and the associated value
    pattern = r'"Name":"[^"]+"'

    # Find matches
    matches = re.findall(pattern, html)

    if matches:
        for match in matches:
            return match
            print(match)

    return "none"


def predict(lang, passage, mode="unshuffled", prompt_setting="zero-shot"):
    
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
        }
    }

    demo = demonstrations.get(lang)[mode]
    
    demo_prompt = ""
    if prompt_setting != "zero-shot":
        demo_prompt = f"""
        
        Here is an example:
        <passage>{demo}</passage>
        <output>"title": "The Scarlet Letter","author": "Nathaniel Hawthorne"</output>
        
        """
    
    prompt = f"""
       You are provided with a passage from a book. Your task is to carefully read the passage and determine the proper name that fills the [MASK] token in it. This name is a proper name (not a pronoun or any other word). You must make a guess, even if you are uncertain:
        {demo_prompt}
        Here is the passage:
        <passage>{passage}</passage>

        Use the following format as output:
       <name>Name</name>
    """

    completion =model_call([{"role":"user","content":prompt}])
    extract = extract_output(completion)
    if extract:
        return extract
    else:
        print(completion)
    return completion

def name_cloze_task(csv_file_name, book_title, prompt_setting="zero-shot"):
    try:
        df = pd.read_csv(csv_file_name)

        for language in df.columns:
            if language not in ['Single_ent','Unnamed: 0']:
                print(f'Running {language}')
                output = []
                for i in range(len(df)):
                    masked_passage = df[language].iloc[i]
                    if "shuffled" in language: 
                        content = predict(language.split('_')[0], masked_passage, "shuffled", prompt_setting)
                    else:
                        content = predict(language.split('_')[0], masked_passage, "unshuffled", prompt_setting)
                    print(f'{i}: {content}')
                    output.append(content)
                index_of_language = df.columns.get_loc(language)
                guess_results = pd.Series(output)
                df.insert(index_of_language + 1, f"{language}_results", guess_results)
                
        df.to_csv(f"./name_cloze_task/fireworks_out/{prompt_setting}/{book_title}_name_cloze_llama405b.csv", index=False, encoding='utf-8')
    except Exception as e:
        print(f'Error: {e}')
        print(f'At column {language}')


def get_folder_names(directory):
    folder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names

if __name__ == "__main__":
    titles = get_folder_names('./Prompts')
    skip_list = [
  "Adventures_of_Sherlock_Holmes",
  "Alice_in_Wonderland",
  "Alice_s_Adventures_in_Wonderland",
  "A_Tale_of_Two_Cities",
  "A_thousand_splendid_suns",
  "Below_Zero",
  "Dracula",
  "Fahrenheit_451",
  "First_Lie_Wins",
  "Frankenstein",
  "Funny_Story",
  "If_Only_I_Had_Told_Her",
  "Just_for_the_Summer",
  "Lies_and_Weddings",
  "Of_Mice_and_Men",
  "Paper_Towns",
  "Pride_and_Prejudice",
  "The_Great_Gatsby",
  "The_Paradise_Problem",
  "The_Picture_of_Dorian_Gray",
  "You_Like_It_Darker_Stories"
]

    for title in titles:
        if title not in skip_list:
            print(f'----------------- running {title} -----------------')
            name_cloze_task(csv_file_name=f"./Prompts/{title}/{title}_filtered_masked.csv", book_title=title, prompt_setting="one-shot") # modify the prompt setting here
