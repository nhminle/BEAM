from bs4 import BeautifulSoup
from openai import OpenAI
import pandas as pd
import os
import time
import sys
from fireworks.client import Fireworks
import re
client = Fireworks(api_key=os.environ['FIREWORKS_API_KEY'])

def model_call(prompt, temp=0.0,retries =3):
    if retries>0:
        try:
            """
            Function to test models
            """
            ## Fireworks
            
            response = client.chat.completions.create(
                model="accounts/fireworks/models/llama-v3p1-405b-instruct", #change to correct model!!!!!!!!
                temperature=temp,
            # top_p=1.0,
                max_tokens=100,
                messages=[{
                "role": "user",
                "content": prompt,
                }],
            )
            
            response = response.choices[0].message.content

    #        print(response, flush=True)

            return response
        except Exception as e:
            print("Some error happened with api, retrying..." + str(e))
            time.sleep(5)
            model_call(prompt,retries-1)
            print("Failed api request, shutting down")
            sys.exit()


def extract_output(html):
    soup = BeautifulSoup(html, 'html.parser')
    translation_tag = soup.find('output')
    if translation_tag:
        return translation_tag.decode_contents()
    # Regex pattern to extract the title and author
    pattern = r'\"title\":\s*\"(.*?)\",\s*\"author\":\s*\"(.*?)\"'
    # Find matches
    matches = re.findall(pattern, html)

    if matches:
        for match in matches:
            title, author = match
            print(f"Title: {title}, Author: {author}")
            return f"Title: {title}, Author: {author}"

    return "Title: none, Author: none"


def predict(lang, passage, mode="unshuffled", prompt_setting="zero-shot"):
    
    demonstrations = {
        "es": {
            "unshuffled": "Hemos de agregar que quemaba tan hondamente el pecho de Hester, que quizá había mayor verdad en el rumor que lo que nuestra moderna incredulidad nos permite aceptar.",
            "shuffled": "lo Hemos quemaba de verdad nos moderna rumor hondamente que que el quizá tan en el mayor había que agregar pecho Hester, que aceptar. de incredulidad permite nuestra"
        },
        "tr": {
            "unshuffled": "Ve Hester'ın göğsünü o kadar derinden yaktı ki, belki de modern şüphemizin kabul etmeye meyilli olmadığı söylentide daha fazla gerçeklik vardı.",
            "shuffled": "ki, yaktı göğsünü gerçeklik vardı. meyilli söylentide belki fazla Hester'ın derinden olmadığı Ve kadar şüphemizin de kabul modern etmeye daha o"
        },
        "vi": {
            "unshuffled": "Và chúng ta tất phải thuật lại rằng nó đã nung đốt thành dấu hằn vào ngực Hester sâu đến nỗi có lẽ trong lời đồn kia có nhiều phần sự thực hơn là đầu óc đa nghi của chúng ta trong thời hiện đại có thể sẵn sàng thừa nhận.",
            "shuffled": "ta phải thuật trong ta trong lẽ thể đại nỗi có nhận. nung đa hằn nghi đốt đồn lời vào dấu sâu Và hơn có sự hiện Hester của có phần thực kia ngực sẵn chúng tất thời nhiều sàng chúng đầu rằng đến là lại thừa đã óc nó thành"
        },
        "en": {
            "unshuffled": "And we must needs say, it seared Hester's bosom so deeply, that perhaps there was more truth in the rumor than our modern incredulity may be inclined to admit.",
            "shuffled": "admit. say, to inclined that the be more must so than it may needs modern we in rumor was deeply, incredulity perhaps our seared bosom there Hester's And truth"
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
        You are provided with a passage in {lang}. Your task is to carefully read and determine which book this passage originates from and who the author is. You must make a guess, even if you are uncertain.
        {demo_prompt}
        Here is the passage:
        <passage>{passage}</passage>

        Use the following format as output:
       <output>"title": "Book name","author": "author name"</output>
    """
#
#    completion = client.chat.completions.create(
#        model="gpt-4o-2024-11-20",
#
#        max_tokens=100,
#
#        messages=[
#            {"role": "user", "content": prompt}
#        ],
#        temperature=0.0
#    )
    completion = model_call(prompt)
    extract = extract_output(completion)
    if extract:
        return extract
    # else:
    #     #print(completion)
    return completion


def direct_probe(csv_file_name, book_title, prompt_setting="zero-shot"):
    #try:
        df = pd.read_csv(csv_file_name)

        for language in df.columns:
            if language not in ['Single_ent','Unnamed: 0']:
                print(f'Running {language}')
                output = []
                for i in range(len(df)):
                    passage = df[language].iloc[i]
                    if "shuffled" in language: 
                        content = predict(language.split('_')[0], passage, "shuffled", prompt_setting)
                    else:
                        content = predict(language.split('_')[0], passage, "unshuffled", prompt_setting)
                    output.append(content)
                    print(f'{i}: {content}')
                index_of_language = df.columns.get_loc(language)
                output_col = pd.Series(output)
                df.insert(index_of_language + 1, f"{language}_results", output_col)

        df.to_csv(f"./direct_probing/fireworks_out/{prompt_setting}/{book_title}_direct_probe_llama405b.csv", index=False, encoding='utf-8')
    #except Exception as e:
     #   print(f'Error: {e}')

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
                'Adventures_of_Huckleberry_Finn',
                    'A_thousand_splendid_suns',
                        'Bride',
                            'Dune',
                                'Just_for_the_Summer',
                                    'Paper_Towns',
                                        'Sense_and_sensibility',
                                            'The_Boy_in_the_Striped_Pyjamas',
                                                'The_Lightning_Thief',
                                                    'The_Ministry_of_Time',
                                                        'The_Picture_of_Dorian_Gray'
                                                        ]

    for title in titles:
       # if title not in skip_list:   
        if title == "1984":         
            print(f'----------------- running {title} -----------------')
            direct_probe(csv_file_name=f"./Prompts/{title}/{title}_filtered.csv", book_title=title, prompt_setting="one-shot") # modify the prompt setting here
            
