from bs4 import BeautifulSoup
from openai import OpenAI
import pandas as pd
import os


client = OpenAI(
    api_key=os.environ['OpenAI_API_KEY'],
)


def extract_output(html):
    soup = BeautifulSoup(html, 'html.parser')
    translation_tag = soup.find('output')
    if translation_tag:
        return translation_tag.decode_contents()
    
    return None


def predict(lang, passage, mode, prompt_setting):
    try:
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

        demo = demonstrations.get(lang, {}).get(mode, "")
        
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

        completion = client.chat.completions.create(
            model="gpt-4o-2024-11-20",

            max_tokens=100,

            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )

        extract = extract_output(completion.choices[0].message.content)
        return extract if extract else completion.choices[0].message.content
    except Exception as e:
        print(f"Error processing passage: {e}")
        return None

def direct_probe(csv_file_name, book_title, prompt_setting):
    try:
        df = pd.read_csv(csv_file_name)

        for language in df.columns:
            if language == 'Single_ent':
                continue
            
            print(f"Processing column: {language}")
            output = []
            mode = "shuffled" if "shuffled" in language.lower() else "unshuffled"
            
            for i in range(len(df)):
                passage = df[language].iloc[i]
                try:
                    base_language = language.split('_')[0]
                    content = predict(base_language, passage, mode, prompt_setting)
                    output.append(content)
                    print(f'Row {i}: {content}')
                except Exception as e:
                    print(f"\nERROR")
                    print(f"Error at row {i} for column {language}: {e}\n")
                    output.append(None)  #NOTE: to maintain DataFrame structure in case of error
            
            index_of_language = df.columns.get_loc(language)
            output_col = pd.Series(output)
            df.insert(index_of_language + 1, f"{language}_results", output_col)

        output_file_name = f"{book_title}_direct_probe_gpt-4o-2024-11-20_{prompt_setting}.csv"
        df.to_csv(output_file_name, index=False, encoding='utf-8')
        print(f"Results saved to {output_file_name}")
    except Exception as e:
        print(f"An error occurred: {e}")


def get_folder_names(directory):
    folder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names

if __name__ == "__main__":
    titles = get_folder_names('/Users/alishasrivastava/BEAM/scripts/Prompts')
    
    for title in titles:
        print(f'----------------- running {title} -----------------')
        direct_probe(csv_file_name=f"/Users/alishasrivastava/BEAM/scripts/Prompts/{title}/{title}_filtered.csv", book_title=title, prompt_setting="zero-shot") # modify the prompt setting here
            