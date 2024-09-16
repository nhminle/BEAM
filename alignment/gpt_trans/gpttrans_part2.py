# part 2 is responsible for checking the status of the batch as well as retreiving the output
from gpttrans_part1 import *
import json
import re
import time

def extract_translation_text(html):
    match = re.search(r'<pas*age>(.*?)</pas*age>', html, re.DOTALL)
    if match:
        return re.sub(r'\n\s*', '', match.group(1))
    return ''

def extract_content_from_jsonl(file, custom_id):
    for line in file:
        data = json.loads(line.strip())
        if data.get('custom_id') == custom_id:
            return data.get('body', {}).get('messages', [])[0].get('content', None)
    return None

def extract_tags(sentence):
    """Extract tags from a sentence."""
    tags = re.findall(r'<(\/?t\d+)>', sentence)
    return set(tags)

def compare_tags(translation, og):
    """Compare tags from two sentences and print whether they are the same or different."""
    tags1 = extract_tags(translation)
    tags2 = extract_tags(og)

    if tags1 != tags2:
        print("The sentences have different tags.")
        print("Tags in the translation but not in the og:", tags1 - tags2)
        print("Tags in the og sentence but not in the translation:", tags2 - tags1)
    
book_title = "A_Tale_of_Two_Cities"
lang = 'vi'

if __name__ == '__main__':
    batch_id = "batch_nyt3reb6mHzMRukpatWb1q3w"
    batch = client.batches.retrieve(batch_id) # get the batch id from the txt
    batch_status = batch.status
    total_time = 0
    while batch_status != "completed":
        print(f"batch is {batch_status}")
        print(f"{batch.request_counts}")
        print(f"total time: {total_time}s")
        time.sleep(30)
        total_time += 30
        batch = client.batches.retrieve(batch_id)
        batch_status = batch.status
    else:
        print(f"batch is {batch_status}")
        print(batch)
        content = client.files.content(batch.output_file_id)
        content.write_to_file(f"/home/nhatminhle_umass_edu/gpt_trans/output_json/{book_title}/{book_title}_{lang}_processed_gpttrans.jsonl")
    path = f'/home/nhatminhle_umass_edu/gpt_trans/output_json/{book_title}'
    with open(f'/home/nhatminhle_umass_edu/gpt_trans/input_json/{book_title}/{book_title}_{lang}_processed.jsonl') as input_jsonl, open(f"/home/nhatminhle_umass_edu/gpt_trans/output_json/{book_title}/{book_title}_{lang}_processed_gpttrans.jsonl", 'r', encoding='utf-8') as translated_jsonl, open(f"/home/nhatminhle_umass_edu/gpt_trans/output_json/{book_title}/{book_title}_{lang}_processed_gpttrans.txt", "w", encoding='utf-8') as translated_txt:
        for line in translated_jsonl:
            data = json.loads(line)

            custom_id = data.get('custom_id')
            content = data.get('response', {}).get('body', {}).get('choices', [])[0].get('message', {}).get('content')

            if custom_id and content:
                translation = extract_translation_text(content)
                if translation != '':
                    compare_tags(translation, f"{extract_translation_text(extract_content_from_jsonl(input_jsonl, custom_id))}")
                    translated_txt.write(f"{translation}\n")
                else:
                    translated_txt.write(f"{extract_translation_text(extract_content_from_jsonl(input_jsonl, custom_id))}\n")