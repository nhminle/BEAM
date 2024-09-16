# part 1 is responsible for uploading the batch file and creating the batch 
from openai import OpenAI
import os

client = OpenAI(
    api_key=os.environ['OpenAI_API_KEY'],
)

book_title = "A_Tale_of_Two_Cities"
lang = 'vi'
path = f'/home/nhatminhle_umass_edu/gpt_trans/input_json/{book_title}/{book_title}_{lang}_processed.jsonl'

if __name__ == '__main__':
    batch_input_file = client.files.create(
      file=open(path, "rb"),
      purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    batch = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
          "description": f"translate {book_title} into english"
        }
    )

    print(batch)

    with open("/home/nhatminhle_umass_edu/gpt_trans/batch_id_reminder.txt", "a") as f:
      f.write(f"{book_title} {lang} batch id: {batch.id}\n")
