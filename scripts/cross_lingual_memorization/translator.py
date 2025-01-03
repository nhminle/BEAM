import requests
import pandas as pd
import time

# Azure Translator API credentials
endpoint = "https://api.cognitive.microsofttranslator.com/"
subscription_key = "" #TODO: Enter Key Here
region = "westus" #TODO: Enter Region Here

# Test languages
languages = {
    "st": "Sesotho",
    "yo": "Yorùbá",
    "my": "Burmese",
    "am": "Amharic",
    "ha": "Hausa",
    "rw": "Kinyarwanda",
    "xh": "Xhosa",
    "ps": "Pashto"
}

#translate a batch of text with exponential backoff for rate limit handling.
def translate_batch_with_backoff(texts, to_lang, max_retries=5, initial_delay=5):
    url = f"{endpoint}translate?api-version=3.0&to={to_lang}"
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Ocp-Apim-Subscription-Region': region,
        'Content-Type': 'application/json'
    }
    body = [{"text": text} for text in texts]
    delay = initial_delay #initial delay in seconds

    for attempt in range(max_retries):
        response = requests.post(url, headers=headers, json=body)
        if response.status_code == 429:  # Too many requests
            retry_after = int(response.headers.get("Retry-After", delay))
            print(f"Rate limit hit. Retrying in {retry_after} seconds...")
            time.sleep(retry_after)
            delay *= 2  # exponential backoff looks ok!
        else:
            response.raise_for_status()
            return [item['translations'][0]['text'] for item in response.json()]

    raise Exception("Max retries exceeded for translation request.")

# estimate cost by char 
def estimate_cost(df, languages, cost_per_million=10, free_tier_limit=2_000_000): #TODO edit free tier limit as you go 
    total_characters = df['En'].str.len().sum()
    total_translation_characters = total_characters * len(languages)

    if total_translation_characters <= free_tier_limit:
        print("Estimated cost: $0 (within free tier)")
    else:
        paid_characters = total_translation_characters - free_tier_limit
        estimated_cost = (paid_characters / 1_000_000) * cost_per_million
        print(f"Estimated cost: ${estimated_cost:.2f}")
    return total_translation_characters

if __name__ == "__main__":
    input_csv = "/Users/alishasrivastava/Desktop/Test/testset/CLM TestSet - Pride and Prejudice.csv" #TODO enter file path 
    output_csv = "CLM_Pride_and_Prejudice.csv" #TODO enter title

    df = pd.read_csv(input_csv)
    if 'En' not in df.columns:
        print("Error: The input CSV does not contain a column named 'En'.")
        exit(1)

    print("Estimating cost...")
    estimate_cost(df, languages)

    # batching translations with delays in between batches
    batch_size = 10
    for lang_code, lang_name in languages.items():
        print(f"Translating to {lang_name}...")
        translations = []
        for i in range(0, len(df), batch_size):
            batch = df['En'][i:i + batch_size].tolist()
            translations.extend(translate_batch_with_backoff(batch, lang_code))
            time.sleep(5)  # delay between batches
        df[lang_name] = translations
        print(f"Completed translations for {lang_name}. Waiting before next language...")
        time.sleep(10)  # delay in between languages

    df.to_csv(output_csv, index=False)
    print(f"Translation complete. Saved to {output_csv}")
