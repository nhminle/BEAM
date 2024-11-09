import stanza
import re
import os


def extract_sentences(input_file, output_file, lang):
    # Initialize Stanza
    stanza.download(lang)  # Download English model if not already downloaded
    nlp = stanza.Pipeline(lang, processors='tokenize', use_gpu=True)
    # Read input text
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    text = re.sub(r"mr\.", "mr", text, flags=re.IGNORECASE)
    text = re.sub(r"mrs\.", "ms", text, flags=re.IGNORECASE)
    text = re.sub(r"dr\.", "dr", text, flags=re.IGNORECASE)
    text = re.sub(r'\.(\w)', r'. \1', text)
    text = re.sub(r'\*+\s*\*+\s*\*+', ' ', text)

    # Process text to extract sentences
    doc = nlp(text)
    offset = 0
    for i, sentence in enumerate(doc.sentences):
        text = text[:offset + sentence.tokens[0].start_char] + f'<t{i}>' + text[offset + sentence.tokens[0].start_char:]
        offset += len(f'<t{i}>')
        text = text[:offset + sentence.tokens[-1].end_char] + f'</t{i}>' + text[offset + sentence.tokens[-1].end_char:]
        offset += len(f'</t{i}>')

    # Write sentences as a single paragraph with \n between them
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)


if __name__ == "__main__":
    language = '' # choose lang
    input_dir = f''
    output_dir = f''
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace(".txt", "_processed.txt"))

            print(f"Starting {input_path}")
            extract_sentences(input_path, output_path, language)
            print(f"Sentences extracted from {input_path} and written to {output_path} as a single paragraph.")
