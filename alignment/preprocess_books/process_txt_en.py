import stanza
import re
import os

# Initialize Stanza
stanza.download('en')  # Download English model if not already downloaded
nlp = stanza.Pipeline('en', processors='tokenize', use_gpu=True)

def extract_sentences(input_file, output_file):
    # Read input text
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    text = re.sub(r"mr\.", "mr", text, flags=re.IGNORECASE)
    text = re.sub(r"mrs\.", "ms", text, flags=re.IGNORECASE)
    text = re.sub(r"dr\.", "dr", text, flags=re.IGNORECASE)
    text = re.sub(r'\.(\w)', r'. \1', text)
    text = re.sub(r'\*+\s*\*+\s*\*+', ' ', text)

    # Process text to extract sentencesy
    doc = nlp(text)
    sentences = []
    for sentence in doc.sentences:
        text= sentence.text.strip()
        sentences.append(text)

    # Write sentences as a single paragraph with \n between them
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(sentences))

if __name__ == "__main__":
    input_dir = f''
    output_dir = f''
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename.replace(".txt", "_processed.txt"))

            print(f"Starting {input_path}")
            extract_sentences(input_path, output_path)
            print(f"Sentences extracted from {input_path} and written to {output_path} as a single paragraph.")
