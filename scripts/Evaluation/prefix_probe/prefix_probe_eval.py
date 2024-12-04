import pandas as pd
import sacrebleu
import seaborn as sns
import matplotlib.pyplot as plt
import evaluate
import os

bertscore = evaluate.load("bertscore")
bleurt = evaluate.load("bleurt")
rouge = evaluate.load("rouge")

def calculate_sentence_metrics(lang, prediction: str, reference: str, metric_name: str) -> float:
    if metric_name == "BERTScore":
        results = bertscore.compute(predictions=[prediction], references=[reference], lang=lang)
        return results["f1"][0]  # Use F1 score
    elif metric_name == "BLEURT":
        results = bleurt.compute(predictions=[prediction], references=[reference])
        return results["scores"][0]  
    elif metric_name == "ROUGE-L":
        results = rouge.compute(predictions=[prediction], references=[reference], rouge_types=["rougeL"])
        return results["rougeL"]
    elif metric_name == "BLEU":
        results = sacrebleu.sentence_bleu(hypothesis=prediction, references=[reference])
        return results.score
    elif metric_name == "ChrF++":
        results = sacrebleu.sentence_chrf(hypothesis=prediction, references=[reference])
        return results.score
    else:
        raise ValueError(f"Unknown metric: {metric_name}")

def main(title):
    df = pd.read_csv(f'/home/nhatminhle_umass_edu/Tasks/out/prefix_probe copy/{title}.csv')
    df_out = pd.DataFrame()

    # available_langs = [col.split('_')[0] for col in df.columns if col.endswith('_prompts_shuffled_results')]
    available_langs = ['en', 'vi', 'es', 'tr']
    print(f"Available languages for {title}: {available_langs}")

    for lang in available_langs:
        predictions_col = f'{lang}_second_half'
        references_col = f'{lang}_results'
        
        if predictions_col in df.columns and references_col in df.columns:
            print(f"Calculating sentence-level metrics for {lang}...")
            
            # df_out[f'{lang}_BERTScore'] = None
            df_out[f'{lang}_BLEURT'] = None
            df_out[f'{lang}_ROUGE-L'] = None
            df_out[f'{lang}_BLEU'] = None
            df_out[f'{lang}_ChrF++'] = None

            for index, row in df.iterrows():
                prediction = row[predictions_col]
                reference = row[references_col]
                
                if pd.notna(prediction) and pd.notna(reference):
                    # df_out.at[index, f'{lang}_BERTScore'] = calculate_sentence_metrics(lang, prediction, reference, "BERTScore")
                    bleurt_score = calculate_sentence_metrics(lang, prediction, reference, "BLEURT")
                    # print(bleurt_score)
                    df_out.at[index, f'{lang}_BLEURT'] = bleurt_score
                    print(bleurt_score)
                    df_out.at[index, f'{lang}_ROUGE-L'] = calculate_sentence_metrics(lang, prediction, reference, "ROUGE-L")
                    df_out.at[index, f'{lang}_BLEU'] = calculate_sentence_metrics(lang, prediction, reference, "BLEU")
                    df_out.at[index, f'{lang}_ChrF++'] = calculate_sentence_metrics(lang, prediction, reference, "ChrF++")

    models = ['OLMo-7B-0724-Instruct-hf', 'Llama-3.1-70B-Instruct', 'Meta-Llama-3.1-8B-Instruct']
    for model in models:
        if model in title:
            df_out.to_csv(f'/home/nhatminhle_umass_edu/Evaluation/prefix_probe/eval/csv/{model}/{title}.csv', index=False, encoding='utf-8')


def list_csv_files(directory):
    try:
        files = os.listdir(directory)
        
        csv_files = [file.replace('.csv', '') for file in files if file.endswith('.csv')]
        
        return csv_files
    except FileNotFoundError:
        print(f"Error: The directory '{directory}' does not exist.")
        return []
    except PermissionError:
        print(f"Error: Permission denied for accessing '{directory}'.")
        return []

titles = list_csv_files('/home/nhatminhle_umass_edu/Tasks/out/prefix_probe copy')

# for t in titles:
#     print(f'---------------- running {t} ----------------')
    # main(t)
    
main('Harry_Potter_and_the_Deathly_Hallows_prefix_probe_OLMo-7B-0724-Instruct-hf') #  + Harry_Potter_and_the_Deathly_Hallows_prefix_probe_OLMo-7B-0724-Instruct-hf