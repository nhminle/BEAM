import pandas as pd
import sacrebleu
import seaborn as sns
import matplotlib.pyplot as plt
import evaluate
import os

bertscore = evaluate.load("bertscore")
bleurt = evaluate.load("bleurt")
rouge = evaluate.load("rouge")

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

def main():
    models = ['EuroLLM-9B-Instruct', 'OLMo-7B-0724-Instruct-hf', 'Llama-3.1-70B-Instruct', 'Llama-3.3-70B-Instruct', 'Meta-Llama-3.1-8B-Instruct', 'OLMo-2-1124-13B-Instruct','Qwen2.5-7B-Instruct-1M','OLMo-2-1124-7B-Instruct','Llama-3.1-8B-Instruct-quantized.w4a16','Llama-3.1-8B-Instruct-quantized.w8a16','Llama-3.1-70B-Instruct-quantized.w4a16','Llama-3.1-8B-Instruct-quantized.w8a16'] # add more models here
    prompt_setting = 'one-shot' # one-shot || zero-shot
    # models = ['EuroLLM-9B-Instruct', 'OLMo-7B-0724-Instruct-hf', 'Llama-3.1-70B-Instruct', 'Llama-3.3-70B-Instruct', 'Meta-Llama-3.1-8B-Instruct', 'OLMo-2-1124-13B-Instruct','Qwen2.5-7B-Instruct-1M','OLMo-2-1124-7B-Instruct','Llama-3.1-8B-Instruct-quantized.w4a16','Llama-3.1-8B-Instruct-quantized.w8a16','Llama-3.1-70B-Instruct-quantized.w4a16','Llama-3.1-8B-Instruct-quantized.w8a16'] # add more models here
    # models = ['OLMo-2-1124-7B-Instruct']
    # models = ['Llama-3.1-8B-Instruct-quantized.w4a16','Llama-3.1-8B-Instruct-quantized.w8a16','Llama-3.1-70B-Instruct-quantized.w4a16','Llama-3.1-8B-Instruct-quantized.w8a16']
    for model in models:
        titles = list_csv_files(f'/Users/emir/Downloads/asd/BEAM/results/prefix_probe/{model}/{prompt_setting}/')
        for title in titles:
            if model in title:
                df = pd.read_csv(f'/Users/emir/Downloads/asd/BEAM/results/prefix_probe/{model}/{prompt_setting}/{title}.csv') # adjust path to files 
                df_out = pd.DataFrame()

                available_langs = ["en", "vi", "es", "tr", "mg", "mai", "ty", "tn", "yo", "st"]
                print(f"Available languages for {title}: {available_langs}")

                for lang in available_langs:
                    predictions_col = f'{lang}_second_half'
                    references_col = f'{lang}_results_raw'
                    
                    if predictions_col in df.columns and references_col in df.columns:
                        print(f"Calculating sentence-level metrics for {lang}...")
                        
                        predictions = []
                        references = []

                        # Initialize columns for metrics
                        df_out[f'{lang}_BLEURT'] = None
                        df_out[f'{lang}_ROUGE-L'] = None
                        df_out[f'{lang}_BLEU'] = None
                        df_out[f'{lang}_ChrF++'] = None

                        for index, row in df.iterrows():
                            prediction = row[predictions_col]
                            reference = row[references_col]
                            
                            if pd.notna(prediction) and pd.notna(reference):
                                predictions.append(prediction)
                                references.append(reference)

                                # Compute sentence-level metrics
                                df_out.at[index, f'{lang}_BLEURT'] = calculate_sentence_metrics(lang, prediction, reference, "BLEURT")
                                df_out.at[index, f'{lang}_ROUGE-L'] = calculate_sentence_metrics(lang, prediction, reference, "ROUGE-L")
                                bleu_score = calculate_sentence_metrics(lang, prediction, reference, "BLEU")
                                # print(bleu_score)
                                df_out.at[index, f'{lang}_BLEU'] = bleu_score
                                df_out.at[index, f'{lang}_ChrF++'] = calculate_sentence_metrics(lang, prediction, reference, "ChrF++")

                        # Compute system-level scores for the whole dataset
                        print(f"Calculating system-level scores for {lang}...")
                        system_scores = {
                            f'{lang}_BLEURT': bleurt.compute(predictions=predictions, references=references)["scores"][0],
                            f'{lang}_ROUGE-L': rouge.compute(predictions=predictions, references=references, rouge_types=["rougeL"])["rougeL"],
                            f'{lang}_BLEU': sacrebleu.corpus_bleu(hypotheses=predictions, references=[references]).score,
                            f'{lang}_ChrF++': sacrebleu.corpus_chrf(hypotheses=predictions, references=[references]).score,
                        }

                        # Append system scores as a new row
                        system_scores_row = pd.DataFrame([system_scores])  
                        df_out = pd.concat([df_out, system_scores_row], ignore_index=True)

                # Add a label for the system scores row
                df_out['Index'] = df_out.index
                df_out.loc[df_out.index[-1], 'Index'] = 'System Scores'

                # Save the output CSV
                output_dir = f'/Users/emir/Downloads/asd/BEAM/results/prefix_probe/{model}/eval/csv'
                os.makedirs(output_dir, exist_ok=True)
                df_out.to_csv(f'{output_dir}/{title}.csv', index=False, encoding='utf-8')




# prompt_setting = 'one-shot' # change
# # titles = list_csv_files(f'/{prompt_setting}/prefix_probe') # add path prefix probe output csv
# # mid = len(titles)//2
# # for t in titles[:mid]:
# #     print(f'---------------- running {t} ----------------')
# #     main(t)
    

main()