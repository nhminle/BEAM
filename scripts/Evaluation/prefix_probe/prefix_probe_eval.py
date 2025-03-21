import pandas as pd
import sacrebleu
import seaborn as sns
import matplotlib.pyplot as plt
import evaluate
import os
import argparse 

# Load metrics
bertscore = evaluate.load("bertscore")
# bleurt = evaluate.load("bleurt")
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
    # elif metric_name == "BLEURT":
    #     results = bleurt.compute(predictions=[prediction], references=[reference])
    #     return results["scores"][0]  
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
    # print(f"Evaluating model: {model}")
    prompt_setting = 'one-shot'  # one-shot || zero-shot
    models = ['gpt-4o-2024-11-20'] # add more models here
    # List CSV files in the results folder
  # Process each CSV file whose title contains the model string
    for model in models:
        results_dir = f'/home/ekorukluoglu_umass_edu/beam2/BEAM/results/prefix_probe/{model}/{prompt_setting}/'
        titles = list_csv_files(results_dir)
        print("CSV file titles found:", titles)
        for title in titles:
            if model in title:
                print(f"\nProcessing file: {title}.csv")
                csv_path = os.path.join(results_dir, f"{title}.csv")
                try:
                    df = pd.read_csv(csv_path)
                except Exception as e:
                    print(f"Error reading {csv_path}: {e}")
                    continue

                # Debug: print columns and first few rows
                print("Columns in CSV:", df.columns.tolist())
                print("First few rows:\n", df.head())

                df_out = pd.DataFrame()  # will hold per-row metrics and a system-level scores row

                available_langs = ["en", "vi", "es", "tr", "mg", "mai", "ty", "tn", "yo", "st"]
                print(f"Available languages for file {title}: {available_langs}")

                for lang in available_langs:
                    predictions_col = f'{lang}_Completion'
                    references_col = f'{lang}_second_half'
                    print(f"Checking for columns: {predictions_col} and {references_col}")

                    if predictions_col in df.columns and references_col in df.columns:
                        print(f"Found columns for {lang}. Processing rows...")
                        predictions = []
                        references = []
                        # For this language, we create new columns in a temporary DataFrame
                        temp_df = pd.DataFrame(index=df.index)
                        temp_df[f'{lang}_ROUGE-L'] = None
                        temp_df[f'{lang}_BLEU'] = None
                        temp_df[f'{lang}_ChrF++'] = None

                        # Process each row
                        processed_rows = 0
                        for index, row in df.iterrows():
                            prediction = row[predictions_col]
                            reference = row[references_col]
                            if pd.notna(prediction) and pd.notna(reference):
                                predictions.append(prediction)
                                references.append(reference)
                                try:
                                    temp_df.at[index, f'{lang}_ROUGE-L'] = calculate_sentence_metrics(lang, prediction, reference, "ROUGE-L")
                                    bleu_score = calculate_sentence_metrics(lang, prediction, reference, "BLEU")
                                    temp_df.at[index, f'{lang}_BLEU'] = bleu_score
                                    temp_df.at[index, f'{lang}_ChrF++'] = calculate_sentence_metrics(lang, prediction, reference, "ChrF++")
                                except Exception as e:
                                    print(f"Error at row {index} for language {lang}: {e}")
                                processed_rows += 1

                        print(f"Processed {processed_rows} valid rows for language {lang}.")
                        
                        # If any rows were processed, compute system-level scores
                        if predictions and references:
                            try:
                                system_scores = {
                                    f'{lang}_ROUGE-L': rouge.compute(predictions=predictions, references=references, rouge_types=["rougeL"])["rougeL"],
                                    f'{lang}_BLEU': sacrebleu.corpus_bleu(hypotheses=predictions, references=[references]).score,
                                    f'{lang}_ChrF++': sacrebleu.corpus_chrf(hypotheses=predictions, references=[references]).score,
                                }
                            except Exception as e:
                                print(f"Error computing system scores for {lang}: {e}")
                                system_scores = {}
                            # Append the system scores as a new row at the end of temp_df
                            system_scores_df = pd.DataFrame([system_scores])
                            temp_df = pd.concat([temp_df, system_scores_df], ignore_index=True)
                        else:
                            print(f"No valid rows for language {lang}; skipping system-level score calculation.")

                        # Merge the metrics for this language into df_out
                        # If df_out is empty, use temp_df; otherwise, merge column-wise.
                        if df_out.empty:
                            df_out = temp_df.copy()
                        else:
                            df_out = pd.concat([df_out, temp_df], axis=1)
                    else:
                        print(f"Columns for {lang} not found in CSV.")
                
                # Add a label for the system scores row if df_out is not empty
                if not df_out.empty:
                    # Create or convert the "Index" column to object type so it can hold strings
                    df_out['Index'] = df_out.index.astype(object)
                    df_out.loc[df_out.index[-1], 'Index'] = 'System Scores'
                else:
                    print("Warning: No data processed in df_out; nothing to label.")
                
                # Save the output CSV if df_out is not empty
                output_dir = f'/home/ekorukluoglu_umass_edu/beam2/BEAM/results/prefix_probe/{model}/eval/csv'
                os.makedirs(output_dir, exist_ok=True)
                output_csv = os.path.join(output_dir, f"{title}.csv")
                if not df_out.empty:
                    df_out.to_csv(output_csv, index=False, encoding='utf-8')
                    print(f"Saved output CSV to {output_csv}")
                else:
                    print(f"No output generated for file {title}.")
            else:
                print(f"Skipping file {title}.csv because it does not match model {model}.")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Prefix probe evaluation script.")
    # parser.add_argument("--model", required=True, help="Name of the model to evaluate")
    # args = parser.parse_args()
    # print("Running eval on model:", args.model)
    # main(args.model)
    main()