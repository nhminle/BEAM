import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

#Setup Data
# stack CSV files from a folder into one DataFrame for aggregate analysis
def load_and_stack_csvs(path2folder):
    all = [os.path.join(path2folder, f) for f in os.listdir(folder_path) if f.endswith('.csv')]
    df_list = []
    for file in all:
        df = pd.read_csv(file)
        df_list.append(df)
    stacked_df = pd.concat(df_list, ignore_index=True)
    return stacked_df

path2folder = "/Users/alishasrivastava/BEAM-scripts/BEAM/scripts/name_cloze_task/Evaluation/eval/csv/OLMo-7B-0724-Instruct-hf" #TODO: add the path to the folder with all the data, ensure 2024 books have been removed
superdf = load_and_stack_csvs(path2folder)

# reshaping df for counts
language_columns = ["en_correct", "es_correct", "tr_correct", "vi_correct"]
melted_data = []
for col in language_columns:
    language = col.split("_")[0]
    correct_counts = superdf[col].value_counts()
    melted_data.append({"language": language, "result": "correct", "count": correct_counts.get("correct", 0)})
    melted_data.append({"language": language, "result": "incorrect", "count": correct_counts.get("incorrect", 0)})

visualization_df = pd.DataFrame(melted_data)

#just plotting here
sns.set(style="whitegrid") #set depreceated? i guess set_theme
flare_palette = sns.color_palette("flare", n_colors=2)
plt.figure(figsize=(8, 6))
sns.barplot(data=visualization_df, x="language", y="count", hue="result", palette=flare_palette)

plt.title("OLMo-7B-0724-Instruct-hf of Correct/Incorrect Counts by Language") #TODO: change name by model
plt.xlabel("Language")
plt.ylabel("Count")
plt.legend(title="Result", loc="upper right")
plt.tight_layout()


plt.savefig('/Users/alishasrivastava/BEAM-scripts/BEAM/scripts/name_cloze_task/Evaluation/eval/OLMo-7B-0724-Instruct-hf_overall_nct') #TODO: change path
