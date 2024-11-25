import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Provided results dictionary
results = {
    'en_results_title_match': 85.0, 'en_results_author_match': 85.0, 'en_results_both_match': 85.0,
    'es_results_title_match': 85.0, 'es_results_author_match': 86.66666666666667, 'es_results_both_match': 85.0,
    'tr_results_title_match': 80.0, 'tr_results_author_match': 80.0, 'tr_results_both_match': 80.0,
    'vi_results_title_match': 73.33333333333333, 'vi_results_author_match': 75.0, 'vi_results_both_match': 73.33333333333333
}

# Extracting only 'both_match' results
languages = ['en', 'es', 'tr', 'vi']
both_match_values = {lang: results[f"{lang}_results_both_match"] for lang in languages}

# Creating a DataFrame for the heatmap
df = pd.DataFrame([both_match_values], index=['Both Match'])

# Plotting the heatmap with languages on the x-axis
plt.figure(figsize=(8, 2))
sns.heatmap(df, annot=True, cmap="viridis", cbar_kws={'label': 'Accuracy (%)'}, fmt=".1f", xticklabels=True, yticklabels=True)
plt.title('Both Match Accuracy by Language')
plt.xlabel('Language')
plt.ylabel('Category')
plt.tight_layout()
plt.show()
