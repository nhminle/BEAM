import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Llama-3.1-8B-Instruct\nw4a16', 'Llama-3.1-8B-Instruct\nw8a16', 'Llama-3.1-8B-Instruct']

# Data reorganized by language group with corrected values
english = [52.0, 54.0, 54.1]
translated = [29.0, 37.4, 36.1]
crosslingual = [16.2, 22.8, 22.4]
english_shuffled = [40.4, 42.7, 41.7]
translated_shuffled = [26.8, 36.1, 35.6]
crosslingual_shuffled = [12.2, 20.1, 19.7]

# Set the positions of the bars
x = np.arange(len(models)) * 1.5  # Increased spacing between model groups
width = 0.13  # Width of bars

# Create the figure and axis
fig, ax = plt.subplots(figsize=(15, 6))

# Set font weight to bold for all text
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# Create bars with spacing between them
rects1 = ax.bar(x - 3.0*width, english, width, label='English', color='#4285F4')
rects2 = ax.bar(x - 1.8*width, english_shuffled, width, label='English Shuffled', color='#42A5F5')
rects3 = ax.bar(x - 0.6*width, translated, width, label='Translated', color='#9575CD')
rects4 = ax.bar(x + 0.6*width, translated_shuffled, width, label='Translations Shuffled', color='#EC407A')
rects5 = ax.bar(x + 1.8*width, crosslingual, width, label='Cross-lingual', color='#FF7043')
rects6 = ax.bar(x + 3.0*width, crosslingual_shuffled, width, label='Cross-lingual Shuffled', color='#FFA726')

# Customize the graph
ax.set_ylabel('')  # Remove y-axis label
ax.set_xticks(x)
ax.set_xticklabels(models, weight='bold')
ax.legend(title='Language Groups', title_fontsize=10, fontsize=10)

# Make legend title bold
legend = ax.get_legend()
legend.get_title().set_fontweight('bold')

# Make y-axis labels bold
ax.tick_params(axis='both', which='major', labelsize=10)
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

# Remove the box around the plot
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Set bottom and left spines to match grid style
ax.spines['bottom'].set_color('gray')
ax.spines['bottom'].set_alpha(0.2)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_color('gray')
ax.spines['left'].set_alpha(0.2)
ax.spines['left'].set_linewidth(1)

# Add value labels on the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        # For very small values, position text differently
        if height < 2:
            y_pos = height + 0.3  # Position above the bar for small values
            va = 'bottom'
            color = 'black'
        else:
            y_pos = height - 0.7  # Adjusted position for larger font
            va = 'top'
            color = 'white'
            
        ax.text(rect.get_x() + rect.get_width()/2., y_pos,
                f'{height}',
                ha='center', va=va,
                color=color,
                weight='bold',
                fontsize=10)  # Increased from 8 to 10

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6)

# Add horizontal grid lines only
ax.yaxis.grid(True, linestyle='-', alpha=0.2, color='gray')
ax.set_axisbelow(True)

# Set y-axis limits to prevent text cutoff
ymax = max([max(english), max(english_shuffled), max(translated),
            max(translated_shuffled), max(crosslingual), max(crosslingual_shuffled)])
ax.set_ylim(0, ymax * 1.15)  # Increased padding at the top to 15%

# Add horizontal grid lines at specific intervals
ax.yaxis.set_major_locator(plt.MultipleLocator(20))
ax.tick_params(axis='y', which='major', length=0)  # Remove y-axis tick marks
ax.tick_params(axis='x', which='major', length=0)  # Remove x-axis tick marks

# Adjust layout
plt.tight_layout()

# Save the plot as PNG with white background
plt.savefig('quantized_graph_by_model.png', dpi=300, bbox_inches='tight', transparent=False, facecolor='white')

# Show the plot
plt.show()
