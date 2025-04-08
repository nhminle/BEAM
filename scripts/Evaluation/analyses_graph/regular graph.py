import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as path_effects
import matplotlib.colors as mcolors

# Data from the new image
categories = ['English', 'Translations', 'Cross-lingual']

# Values from the new image
dp_standard = [61.2, 43.5, 30.0]
dp_standard_shuffled = [47.6, 37.7, 23.1]
dp_masked = [33.4, 12.7, 6.2]
dp_masked_shuffled = [18.1, 6.0, 5.3]
dp_no_ne = [37.9, 19.0, 8.3]
dp_no_ne_shuffled = [21.9, 12.6, 5.3]

# Set up the bar positions
x = np.arange(len(categories))
width = 0.125

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Position multipliers for each bar within a group - increased spacing
multipliers = [-3.0, -1.8, -0.6, 0.6, 1.8, 3.0]
positions = []
for i in range(len(categories)):
    for m in multipliers:
        positions.append(i + m * width)

# Define colors and their darker versions
colors = {
    'standard': '#3366CC',
    'standard_shuffled': '#5C9CE6',
    'masked': '#9966CC',
    'masked_shuffled': '#E66C8A',
    'no_ne': '#E67C32',
    'no_ne_shuffled': '#E6CF32'
}

# Function to darken a color
def darken_color(color, factor=0.7):
    """Returns a darker version of the given color"""
    rgb = mcolors.to_rgb(color)
    return tuple(x * factor for x in rgb)

# Create bars with the exact colors from the image
rects1 = ax.bar(positions[::6], dp_standard, width, label='Direct Probe: standard', color=colors['standard'])
rects2 = ax.bar(positions[1::6], dp_standard_shuffled, width, label='Direct Probe: standard - shuffled', color=colors['standard_shuffled'])
rects3 = ax.bar(positions[2::6], dp_masked, width, label='Direct Probe: masked', color=colors['masked'])
rects4 = ax.bar(positions[3::6], dp_masked_shuffled, width, label='Direct Probe: masked - shuffled', color=colors['masked_shuffled'])
rects5 = ax.bar(positions[4::6], dp_no_ne, width, label='Direct Probe: no-NE', color=colors['no_ne'])
rects6 = ax.bar(positions[5::6], dp_no_ne_shuffled, width, label='Direct Probe: no-NE - shuffled', color=colors['no_ne_shuffled'])

# Add value labels ON the bars with matching glow effect
def autolabel(rects, color):
    for rect in rects:
        height = rect.get_height()
        if height < 10:
            y_pos = height/2
        else:
            y_pos = height - min(height * 0.2, 5)
            
        text = ax.text(rect.get_x() + rect.get_width()/2., y_pos,
                f'{height}',
                ha='center', va='center',
                color='white',
                fontsize=10,
                weight='bold')
        # Add glow effect with darker version of bar color
        text.set_path_effects([
            path_effects.Stroke(linewidth=1.5, foreground=darken_color(color), alpha=0.5),
            path_effects.Normal()
        ])

# Apply labels with matching glow colors
autolabel(rects1, colors['standard'])
autolabel(rects2, colors['standard_shuffled'])
autolabel(rects3, colors['masked'])
autolabel(rects4, colors['masked_shuffled'])
autolabel(rects5, colors['no_ne'])
autolabel(rects6, colors['no_ne_shuffled'])

# Customize the graph
ax.set_ylabel('')
ax.set_xticks(x)
ax.set_xticklabels(categories, weight='bold')

# Add y-axis ticks and labels
ax.set_yticks(np.arange(0, 71, 20))
ax.set_yticklabels(['0', '20', '40', '60'], weight='bold')

# Create legend in upper right corner
legend = ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.02),
              ncol=1, fontsize=9, title='One Shot', 
              title_fontsize=10, frameon=True)

plt.setp(legend.get_title(), weight='bold')
for text in legend.get_texts():
    text.set_weight('bold')

# Add horizontal grid lines
ax.yaxis.grid(True, linestyle='-', alpha=0.2, color='lightgray')
ax.set_axisbelow(True)

# Set spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['left'].set_color('lightgray')
ax.spines['bottom'].set_color('lightgray')

# Set y-axis limits
ax.set_ylim(0, 65)

# Add horizontal grid lines at specific intervals
ax.yaxis.set_major_locator(plt.MultipleLocator(20))
ax.tick_params(axis='y', which='major', length=0)
ax.tick_params(axis='x', which='major', length=0)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('dp_comparison_graph.png', dpi=300, bbox_inches='tight', transparent=False, facecolor='white')

# Show the plot
plt.show()
