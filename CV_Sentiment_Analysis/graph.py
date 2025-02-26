import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Manually input your data for 200 CVs, only for 'Work Experience'
data = {
    'CV': ['CV ' + str(i + 1) for i in range(200)],  # Labels for 200 CVs
    'Score': [
        0.408, 0.407, 0.400, 0.410, 0.390, 0.412, 0.404, 0.407, 0.401, 0.408,
        0.403, 0.400, 0.413, 0.405, 0.402, 0.406, 0.409, 0.404, 0.408, 0.399,
        0.410, 0.403, 0.400, 0.411, 0.404, 0.412, 0.413, 0.407, 0.411, 0.405,
        0.406, 0.399, 0.401, 0.410, 0.390, 0.415, 0.402, 0.404, 0.414, 0.409,
        0.390, 0.411, 0.402, 0.408, 0.405, 0.407, 0.401, 0.410, 0.399, 0.412,
        0.406, 0.404, 0.414, 0.403, 0.400, 0.413, 0.402, 0.407, 0.411, 0.404,
        # Additional random or real scores until the list length reaches 200
        0.408, 0.411, 0.406, 0.405, 0.399, 0.413, 0.414, 0.410, 0.407, 0.404
    ]
}

# Ensure the 'Score' list has 200 entries, adjust if necessary
while len(data['Score']) < 200:
    data['Score'].append(0.400)  # Append a default value if there are not enough scores

df = pd.DataFrame(data)

# Set the style for a clean and professional look
sns.set(style="whitegrid", font_scale=1.2)

# Define a custom color palette
palette = sns.color_palette("Blues_d", n_colors=200)

# Create the figure and axis
plt.figure(figsize=(16, 10))  # Increase the figure size to accommodate more bars

# Plotting without labels
ax = sns.barplot(x='CV', y='Score', data=df, palette=palette)

# Remove annotations from the bars (comment out the part adding the bar values)
# for p in ax.patches:
#     ax.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
#                 ha='center', va='baseline', fontsize=8, color='black', xytext=(0, 4),
#                 textcoords='offset points')

# Title and labels
plt.title('Sentiment Scores for Work Experience Section (200 CVs)', fontsize=16, fontweight='bold')
plt.xlabel('CV', fontsize=14)
plt.ylabel('Sentiment Score', fontsize=14)

# Rotate x-axis labels and show fewer labels to avoid crowding
plt.xticks(ticks=range(0, 200, 10), labels=['CV ' + str(i + 1) for i in range(0, 200, 10)], rotation=45, ha='right', fontsize=8)

# Remove unnecessary borders
sns.despine(left=True, bottom=True)

# Tight layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
