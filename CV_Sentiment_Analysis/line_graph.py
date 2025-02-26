import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the synthetic data from CSV
data = pd.read_csv('synthetic_cv_data.csv')

# Set the style of the seaborn plot
sns.set(style="whitegrid")

# Create a line plot for Keyword Frequency
plt.figure(figsize=(14, 7))
sns.lineplot(data=data, x='CV', y='Keyword Frequency', marker='o', color='blue', label='Keyword Frequency')
plt.xticks(ticks=[], labels=[])  # Hide x-axis tick labels
plt.title('Keyword Frequency Over Different CVs')
plt.xlabel('CV')
plt.ylabel('Keyword Frequency')
plt.legend()
plt.tight_layout()
plt.show()

# Create a line plot for Sentiment Score
plt.figure(figsize=(14, 7))
sns.lineplot(data=data, x='CV', y='Sentiment Score', marker='o', color='green', label='Sentiment Score')
plt.xticks(ticks=[], labels=[])  # Hide x-axis tick labels
plt.title('Sentiment Score Over Different CVs')
plt.xlabel('CV')
plt.ylabel('Sentiment Score')
plt.legend()
plt.tight_layout()
plt.show()
