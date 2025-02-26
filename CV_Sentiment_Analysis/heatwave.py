# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from transformers import pipeline

# Load the data
cv_data = pd.read_csv('cv_data.csv', on_bad_lines='skip')

# Preview the data
print("Data preview:")
print(cv_data.head())

# Load the sentiment analysis pipeline from Hugging Face transformers
sentiment_analyzer = pipeline('sentiment-analysis')

# Apply sentiment analysis on the job descriptions
print("Performing sentiment analysis...")
cv_data['sentiment'] = cv_data['Job Description'].apply(lambda x: sentiment_analyzer(x)[0])

# Extract sentiment labels and scores
cv_data['sentiment_label'] = cv_data['sentiment'].apply(lambda x: x['label'])
cv_data['sentiment_score'] = cv_data['sentiment'].apply(lambda x: x['score'])

# Extract experience years from job descriptions
def extract_experience(description):
    # Regex to find years of experience
    match = re.search(r'(\d+)\s*years?', description, re.IGNORECASE)
    return int(match.group(1)) if match else 0

cv_data['Experience Years'] = cv_data['Job Description'].apply(extract_experience)

# Create a DataFrame for heatmap
heatmap_data = cv_data.groupby('Experience Years')['sentiment_score'].mean().reset_index()

# Create a pivot table for the heatmap
heatmap_data_pivot = heatmap_data.pivot_table(index='Experience Years', values='sentiment_score')

# Create a heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(heatmap_data_pivot, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Heatmap of Average Sentiment Scores by Experience Years')
plt.xlabel('Average Sentiment Score')
plt.ylabel('Experience Years')
plt.show()
