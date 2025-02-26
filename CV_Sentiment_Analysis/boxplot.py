# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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

# Extract sentiment labels (positive, negative, neutral) and scores from the sentiment column
cv_data['sentiment_label'] = cv_data['sentiment'].apply(lambda x: x['label'])
cv_data['sentiment_score'] = cv_data['sentiment'].apply(lambda x: x['score'])

# Display the results with new sentiment columns
print("\nData with sentiment analysis results:")
print(cv_data[['Job Title', 'Job Description', 'sentiment_label', 'sentiment_score']].head())

# To balance the data, let's resample the dataset so each job title has a similar number of entries
# Find the minimum count of job titles
min_count = cv_data['Job Title'].value_counts().min()

# Resample the data to ensure equal number of job titles
balanced_data = cv_data.groupby('Job Title').apply(lambda x: x.sample(min_count)).reset_index(drop=True)

# Display the balanced data
print("\nBalanced data preview:")
print(balanced_data['Job Title'].value_counts())

# Define a custom color palette with different colors for each job title
palette = sns.color_palette("husl", len(balanced_data['Job Title'].unique()))  # "husl" gives a wide range of colors

# Boxplot for sentiment scores by job title on balanced data
plt.figure(figsize=(14, 8))
sns.boxplot(data=balanced_data, x='Job Title', y='sentiment_score',
            palette=palette,  # Assign different colors to each box
            linewidth=2.5,  # Increase thickness of box edges
            boxprops=dict(edgecolor='black', linewidth=2.5),  # Box properties with black edges
            whiskerprops=dict(color='black', linewidth=2.5),  # Whisker properties
            capprops=dict(color='black', linewidth=2.5),  # Caps on whiskers
            medianprops=dict(color='brown', linewidth=30))  # Median line properties

plt.title('Boxplot of Sentiment Scores by Job Titles (Balanced Data)', fontsize=14)
plt.xlabel('Job Title', fontsize=12)
plt.ylabel('Sentiment Score', fontsize=12)
plt.xticks(rotation=90, fontsize=10)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nAnalysis complete.")
