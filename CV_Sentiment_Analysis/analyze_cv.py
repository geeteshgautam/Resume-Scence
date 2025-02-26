# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline

# Load the data
cv_data = pd.read_csv('cv_data.csv',on_bad_lines='skip')

with open('cv_data.csv', 'r') as file:
    for i, line in enumerate(file):
        print(i, line)

# Preview the data
print("Data preview:")
print(cv_data.head())

# Load the sentiment analysis pipeline from Hugging Face transformers
sentiment_analyzer = pipeline('sentiment-analysis')

# Apply sentiment analysis on the job descriptions
# We assume that the 'Job Description' column contains the text you want to analyze
print("Performing sentiment analysis...")
cv_data['sentiment'] = cv_data['Job Description'].apply(lambda x: sentiment_analyzer(x)[0])

# Extract sentiment labels (positive, negative, neutral) and scores from the sentiment column
cv_data['sentiment_label'] = cv_data['sentiment'].apply(lambda x: x['label'])
cv_data['sentiment_score'] = cv_data['sentiment'].apply(lambda x: x['score'])

# Display the results with new sentiment columns
print("\nData with sentiment analysis results:")
print(cv_data[['Job Title', 'Job Description', 'sentiment_label', 'sentiment_score']].head())

# Grouping the data to analyze sentiment distribution by label (e.g., positive, negative)
print("\nGrouping by sentiment labels for average sentiment scores...")
sentiment_distribution = cv_data.groupby('sentiment_label')['sentiment_score'].mean()

# Grouping by job title to get sentiment scores for each job title
print("\nGrouping by Job Titles for sentiment scores...")
job_sentiment = cv_data.groupby('Job Title')['sentiment_score'].mean()

# Visualizing the sentiment distribution with a line graph
plt.figure(figsize=(10,6))
sentiment_distribution.plot(kind='line', color='blue', marker='o')
plt.title('Sentiment Distribution Across Job Descriptions')
plt.xlabel('Sentiment Label')
plt.ylabel('Average Sentiment Score')
plt.grid(True)
plt.show()

# Visualizing sentiment scores for each job title with a line graph
plt.figure(figsize=(12,8))
job_sentiment.plot(kind='line', color='green', marker='x')
plt.title('Sentiment Scores by Job Titles')
plt.xlabel('Job Title')
plt.ylabel('Average Sentiment Score')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nAnalysis complete.")
