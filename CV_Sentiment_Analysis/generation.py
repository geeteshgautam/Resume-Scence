import pandas as pd
import random

# Keywords to analyze
keywords = ["Leadership", "Innovation", "Teamwork", "Communication", "Problem Solving"]

# Generate random data for 200 CVs
data = []
for cv in range(1, 201):
    for keyword in keywords:
        keyword_frequency = random.randint(1, 10)  # Random frequency between 1 and 10
        sentiment_score = round(random.uniform(0.4, 1.0), 2)  # Random sentiment score between 0.4 and 1.0
        data.append([f"CV {cv}", keyword, keyword_frequency, sentiment_score])

# Create a DataFrame
df = pd.DataFrame(data, columns=["CV", "Keyword", "Keyword Frequency", "Sentiment Score"])

# Save the dataset to a CSV file
df.to_csv("cv_keyword_sentiment_data.csv", index=False)

# Preview the first few rows
print(df.head())
