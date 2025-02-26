# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
import plotly.graph_objects as go

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

# Group data for Sankey diagram
# Assuming you have a 'Qualifications' column
qualification_job_counts = cv_data.groupby(['Qualifications', 'Job Title']).size().reset_index(name='count')

# Create unique labels for nodes
labels = list(pd.concat([qualification_job_counts['Qualifications'], qualification_job_counts['Job Title']]).unique())

# Create source and target indices
qualification_indices = qualification_job_counts['Qualifications'].apply(lambda x: labels.index(x)).tolist()
job_title_indices = qualification_job_counts['Job Title'].apply(lambda x: labels.index(x)).tolist()

# Create the Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=labels,
    ),
    link=dict(
        source=qualification_indices,  # Indices correspond to qualifications
        target=job_title_indices,      # Indices correspond to job titles
        value=qualification_job_counts['count'],  # The counts of each qualification-job title pair
    ))])

fig.update_layout(title_text="Sankey Diagram of Qualifications and Job Titles", font_size=10)
fig.show()

# Visualizing sentiment distribution (already in your code)
# ...

print("\nAnalysis complete.")
