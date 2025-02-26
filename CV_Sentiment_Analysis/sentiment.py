# Importing necessary libraries
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# Load your data into a Pandas DataFrame
data = {
    "Job Title": [
        "Data Scientist", "Marketing Manager", "Public Health Researcher", "UX Researcher", 
        "Machine Learning Engineer", "Travel Consultant", "Product Designer", "Copywriter", 
        "Pharmaceutical Sales Representative", "Talent Acquisition Specialist"
    ],
    "Job Description": [
        "Data Analysis, Machine Learning, Programming", 
        "Marketing Strategy, Campaign Management, Budgeting", 
        "Data Collection, Analysis, Reporting", 
        "User Interviews, Usability Testing, Data Analysis", 
        "Deep Learning, Python, Model Development", 
        "Customer Service, Itinerary Planning, Destination Knowledge", 
        "3D Modeling, Prototyping, User Research", 
        "Creative Writing, Brand Messaging, SEO", 
        "Sales Strategy, Client Relations, Product Knowledge", 
        "Recruitment, Interviewing, Employer Branding"
    ]
}

df = pd.DataFrame(data)

# Initialize the sentiment analysis pipeline
sentiment_analyzer = pipeline("sentiment-analysis")

# Apply sentiment analysis to each job description
df['Sentiment'] = df['Job Description'].apply(lambda description: sentiment_analyzer(description)[0]['label'])

# Map sentiment labels to numerical values for graphing
df['Sentiment Score'] = df['Sentiment'].map({'POSITIVE': 1, 'NEGATIVE': -1, 'NEUTRAL': 0})

# Plotting the sentiment score in a line graph
plt.figure(figsize=(10, 6))
plt.plot(df['Job Title'], df['Sentiment Score'], marker='o', linestyle='-', color='b')
plt.title('Sentiment Analysis of Job Descriptions')
plt.xlabel('Job Title')
plt.ylabel('Sentiment Score')
plt.xticks(rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()

# Show the plot
plt.show()
