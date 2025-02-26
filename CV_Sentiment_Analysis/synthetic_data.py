import pandas as pd
import numpy as np

# Define the base data
data = {
    "CV": ["CV 1"] * 5,
    "Keyword": ["Leadership", "Innovation", "Teamwork", "Communication", "Problem Solving"],
    "Keyword Frequency": [4, 8, 9, 5, 2],
    "Sentiment Score": [0.54, 0.59, 1.00, 0.87, 0.98]
}

# Create a DataFrame from the base data
df = pd.DataFrame(data)

# Function to generate synthetic data
def generate_synthetic_data(df, num_datasets):
    all_dfs = []
    for i in range(num_datasets):
        # Create a copy of the base data
        temp_df = df.copy()
        # Update CV number
        temp_df['CV'] = f"CV {i + 1}"
        # Add some random noise to Frequency and Sentiment Score
        temp_df['Keyword Frequency'] = temp_df['Keyword Frequency'] + np.random.randint(-2, 3, size=len(temp_df))
        temp_df['Sentiment Score'] = temp_df['Sentiment Score'] + np.random.uniform(-0.05, 0.05, size=len(temp_df))
        # Clip values to valid ranges
        temp_df['Keyword Frequency'] = temp_df['Keyword Frequency'].clip(lower=0)
        temp_df['Sentiment Score'] = temp_df['Sentiment Score'].clip(lower=0, upper=1)
        all_dfs.append(temp_df)
    
    # Concatenate all DataFrames into one
    return pd.concat(all_dfs, ignore_index=True)

# Generate 200 data sets
synthetic_data = generate_synthetic_data(df, 200)

# Save to a CSV file
synthetic_data.to_csv('synthetic_cv_data.csv', index=False)

print("200 synthetic data sets have been generated and saved to 'synthetic_cv_data.csv'.")
