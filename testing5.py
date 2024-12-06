import pandas as pd

math = pd.read_csv('/kaggle/input/aime-problem-set-1983-2024/AIME_Dataset_1983_2024.csv')

# Create empty list to store filtered problems
filtered_problems = []

# Iterate through the dataframe rows
for _, row in math.iterrows():
    if row['Problem Number'] > 5:
        filtered_problems.append((row['Question'], row['Answer']))

# Create new dataframe from the filtered problems
result_df = pd.DataFrame(filtered_problems, columns=['Question', 'Answer'])

# Save the dataframe to a CSV file
result_df.to_csv('filtered_problems.csv', index=False)
