import pandas as pd

# Read the CSV file
df = pd.read_excel('datasets/training-dataset-finalized.xlsx')

# Function to extract the middle part of the code
def extract_middle_part(code):
    return code[4:-4]

# Apply the function to the relevant column and create a new column
df['MiddlePart'] = df['CODE'].apply(extract_middle_part)

# Save the updated DataFrame back to a new CSV file
df.to_excel('datasets/features-mid.xlsx', index=False)
