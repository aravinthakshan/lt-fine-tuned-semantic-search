import pandas as pd

def extract_code_parts(location, code_column):
    # Load the Excel file
    df = pd.read_excel(location)
    
    # Extract the first four and last four characters of the code
    df['FirstFour'] = df[code_column].str[:4]
    
    # Return the modified DataFrame
    return df

# Example usage
location = 'datasets/features-mid.xlsx'  # Change this to the path of your Excel file
code_column = 'CODE'  # Change this to the name of your code column

df_result = extract_code_parts(location, code_column)
print(df_result)

# Optionally, save the result to a new Excel file
df_result.to_excel('datasets/modified_sample5.xlsx', index=False)
