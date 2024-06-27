import pandas as pd
import numpy as np

def shuffle_csv_in_place(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Shuffle the DataFrame rows
    df = df.sample(frac=1).reset_index(drop=True)
    
    # Write the shuffled DataFrame back to the same file
    df.to_csv(file_path2, index=False)
    
# Example usage
file_path = 'datasets/features-output-final-cleaned.csv'  # Replace with your CSV file path
file_path2 = 'datasets/shuffled-features-output-final-cleaned.csv'
shuffle_csv_in_place(file_path)
