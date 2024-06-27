import csv
import json

def csv_to_jsonl(csv_file_path, jsonl_file_path):
    # Read the CSV file
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        # Write each row as a JSON object to the JSONL file
        with open(jsonl_file_path, mode='w', encoding='utf-8') as jsonl_file:
            for row in csv_reader:
                jsonl_file.write(json.dumps(row) + '\n')

# Example usage
csv_file_path = 'datasets-usefull/training-data-categories.csv'
jsonl_file_path = 'output-file.jsonl'
csv_to_jsonl(csv_file_path, jsonl_file_path)
print("Conversion Done !!")