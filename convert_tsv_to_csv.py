import pandas as pd
import os

# List files in dataset directory
dataset_dir = './dataset'
if os.path.exists(dataset_dir):
    print(f"\nFiles in dataset directory:")
    for file in os.listdir(dataset_dir):
        print(f"  - {file}")

# Read TSV file
tsv_file = './dataset/4343106D673CD682131D3EBA49C069D1ki.tsv'
csv_file = './dataset/binding_affinity_data.csv'

print(f"\nReading TSV file: {tsv_file}")
print(f"File exists: {os.path.exists(tsv_file)}")

if os.path.exists(tsv_file):
    try:
        # Read TSV file
        df = pd.read_csv(tsv_file, sep='\t', low_memory=False, encoding='utf-8')
        print(f"Successfully read TSV file. Shape: {df.shape}")
        print(f"Columns: {len(df.columns)}")
        
        # Save as CSV
        df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"\nSuccessfully converted to CSV: {csv_file}")
        print(f"CSV file size: {os.path.getsize(csv_file) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"Error: {e}")
else:
    print(f"TSV file not found at: {tsv_file}")

