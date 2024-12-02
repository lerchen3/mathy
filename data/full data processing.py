import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
import csv
import os

# Load the MathBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("tbs17/MathBERT-custom")
model = AutoModel.from_pretrained("tbs17/MathBERT-custom")

def get_embedding(sentence):
    inputs = tokenizer(sentence, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    # Compute the mean of the last hidden states
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()
    return embeddings.numpy()

def process_questions(input_file, output_file):
    # Load the data from the CSV file
    df = pd.read_csv(input_file)
    
    # Create empty output CSV with headers
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['problem', 'solution', 'context_vector'])
    
    # Process each question one at a time
    for index, row in df.iterrows():
        try:
            # Get embedding for the problem
            embedding = get_embedding(row['problem'])
            
            # Append to CSV
            with open(output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    row['problem'],
                    row['solution'],
                    embedding.tolist()  # Convert numpy array to list for CSV storage
                ])
            
            if index % 10 == 0:  # Print progress every 10 items
                print(f"Processed {index + 1}/{len(df)} questions")
                
        except Exception as e:
            print(f"Error processing question {index + 1}: {str(e)}")
            continue
    
    print(f"Processing complete. Results saved to {output_file}")

# Update file paths to use the data directory
process_questions('data/data.csv', 'data/processed_data.csv')
