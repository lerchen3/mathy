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
    
    # Check if the output file exists to determine if headers should be written
    file_exists = os.path.isfile(output_file)
    
    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['question_text', 'context_vector']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write headers only if the file does not exist
        if not file_exists:
            writer.writeheader()
        
        for index, row in df.iterrows():
            try:
                question = row['problem']
                
                # Get context vector embedding
                context_vector = get_embedding(question)
                
                # Write the result to the CSV file
                writer.writerow({
                    "question_text": question,
                    "context_vector": context_vector.tolist()  # Convert numpy array to list for CSV
                })
                
                # Optional: Add progress indicator
                if (index + 1) % 10 == 0:
                    print(f"Processed {index + 1} questions...")
                    
            except Exception as e:
                print(f"Error processing question at index {index}: {str(e)}")
                continue
    
    print(f"Processing complete. Results saved to {output_file}")

# Update file paths to use the data directory
process_questions('data/data.csv', 'data/processed_data.csv')
