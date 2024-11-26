import pandas as pd
from embeddings import get_embedding
import os

def process_questions(input_file, output_file):
    # Load the data from the CSV file
    df = pd.read_csv(input_file)
    results = []

    for index, row in df.iterrows():
        try:
            question = row['problem']
            
            # Get context vector embedding
            context_vector = get_embedding(question)
            
            # Append results
            results.append({
                "question_text": question,
                "context_vector": context_vector
            })
            
            # Optional: Add progress indicator
            if (index + 1) % 10 == 0:
                print(f"Processed {index + 1} questions...")
                
        except Exception as e:
            print(f"Error processing question at index {index}: {str(e)}")
            continue

    # Create a DataFrame from results and save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Processing complete. Results saved to {output_file}")

process_questions('data.csv', 'processed_data.csv')
