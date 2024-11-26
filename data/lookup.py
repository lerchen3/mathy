import numpy as np
import pandas as pd
from embeddings import get_embedding
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_questions(question: str, data_file: str = 'processed_data.csv', top_k: int = 5) -> list:
    """
    Find the top k most similar questions using cosine similarity.
    
    Args:
        question (str): The input question to find similar matches for
        data_file (str): Path to the CSV file containing processed questions and embeddings
        top_k (int): Number of similar questions to return
        
    Returns:
        list: List of tuples containing (question_text, similarity_score)
    """
    # Get embedding for the input question
    question_embedding = get_embedding(question)
    
    # Load the processed data
    df = pd.read_csv(data_file)
    
    # Convert string representation of embeddings back to numpy arrays
    embeddings = df['context_vector'].apply(eval).tolist()
    
    # Calculate cosine similarity between input question and all stored questions
    similarities = cosine_similarity([question_embedding], embeddings)[0]
    
    # Get indices of top k similar questions
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    # Create list of (question, similarity) tuples
    similar_questions = [
        (df.iloc[idx]['question_text'], similarities[idx])
        for idx in top_indices
    ]
    
    return similar_questions
