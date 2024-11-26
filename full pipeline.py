from typing import List, Tuple
from data.lookup import find_similar_questions
from scaffolding.lerchen import batch_message_generate, predict_for_question

def generate_helpful_info(question: str, similar_questions: List[Tuple[str, float]]) -> List[Tuple[str, str]]:
    """
    Generate helpful mathematical insights based on similar questions
    """
    prompt = [
        {'role': 'system', 'content': (
            "You are a mathematical expert that excels at identifying patterns and extracting "
            "useful mathematical concepts, techniques, and intuitions from similar problems."
        )},
        {'role': 'user', 'content': (
            f"Current Question:\n{question}\n\n"
            "Here are some similar questions:\n"
            + "\n\n".join([f"Question (similarity: {sim:.3f}): {q}" for q, sim in similar_questions])
            + "\n\nBased on these similar questions, please identify 3-5 key mathematical concepts, "
            "techniques, formulas, or intuitions that could be helpful for solving the current question. "
            "Format each insight as:\n"
            "→ Fact: [mathematical fact or technique]\n"
            "Reasoning: [why this might be helpful for the current question]"
        )}
    ]
    
    # Use the LLM to generate insights
    response = batch_message_generate([prompt])[0][-1]['content']
    
    # Parse the response to extract facts and reasoning
    facts_and_reasoning = []
    current_fact = None
    current_reasoning = []
    
    for line in response.split('\n'):
        line = line.strip()
        if line.startswith('→ Fact:') or line.startswith('Fact:'):
            if current_fact:
                facts_and_reasoning.append((current_fact, ' '.join(current_reasoning)))
            current_fact = line.split(':', 1)[1].strip()
            current_reasoning = []
        elif line.startswith('Reasoning:'):
            current_reasoning.append(line.split(':', 1)[1].strip())
        elif current_reasoning:
            current_reasoning.append(line)
    
    # Add the last pair if exists
    if current_fact:
        facts_and_reasoning.append((current_fact, ' '.join(current_reasoning)))
    
    return facts_and_reasoning

def solve_question(question: str, database_path: str = 'processed_data.csv') -> int:
    """
    Main pipeline function that combines lookup, info generation, and solution
    """
    # Step 1: Lookup similar questions
    similar_questions = find_similar_questions(question, database_path)
    
    # Step 2: Generate helpful information
    info_set = generate_helpful_info(question, similar_questions)
    
    # Step 3: Use predict_for_question with the generated info_set
    return predict_for_question(question, info_set)
