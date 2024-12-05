from transformers import set_seed
set_seed(20090302)
import os
import gc
import time
import warnings

import pandas as pd
import polars as pl

import torch

pd.set_option('display.max_colwidth', None)
cutoff_time = time.time() + (4 * 60 + 45) * 60

from vllm import LLM, SamplingParams

warnings.simplefilter('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def clean_memory(deep=False):
    gc.collect()
    if deep:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()

llm_model_pth = '/kaggle/input/m/shelterw/qwen2.5/transformers/qwq-32b-preview-awq/1'

llm = LLM(
    llm_model_pth,
    #dtype="half",                # The data type for the model weights and activations
    #max_num_seqs=128,              # Maximum number of sequences per iteration. Default is 256
    max_model_len=16384,#4096*10,          # Model context length
    trust_remote_code=True,      # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
    tensor_parallel_size=4,      # The number of GPUs to use for distributed execution with tensor parallelism
    gpu_memory_utilization=0.96, # The ratio (between 0 and 1) of GPU memory to reserve for the model
)
tokenizer = llm.get_tokenizer()

import re

thoughts = [
    'Please use chained reasoning to put the answer in \\boxed{}.',
    'Please reflect and verify while reasoning and put the answer in \\boxed{}.',
    'Solve the following problem using concise and clear reasoning by placing the answer in \\boxed{}.',
    'You are a helpful and reflective maths assistant, please reason step by step to put the answer in \\boxed{}.',
    'You are the smartest maths expert in the world, please spike this question and put the answer in \\boxed{}.'
]

def make_next_prompt(text,round_idx):
    default_prompt = thoughts[(round_idx+1)%len(thoughts)] #'No boxed answer found,please generate python code or put the answer within \\boxed{}.'
    default_python_code = f"print('{default_prompt}')"
    return default_python_code

import re

def extract_boxed_text(text):
    pattern = r'oxed{(.*?)}' #\boxed{...} gets turned into oxed{...}; the left off b is intentional.
    matches = re.findall(pattern, text)
    if not matches:
        return -1
    content = matches[0]
    if content.isdigit():
        num = int(content)
    else:
        nums = re.findall(r'\d+', content)
        if not nums:
            return -1
        num = int(nums[-1])
    return num % 1000

from collections import Counter
def select_answer(answers):
    valid_answers = []
    for answer in answers:
        try:
            if answer != -1:
                num = int(answer)
                if 1 < num < 999 and num % 100 > 0:
                    valid_answers.append(num)
        except (ValueError, TypeError):
            continue
    if not valid_answers:
        return 210
    most_common = Counter(valid_answers).most_common(1)
    return most_common[0][1] % 1000 if most_common else 49

sampling_params = SamplingParams(
    temperature=1.0,              # randomness of the sampling
    min_p=0.01,
    skip_special_tokens=True,     # Whether to skip special tokens in the output.
    #max_tokens=1800,
    max_tokens=16384,
    #stop=["```output"],
)

def batch_message_generate(list_of_messages) -> list[list[dict]]:
    list_of_texts = [
        tokenizer.apply_chat_template(
            conversation=messages,
            tokenize=False,
            add_generation_prompt=True
        )
        for messages in list_of_messages
    ]
    
    request_output = llm.generate(
        prompts=list_of_texts,
        sampling_params=sampling_params,
    )
    
    boxed_answers = []
    for messages, single_request_output in zip(list_of_messages, request_output):
        response_text = single_request_output.outputs[0].text
        messages.append({'role': 'assistant', 'content': response_text})
        boxed_answers.append(extract_boxed_text(response_text))
        print(messages[-1])

    return list_of_messages, boxed_answers

from collections import Counter, defaultdict
def predict_with_response(question: str, answer: int, prompt_heads: list[str]) -> list[tuple[str, str, str, int, bool]]:
    """
    Returns a list of (prompt_head, response, extracted_answer, solved) for each prompt head.
    
    Args:
        question: The question to ask
        answer: The correct answer to the question
        prompt_heads: List of system prompts to use
        
    Returns:
        List of tuples, each containing:
        - prompt_head: The system prompt used
        - response: The full response from the assistant
        - extracted_answer: The extracted boxed answer (-1 if none found)
        - solved: Boolean indicating if the answer matches the correct answer
    """
    question += "\nIf the final answer is a number larger than 1000, take modulo 1000. "
    if time.time() > cutoff_time:
        return [(head, "", -1, False) for head in prompt_heads]
    
    list_of_messages = [
        [
            {"role": "system", "content": head},
            {"role": "user", "content": question}
        ] for head in prompt_heads
    ]

    # Get responses
    list_of_messages, boxed_answers = batch_message_generate(list_of_messages)
    
    # Create result list
    results = []
    for head, messages, extracted_answer in zip(prompt_heads, list_of_messages, boxed_answers):
        assistant_response = messages[-1]['content']
        solved = extracted_answer != -1 and extracted_answer == answer
        results.append((question, head, assistant_response, extracted_answer, solved))
    
    return results

import pandas as pd
thoughts = [
    # Original 5 prompt heads
    'Please use chained reasoning to put the answer in \\boxed{}.', 
    'Please reflect and verify while reasoning and put the answer in \\boxed{}.', 
    'Solve the following problem using concise and clear reasoning and put the answer in \\boxed{}.', 
    'You are a helpful and reflective math assistant. Please reason step by step to put the answer in \\boxed{}.', 
    'You are the smartest math expert in the world. Please spike this question and put the answer in \\boxed{}.', 
    
    # Step-by-Step Thinking
    'Solve this problem using the most explicit step-by-step reasoning in \\boxed{}.', 
    'Break down the solution into the clearest, most logical steps in \\boxed{}.', 
    'Solve by creating a comprehensive, step-by-step mathematical breakdown in \\boxed{}.', 
    'Use methodical, incremental reasoning to solve this problem in \\boxed{}.', 
    'Navigate the solution through a clear, sequential mathematical path in \\boxed{}.', 

    # Efficiency and Optimization
    'Solve this problem using the most mathematically efficient approach in \\boxed{}.', 
    'Find the most streamlined solution path with minimal computational steps in \\boxed{}.', 
    'Optimize your problem-solving strategy to minimize computational complexity in \\boxed{}.', 
    'Solve using the most direct and efficient mathematical reasoning in \\boxed{}.', 
    'Identify and apply the most computationally elegant solution in \\boxed{}.', 

    # Verification and Double-Checking
    'Solve the problem, then meticulously verify each step of your solution in \\boxed{}.', 
    'Cross-validate your solution using multiple mathematical verification techniques in \\boxed{}.', 
    'Solve, then conduct a rigorous mathematical self-audit of your approach in \\boxed{}.', 
    'Verify your solution through independent mathematical cross-checking in \\boxed{}.', 
    'Apply comprehensive verification to ensure the absolute accuracy of your solution in \\boxed{}.', 

    # Critical and Reflective Thinking
    'Solve the problem, then critically analyze your mathematical reasoning in \\boxed{}.', 
    'Reflect deeply on your solution method and mathematical approach in \\boxed{}.', 
    'Use critical thinking to evaluate the most robust solution strategy in \\boxed{}.', 
    'Solve while maintaining a reflective and analytical mathematical mindset in \\boxed{}.', 
    'Critically examine and refine your problem-solving approach in \\boxed{}.', 

    # Precision and Minimal Reasoning
    'Solve using the most precise and minimal set of mathematical principles in \\boxed{}.', 
    'Apply mathematical reasoning with maximum precision and minimal assumptions in \\boxed{}.', 
    'Identify and use only the most essential mathematical steps in \\boxed{}.', 
    'Solve with absolute mathematical precision and computational efficiency in \\boxed{}.', 
    'Use the most lean and precise mathematical reasoning approach in \\boxed{}.', 

    # Additional Problem-Solving Strategies
    'Solve by systematically unpacking the mathematical structure of the problem in \\boxed{}.', 
    'Use logical decomposition to break down and solve the mathematical challenge in \\boxed{}.', 
    'Apply a structured, systematic approach to mathematical problem-solving in \\boxed{}.', 
    'Solve by identifying and leveraging the core mathematical principles in \\boxed{}.', 
    'Use a methodical approach to extract the most elegant mathematical solution in \\boxed{}.', 

    # Final Focused Approaches
    'Solve using pure mathematical logic and systematic reasoning in \\boxed{}.', 
    'Apply the most rigorous mathematical deduction techniques in \\boxed{}.', 
    'Solve through a careful, deliberate mathematical reasoning process in \\boxed{}.', 
    'Use fundamental mathematical principles to drive your solution in \\boxed{}.', 
    'Apply a disciplined, mathematical approach to problem resolution in \\boxed{}.'
]
questions = pd.read_csv('/kaggle/input/hmmt-questions-yay/hmmtquestionsyay.csv')

import random
data = []
cnt = 0
question_answer_data = []
batch_size = 6

# Process questions in batches
for i in range(0, len(questions), batch_size):
    batch = questions.iloc[i:i + batch_size]
    
    # Create messages to ask the LLM if the solution contains a boxed answer
    list_of_messages = [
        [
            {"role": "user", "content": f"Question: {row['Question']}\nSolution: {row['Solution']}\nPlease check if the solution contains a boxed answer. If it does, return the answer in \\boxed{{}}. Otherwise, return -1."}
        ] for _, row in batch.iterrows()
    ]
    
    # Get responses and extract answers
    _, boxed_answers = batch_message_generate(list_of_messages)
    
    # Store valid question-answer pairs
    for (_, row), extracted_answer in zip(batch.iterrows(), boxed_answers):
        if extracted_answer != -1:  # Only keep if we successfully extracted an answer
            question_answer_data.append([row['Question'], extracted_answer])

# Save results
result_df = pd.DataFrame(question_answer_data, columns=['question', 'answer'])
result_df.to_csv('hmmt_question_answer_data.csv', index=False)