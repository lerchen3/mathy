import os
import gc
import time
import warnings
import re
import ast
import subprocess
import tempfile
from typing import List, Tuple

import pandas as pd
from tqdm import tqdm  # For progress tracking
import torch
from vllm import LLM, SamplingParams

# Set up environment variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure LLM with Qwen-72B model
llm_model_pth = '/kaggle/input/qwen2.5/transformers/72b-instruct-awq/1'

llm = LLM(
    llm_model_pth,
    dtype="half",                # The data type for the model weights and activations
    max_num_seqs=8,              # Maximum number of sequences per iteration. Default is 256
    max_model_len=8192,          # Model context length
    trust_remote_code=True,      # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
    tensor_parallel_size=4,      # The number of GPUs to use for distributed execution with tensor parallelism
    gpu_memory_utilization=0.95, # The ratio (between 0 and 1) of GPU memory to reserve for the model
    seed=2024,
)
tokenizer = llm.get_tokenizer()

# Define Sampling Parameters
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    stop=None
)

def batch_message_generate(list_of_messages: List[List[dict]]) -> List[List[dict]]:
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
    
    for messages, single_request_output in zip(list_of_messages, request_output):
        messages.append({'role': 'assistant', 'content': single_request_output.outputs[0].text})

    return list_of_messages

def clean_memory(deep=False):
    gc.collect()
    if deep:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()

# Read the CSV file containing math competition solutions
math = pd.read_csv('/kaggle/input/un-parsed-soluions-file/math_competitions_solutions.csv')

All_questions = []
cnt = 0
save_counter = 0  # Add counter for periodic saves

for pdf_text in tqdm(math['text'], desc="Processing documents"):
    # Reduce chunk size to be safer
    chunk_size = 4096
    
    # Add chunk progress
    words = pdf_text.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    for chunk_num, chunk in enumerate(chunks):
        if chunk_num % 10 == 0:
            print(f"Processing chunk {chunk_num}/{len(chunks)}")
            
        prompt = (
            "Extract and write in LaTeX all of the question-answers pairs of the following text "
            "in a python list of tuples. Skip any questions that you cannot write LaTeX for, or do "
            "not make sense with what you are given. For example, 'Find the sum of the first 1000 terms of the sequence' is not a valid question. Return only the python list of tuples.\n\n"
            "Requirements:\n"
            "- Questions should be self-contained (no context needed)\n"
            "- All answers must be integers from 0 to 999, inclusive. (In particular, you will definitely have to add the string 'If the final answer is a number larger than 1000, take modulo 1000' to the question a lot).\n"
            "- Include any necessary integer extraction instructions in the question\n"
            "- Use proper LaTeX formatting\n\n"
            "Format: [(r'question_in_latex', integer_answer), etc.]\n"
            "Example: [(r'What is $\sqrt{50} + \sqrt{50}$? If your answer can be expressed in the form $a\sqrt{b}$, where "
            "$a$ and $b$ are positive integers such that $b$ is not divisible by any perfect square, compute $a+b \pmod{1000}$.', 12)]\n\n"
            "Text to process: " + chunk
        )
        
        conversation = [
            {'role': 'user', 'content': prompt}
        ]
        
        # Generate response using batch_message_generate
        responses = batch_message_generate([conversation])
        response = responses[0][-1]['content']
        text = response
        
        # Extract text between the first '[' and the last ']'
        start_index = text.find('[')
        end_index = text.rfind(']')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            code_block = text[start_index:end_index+1].strip()
            # Attempt to safely evaluate the code block
            try:
                # Use multiple regex patterns to catch different possible formats
                patterns = [
                    r"\(r?'(.*?)',\s*(\d+)\)",  # Original: (r'question', 123) or ('question', 123)
                    r"\[r?'(.*?)',\s*(\d+)\]",  # [r'question', 123] or ['question', 123]
                    r"{\s*'question':\s*r?'(.*?)',\s*'answer':\s*(\d+)}",  # {'question': 'text', 'answer': 123}
                ]
                
                all_matches = []
                for pattern in patterns:
                    matches = re.findall(pattern, code_block)
                    all_matches.extend(matches)
                
                # Convert matches to a list of tuples
                for q, a in all_matches:
                    try:
                        # First verify it's a valid integer
                        a = int(a)
                        
                        # Case 2: If already in valid range, use as is
                        if 0 <= a <= 999:
                            All_questions.extend([(q, a)])
                        # Case 3: If larger, take modulo and add instruction
                        else:
                            mod_answer = a % 1000
                            q = q + "\nIf the final answer is a number larger than 1000, take modulo 1000"
                            All_questions.extend([(q, mod_answer)])
                        cnt += 1
                        if(cnt%10 == 1):
                            print("Hey the questions are actually getting scraped! Check this out:")
                            print(q)
                            print(a)
                            
                        # Add periodic saving
                        if cnt % 100 == 0:
                            save_counter += 1
                            df = pd.DataFrame(All_questions, columns=['Question', 'Answer'])
                            df.to_csv(f'allquestionanswers{save_counter}.csv', index=False)
                            print(f"Saved checkpoint {save_counter} with {len(All_questions)} questions")
                    except ValueError:
                        # Case 1: Skip if not an integer
                        print(f"Skipping non-integer answer: {a}")
            except Exception as e:
                print("Error evaluating response:")
                print(code_block)
                print(f"Exception: {e}")
        else:
            print("No list found in response:")
            print(text)

# Save All_questions to CSV
df = pd.DataFrame(All_questions, columns=['Question', 'Answer'])
df.to_csv('allquestionanswers.csv', index=False)
print("Questions and answers saved!")