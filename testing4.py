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
    max_model_len=32768,          # Model context length
    trust_remote_code=True,      # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
    tensor_parallel_size=4,      # The number of GPUs to use for distributed execution with tensor parallelism
    gpu_memory_utilization=0.95, # The ratio (between 0 and 1) of GPU memory to reserve for the model
    seed=2024,
)
tokenizer = llm.get_tokenizer()

# Define Sampling Parameters
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=32768,
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

# Load the checkpoint
checkpoint_df = pd.read_csv('/kaggle/input/checkpoints/allquestionanswers8.csv')
All_questions = list(zip(checkpoint_df['Question'], checkpoint_df['Answer']))
cnt = len(All_questions)  # Initialize counter to current length
save_counter = 8  # Start from checkpoint 8

def process_batch(batch_texts: List[str]) -> None:
    global cnt, save_counter, All_questions
    
    conversations = []
    for pdf_text in batch_texts:
        # Reduce chunk size to be safer
        chunk_size = 4096
        words = pdf_text.split()
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        
        for chunk in chunks:
            prompt = (
                "Extract and write in LaTeX all of the question-answers pairs of the following text "
                "in a python list of tuples. Skip any questions that you cannot write LaTeX for, or do "
                "not make sense with what you are given. For example, 'What is the sum of the first 1000 terms of the sequence?' is not a valid question; however, 'Define the sequence $a$ by $a_0 = 1$, and $a_i = 2 \cdot a_{i-1}$ for all $i \geq 1$. What is the sum of the first 1000 terms of the sequence?' IS a valid question.\n\n"
                "Requirements:\n"
                "- Before doing anything, make sense of each question. Make sure you know what the question is asking, and that you are able to provide a complete and coheisve question, like the example above. If you are not sure, just skip the question; it may be the case that there are simply no questions in the text. \n"
                "- Questions should be self-contained (no context needed)\n"
                "- All answers must be integers from 0 to 999, inclusive. (In particular, you will definitely have to add the string 'If the final answer is a number larger than 1000, take modulo 1000' to the question a lot).\n"
                "- Include any necessary integer extraction instructions in the question\n"
                "- Use proper LaTeX formatting\n\n"
                "Format: [(r'question_in_latex (as a python string)', integer_answer (as an integer from 0 to 999, inclusive)), etc.]\n"
                "Example: [(r'What is $\sqrt{50} + 1000\sqrt{2}$? If your answer can be expressed in the form $a\sqrt{b}$, where "
                "$a$ and $b$ are positive integers such that $b$ is not divisible by any perfect square, compute. If the final answer is a number larger than 1000, take modulo 1000.', 7)]\n\n"
                "Text to process: " + chunk
            )
            conversations.append([{'role': 'user', 'content': prompt}])
            
            # Process in batches of 8
            if len(conversations) >= 8:
                process_conversation_batch(conversations)
                conversations = []
    
    # Process any remaining conversations
    if conversations:
        process_conversation_batch(conversations)

def process_conversation_batch(conversations: List[List[dict]]) -> None:
    global cnt, save_counter, All_questions
    
    # Generate responses for the batch
    responses = batch_message_generate(conversations)
    
    # Process each response
    for response in responses:
        text = response[-1]['content']
        
        # Extract text between the first '[' and the last ']'
        start_index = text.find('[')
        end_index = text.rfind(']')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            code_block = text[start_index:end_index+1].strip()
            try:
                patterns = [
                    r"\(r?'(.*?)',\s*(\d+)\)",
                    r"\[r?'(.*?)',\s*(\d+)\]",
                    r"{\s*'question':\s*r?'(.*?)',\s*'answer':\s*(\d+)}",
                ]
                
                all_matches = []
                for pattern in patterns:
                    matches = re.findall(pattern, code_block)
                    all_matches.extend(matches)
                
                for q, a in all_matches:
                    try:
                        a = int(a)
                        if 0 <= a <= 999:
                            All_questions.extend([(q, a)])
                        else:
                            mod_answer = a % 1000
                            q = q + "\nIf the final answer is a number larger than 1000, take modulo 1000"
                            All_questions.extend([(q, mod_answer)])
                        cnt += 1
                        if(cnt%10 == 1):
                            print("Hey the questions are actually getting scraped! Check this out:")
                            print(q)
                            print(a)
                        
                        if cnt % 100 == 0:
                            save_counter += 1
                            df = pd.DataFrame(All_questions, columns=['Question', 'Answer'])
                            df.to_csv(f'allquestionanswers{save_counter}.csv', index=False)
                            print(f"Saved checkpoint {save_counter} with {len(All_questions)} questions")
                    except ValueError:
                        print(f"Skipping non-integer answer: {a}")
            except Exception as e:
                print("Error evaluating response:")
                print(code_block)
                print(f"Exception: {e}")
        else:
            print("No list found in response:")
            print(text)

# Modified main processing loop to start from batch 16
batch_size = 8
total_batches = len(math['text'])
start_batch = 16  # Start from batch 16 (index 15)

for i in tqdm(range(start_batch * batch_size, len(math['text']), batch_size), 
              initial=start_batch, 
              total=len(math['text'])//batch_size,
              desc="Processing document batches"):
    batch_texts = math['text'][i:i+batch_size].tolist()
    process_batch(batch_texts)

# Save All_questions to CSV
df = pd.DataFrame(All_questions, columns=['Question', 'Answer'])
df.to_csv('allquestionanswers.csv', index=False)
print("Questions and answers saved!")