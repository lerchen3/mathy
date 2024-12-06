import requests
from pdfminer.high_level import extract_text
import pandas as pd
from io import BytesIO
import os
import csv
import time
import logging

# Configure logging
logging.basicConfig(
    filename='crawler.log',  # Unified log file for all sources
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Initialize an empty list to store data
data = []

# ---------------------------
# HMMT (High School Math Tournament) Configuration
# ---------------------------

# For February
hmmt_feb_years = range(2004, 2025)
hmmt_feb_month = 'feb'
hmmt_feb_problem_types = ['team', 'algnt', 'comb', 'geo', 'guts']

# For November
hmmt_nov_years = range(2012, 2025)
hmmt_nov_month = 'nov'
hmmt_nov_problem_types = ['team', 'gen', 'theme', 'guts']

# HMMT Base URL pattern
hmmt_base_url = 'https://hmmt-archive.s3.amazonaws.com/tournaments/{year}/{month}/{problem_type}/solutions.pdf'

# ---------------------------
# SMT (Stanford Math Tournament) Configuration
# ---------------------------

# Years to process for SMT
smt_years = range(2004, 2025)

# Subjects to process for SMT (as per user request)
smt_subjects = ['algebra', 'geometry', 'team', 'discrete', 'advanced']

# SMT Base URL pattern
smt_base_url = 'https://www.stanfordmathtournament.com/pdfs/smt{year}/{subject}-solutions.pdf'

# ---------------------------
# OMO and NIMO Configuration
# ---------------------------

# Path to the folder containing the PDF files
folder_path = "/kaggle/input/omo-and-nimo"  # Adjust the folder path as needed

# List of file names to parse
file_names = [
    "OMOSpring20Solutions.pdf", "OMOFall12Soln.pdf", "OMOFall13Solns.pdf",
    "OMOFall14Solns.pdf", "OMOFall15Solns.pdf", "OMOFall16Solns.pdf",
    "OMOFall17Solns.pdf", "OMOFall18Solns.pdf", "OMOFall19Solns.pdf",
    "OMOSpring14Solns.pdf", "OMOSpring15Solns.pdf", "OMOSpring16Solns.pdf",
    "OMOSpring17Solns.pdf", "OMOSpring18Solns.pdf", "OMOSpring19Solns.pdf",
    "NIMO All Problems.pdf",
]

# ---------------------------
# Function to Process PDFs
# ---------------------------

def process_pdf_content(pdf_content, source):
    """
    Extracts text from a PDF content and appends it to the data list.

    Args:
        pdf_content (BytesIO or file object): The PDF content to extract text from.
        source (str): Identifier for the source for logging purposes.
    """
    try:
        text = extract_text(pdf_content)
        data.append({'source': source, 'text': text})
        logging.info(f"Processed {source}")
    except Exception as e:
        logging.error(f"Error processing {source}: {e}")
    finally:
        time.sleep(1)  # Sleep for 1 second between requests to be polite

def process_online_pdfs(url, source):
    """
    Downloads a PDF from the given URL, extracts its text, and appends it to the data list.

    Args:
        url (str): The URL of the PDF to download.
        source (str): Identifier for the source for logging purposes.
    """
    try:
        response = requests.get(url)
        if response.status_code == 200:
            pdf_content = BytesIO(response.content)
            process_pdf_content(pdf_content, source)
        else:
            logging.warning(f"Missing file ({source}): {url}")
    except Exception as e:
        logging.error(f"Error processing {source} ({url}): {e}")
    finally:
        time.sleep(1)  # Sleep for 1 second between requests to be polite

def process_local_pdf(file_path, source):
    """
    Reads a local PDF file, extracts its text, and appends it to the data list.

    Args:
        file_path (str): The path to the local PDF file.
        source (str): Identifier for the source for logging purposes.
    """
    try:
        with open(file_path, 'rb') as f:
            if source == "NIMO All Problems.pdf":
                # Special case: Print the whole PDF string and do NOT append it to data
                text = extract_text(f)
                print(text)
                logging.info(f"Printed content of {source}")
            else:
                # Process and append to data
                process_pdf_content(f, source)
    except Exception as e:
        logging.error(f"Error processing {source} ({file_path}): {e}")
    finally:
        time.sleep(1)  # Sleep for 1 second between processing to be polite

# ---------------------------
# Processing HMMT PDFs
# ---------------------------

def process_hmmt_pdfs():
    # Process February HMMT PDFs
    for year in hmmt_feb_years:
        for problem_type in hmmt_feb_problem_types:
            url = hmmt_base_url.format(year=year, month=hmmt_feb_month, problem_type=problem_type)
            source = f"HMMT-February {year} {problem_type}"
            process_online_pdfs(url, source=source)

    # Process November HMMT PDFs
    for year in hmmt_nov_years:
        for problem_type in hmmt_nov_problem_types:
            url = hmmt_base_url.format(year=year, month=hmmt_nov_month, problem_type=problem_type)
            source = f"HMMT-November {year} {problem_type}"
            process_online_pdfs(url, source=source)

# ---------------------------
# Processing SMT PDFs
# ---------------------------

def process_smt_pdfs():
    for year in smt_years:
        for subject in smt_subjects:
            url = smt_base_url.format(year=year, subject=subject)
            source = f"SMT {year} {subject}"
            process_online_pdfs(url, source=source)

# ---------------------------
# Processing OMO and NIMO PDFs
# ---------------------------

def process_omo_nimo_pdfs():
    for file_name in file_names:
        full_path = os.path.join(folder_path, file_name)
        if os.path.exists(full_path):
            source = file_name  # Use the file name as the source identifier
            process_local_pdf(full_path, source=source)
        else:
            logging.warning(f"File not found: {full_path}")

# ---------------------------
# Main Execution
# ---------------------------

if __name__ == "__main__":
    logging.info("Starting PDF crawling and extraction.")

    # Process HMMT PDFs
    logging.info("Processing HMMT PDFs.")
    process_hmmt_pdfs()

    # Process SMT PDFs
    logging.info("Processing SMT PDFs.")
    process_smt_pdfs()

    # Process OMO and NIMO PDFs
    logging.info("Processing OMO and NIMO PDFs.")
    process_omo_nimo_pdfs()

    logging.info("Completed PDF crawling and extraction.")
    
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
    
    All_questions = []
    cnt = 0
    save_counter = 0  # Add counter for periodic saves
    
    for pdf in tqdm(data, desc="Processing documents"):
        # Reduce chunk size to be safer
        print(f"Processing source: {pdf['source']}")
        chunk_size = 4096
        
        # Add chunk progress
        words = pdf['text'].split()
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        for chunk_num, chunk in enumerate(chunks):
            if chunk_num % 10 == 0:
                print(f"Processing chunk {chunk_num}/{len(chunks)}")
                
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