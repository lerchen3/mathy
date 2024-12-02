import os
import gc
import time
import warnings
from typing import List, Tuple
import pandas as pd
import polars as pl

import torch
import kaggle_evaluation.aimo_2_inference_server

pd.set_option('display.max_colwidth', None)
cutoff_time = time.time() + (4 * 60 + 50) * 60

from vllm import LLM, SamplingParams

warnings.simplefilter('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

llm_model_pth = "/kaggle/input/qwen2.5/transformers/qwq-32b-preview-awq/1"

llm = LLM(
    llm_model_pth,
    dtype="half",                # The data type for the model weights and activations
    max_num_seqs=8,              # Maximum number of sequences per iteration; for batch generation
    max_model_len=32768,          # Model context length
    trust_remote_code=True,      # Trust remote code; need because notebook runs offline
    tensor_parallel_size=4,      # tensor parallelism
    gpu_memory_utilization=0.97, # GPU memory utilization
    seed=3,
)
tokenizer = llm.get_tokenizer()
import re
import keyword

def extract_python_code(text):
    pattern = r'```python\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    return "\n\n".join(matches)

def parse_steps(text):
    """
    Parse numbered steps from text, capturing all content between step numbers.
    """
    if not isinstance(text, str):
        return []
        
    steps = []
    lines = text.split('\n')
    current_step = None
    current_text = []
    
    for line in lines:
        line = line.strip()
        step_match = re.match(r'^(\d+)\.\s*(.*)', line)
        
        if step_match:
            # If we have a previous step, save it
            if current_step is not None:
                steps.append(' '.join(current_text).strip())
            # Start new step
            current_step = int(step_match.group(1))
            current_text = [step_match.group(2)]
        elif current_step is not None:
            current_text.append(line)
    
    # Don't forget the last step
    if current_step is not None:
        steps.append(' '.join(current_text).strip())
    
    return steps

def process_python_code(query):
    # Add import statements
    # Also print variables if they are not inside any indentation
    query = "import math\nimport numpy as np\nimport sympy as sp\n" + query
    current_rows = query.strip().split("\n")
    new_rows = []
    for row in current_rows:
        new_rows.append(row)
        if not row.startswith(" ") and "=" in row:
            variables_to_print = row.split("=")[0].strip()
            for variable_to_print in variables_to_print.split(","):
                variable_to_print = variable_to_print.strip()
                if variable_to_print.isidentifier() and not keyword.iskeyword(variable_to_print):
                    if row.count("(") == row.count(")") and row.count("[") == row.count("]"):
                        # TODO: use some AST to parse code
                        new_rows.append(f'\ntry:\n    print(f"{variable_to_print}={{str({variable_to_print})[:100]}}")\nexcept:\n    pass\n')
                # Ensure that the final answer is printed
                if 'print(' in row and 'final_answer' in row:
                    new_rows.append('print(final_answer)')
    return "\n".join(new_rows)

def extract_boxed_text(text):
    pattern = r'boxed{(.*?)}'
    matches = re.findall(pattern, text)
    if not matches:
        return ""
    return matches[0]

import os
import tempfile
import subprocess

class PythonREPL:
    def __init__(self, timeout=5):
        self.timeout = timeout

    def __call__(self, query):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, "tmp.py")
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(query)
            
            try:
                result = subprocess.run(
                    ["python3", temp_file_path],
                    capture_output=True,
                    check=False,
                    text=True,
                    timeout=self.timeout,
                )
            except subprocess.TimeoutExpired:
                return False, f"Execution timed out after {self.timeout} seconds."

            stdout = result.stdout.strip()
            stderr = result.stderr.strip()

            if result.returncode == 0:
                return True, stdout
            else:
                # Process the error message to remove the temporary file path
                # This makes the error message cleaner and more user-friendly
                error_lines = stderr.split("\n")
                cleaned_errors = []
                for line in error_lines:
                    if temp_file_path in line:
                        # Remove the path from the error line
                        line = line.replace(temp_file_path, "<temporary_file>")
                    cleaned_errors.append(line)
                cleaned_error_msg = "\n".join(cleaned_errors)
                # Include stdout in the error case
                combined_output = f"{stdout}\n{cleaned_error_msg}" if stdout else cleaned_error_msg
                return False, combined_output
            

sampling_params = SamplingParams(
    temperature=0.01,
    top_k=20,
    top_p=0.8,
    max_tokens=32768, # (in other words, unlimited)
)

def batch_message_generate(list_of_messages) -> List[List[dict]]:
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

#======================================================================================

def step1_ask_if_code_needed(question: str) -> str:
    print("Step 1: Determining if Python code is needed.")
    messages = [{'role': 'user', 'content': f"Given the following question, would using code make it significantly easier to solve? Please answer only one word: 'yes' or 'no'.\n\nQuestion:\n{question}"}]
    try:
        response = batch_message_generate([messages])[0][-1]['content'].strip().lower()
        print(f"LLM response: {response}")
        if 'yes' in response:
            print("Decision: Yes, code is needed.")
            return 'yes'
        elif 'no' in response:
            print("Decision: No, code is not needed.")
            return 'no'
        else:
            print("Unexpected response. Defaulting to 'no'.")
            return 'no'  # Default to 'no' if unexpected response
    except Exception as e:
        print(f"Error in step1_ask_if_code_needed: {e}")
        return 'no'  # Default to 'no' in case of error

def step2_generate_approaches(question: str, code_needed: str) -> List[Tuple[str, bool]]:
    import re
    print("Step 2: Generating solution approaches.")
    messages = []
    if code_needed == 'yes':
        print("Python code is needed. Generating 3 approaches with code and 3 without.")
        prompt = (
            f"Generate three solution approaches that use Python code and three that do not for the following question. "
            "The approaches should be brief and should not be full solutions. Please format your response as follows:\n\n"
            "START OF ANALYSIS \n\n[Analysis text]\n\nEND OF ANALYSIS\n\n"
            "START OF APPROACH 1\n\n[Approach text]\n\nEND OF APPROACH 1\n"
            "START OF APPROACH 2\n\n[Approach text]\n\nEND OF APPROACH 2\n"
            "START OF APPROACH 3\n\n[Approach text]\n\nEND OF APPROACH 3\n"
            "START OF APPROACH 4\n\n[Approach text]\n\nEND OF APPROACH 4\n"
            "START OF APPROACH 5\n\n[Approach text]\n\nEND OF APPROACH 5\n"
            "START OF APPROACH 6\n\n[Approach text]\n\nEND OF APPROACH 6\n\n"
            f"Question:\n{question}"
        )
    else:
        print("Python code is not needed. Generating 6 approaches without code.")
        prompt = (
            f"Generate six solution approaches for the following question. "
            "The approaches should be brief and should not be full solutions. Please format your response as follows:\n\n"
            "START OF ANALYSIS \n\n[Analysis text]\n\nEND OF ANALYSIS\n\n"
            "START OF APPROACH 1\n\n[Approach text]\n\nEND OF APPROACH 1\n"
            "START OF APPROACH 2\n\n[Approach text]\n\nEND OF APPROACH 2\n"
            "START OF APPROACH 3\n\n[Approach text]\n\nEND OF APPROACH 3\n"
            "START OF APPROACH 4\n\n[Approach text]\n\nEND OF APPROACH 4\n"
            "START OF APPROACH 5\n\n[Approach text]\n\nEND OF APPROACH 5\n"
            "START OF APPROACH 6\n\n[Approach text]\n\nEND OF APPROACH 6\n\n"
            f"Question:\n{question}"
        )
    messages.append({'role': 'user', 'content': prompt})
    try:
        response = batch_message_generate([messages])[0][-1]['content']
        print("LLM response received.")
        print("the response was: \n\n\n")
        print(response)
        print("\n\n\n\n\n\n")
        # Now parse the approaches
        pattern = r'START OF APPROACH \d+\n\n(.*?)\n\nEND OF APPROACH \d+'
        matches = re.findall(pattern, response, re.DOTALL)
        approaches = []
        for i, approach_text in enumerate(matches):
            if code_needed == 'yes' and i < 3:
                uses_code = True
            else:
                uses_code = False
            approaches.append((approach_text.strip(), uses_code))
        # Ensure we have at least 6 approaches
        if len(approaches) < 6:
            print(f"Only {len(approaches)} approaches found. Filling in with default approaches.")
            approaches += [("No approach provided.", False)] * (6 - len(approaches))
        else:
            print(f"{len(approaches)} approaches generated.")
        return approaches[:6]
    except Exception as e:
        print(f"Error in step2_generate_approaches: {e}")
        # In case of error, return fallback approaches
        return [("No approach provided.", False)] * 6

def step3_create_messages(question: str, approaches: List[Tuple[str, bool]]) -> List[List[dict]]:
    print("Step 3: Creating messages for each approach.")
    messages_list = []
    for idx, (approach, uses_code) in enumerate(approaches, 1):
        print(f"Processing approach {idx}: {'with' if uses_code else 'without'} code.")
        messages = [{'role': 'user', 'content': question}]
        if uses_code:
            approach_message = f"Approach the problem using {approach}. Your first response line should be 'import sympy as sp'. Return the final solution with the final answer when you finish. You can stop writing at any time if you feel that the approach will not amount to anything."
        else:
            approach_message = f"Approach the problem using {approach}. Return the final solution with the final answer when you finish. You can stop writing at any time if you feel that the approach will not amount to anything."
        messages.append({'role': 'user', 'content': approach_message})
        messages_list.append(messages)
    return messages_list

def step4_run_approaches(messages_list: List[List[dict]]) -> List[List[dict]]:
    print("Step 4: Running approaches using the LLM.")
    try:
        responses = batch_message_generate(messages_list)
        print("LLM responses received for all approaches.")
        return responses
    except Exception as e:
        print(f"Error in step4_run_approaches: {e}")
        # Return empty responses in case of error
        return [[] for _ in messages_list]

def step5_process_code_responses(responses: List[List[dict]], approaches: List[Tuple[str, bool]]) -> List[str]:
    print("Step 5: Processing code responses and gathering writeups.")
    processed_writeups = []
    python_repl = PythonREPL()
    for idx, (response, (approach, uses_code)) in enumerate(zip(responses, approaches), 1):
        print(f"Processing response for approach {idx}: {'with' if uses_code else 'without'} code.")
        assistant_message = response[-1]['content'] if response else ""
        if uses_code:
            # Extract code
            code = extract_python_code(assistant_message)
            print(f"Extracted code for approach {idx}:\n{code}")
            code = process_python_code(code)
            success, repl_output = python_repl(code)
            if success:
                print(f"Code execution successful for approach {idx}.")
            else:
                print(f"Code execution failed for approach {idx}.")
            # Append outputs to writeup
            writeup = assistant_message + "\n\n" + repl_output
        else:
            writeup = assistant_message
        processed_writeups.append(writeup)
    return processed_writeups

def step6_generate_final_answer(writeups: List[str]) -> str:
    print("Step 6: Generating the final answer from the writeups.")
    # Combine writeups
    combined_writeups = "\n\n".join(writeups)
    messages = [{'role': 'user', 'content': f"Here are several solution attempts:\n\n{combined_writeups}\n\nPlease analyze the writeups briefly and then return the answer with the most foolproof and convincing solution; return it in \\boxed{{}} so that we can parse it."}]
    try:
        final_response = batch_message_generate([messages])[0][-1]['content']
        print("LLM final response received.")
        final_answer = extract_boxed_text(final_response)
        if final_answer:
            print(f"Final answer extracted: {final_answer}")
            return final_answer
        else:
            print("No boxed answer found. Returning fallback answer.")
            return '210'
    except Exception as e:
        print(f"Error in step6_generate_final_answer: {e}")
        # In case of error, return fallback answer
        return '210'

def predict_for_question(question: str) -> int:
    start_time = time.time()
    TIMEOUT = 600  # 10 minutes in seconds
    
    def check_timeout():
        if time.time() - start_time > TIMEOUT:
            print("Step timeout: Prediction took longer than 10 minutes. Returning 210.")
            return True
        return False

    import os
    if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        if "Triangle" not in question:
            print("Fallback condition met. Returning 210.")
            return 210
    if time.time() > cutoff_time:
        print("Time cutoff exceeded. Returning 210.")
        return 210

    # Step 1: Determine if Python code is needed
    code_needed = step1_ask_if_code_needed(question)
    if check_timeout(): return 210

    # Step 2: Generate solution approaches
    approaches = step2_generate_approaches(question, code_needed)
    if check_timeout(): return 210

    # Error handling: Ensure we have at least 6 approaches
    if len(approaches) < 6:
        print(f"Only {len(approaches)} approaches generated. Filling in with default approaches.")
        approaches += [("No approach provided.", False)] * (6 - len(approaches))
        approaches = approaches[:6]

    # Step 3: Create messages for each approach
    messages_list = step3_create_messages(question, approaches)
    if check_timeout(): return 210

    # Step 4: Run the approaches using the LLM
    responses = step4_run_approaches(messages_list)

    # Step 5: Process code responses and gather writeups
    writeups = step5_process_code_responses(responses, approaches)

    # Step 6: Generate the final answer
    final_answer = step6_generate_final_answer(writeups)

    # Step 7: Return the answer
    try:
        final_int_answer = int(final_answer)
        print(f"Final answer: {final_int_answer}")
        final_int_answer = final_int_answer%1000
        return final_int_answer
    except ValueError:
        print("Final answer is not an integer. Returning fallback answer 210.")
        return 210