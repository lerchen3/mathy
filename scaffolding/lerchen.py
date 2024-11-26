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
cutoff_time = time.time() + (4 * 60 + 30) * 60

from vllm import LLM, SamplingParams

warnings.simplefilter('ignore')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import ctypes

def clean_memory(deep=False):
    gc.collect()
    if deep:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()

llm_model_pth = '/kaggle/input/qwen2.5/transformers/72b-instruct-awq/1'

llm = LLM(
    llm_model_pth,
    dtype="half",                # The data type for the model weights and activations
    max_num_seqs=8,              # Maximum number of sequences per iteration. Default is 256
    max_model_len=4096,          # Model context length
    trust_remote_code=True,      # Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer
    tensor_parallel_size=4,      # The number of GPUs to use for distributed execution with tensor parallelism
    gpu_memory_utilization=0.95, # The ratio (between 0 and 1) of GPU memory to reserve for the model
    seed=2024,
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

from collections import Counter
import random
def select_answer(answers):
    counter = Counter()
    for answer in answers:
        try:
            if int(answer) == float(answer) and int(answer) != 210:
                counter[int(answer)] += 1 + random.random() / 1_000
        except:
            pass
    if not counter:
        return 210
    _, answer = sorted([(v,k) for k,v in counter.items()], reverse=True)[0]
    return answer%1000

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
    temperature=0.7,
    top_p=0.95,
    max_tokens=4096,
    stop=None
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

def batch_message_filter(list_of_messages) -> Tuple[List[List[dict]], List[str]]:
    extracted_answers = []
    list_of_messages_to_keep = []
    for messages in list_of_messages:
        answer = extract_boxed_text(messages[-1]['content'])
        if answer:
            extracted_answers.append(answer)
        else:
            list_of_messages_to_keep.append(messages)
    return list_of_messages_to_keep, extracted_answers

def batch_message_execute(list_of_messages) -> List[List[dict]]:
    for messages in list_of_messages:
        python_code = extract_python_code(messages[-1]['content'])
        python_code = process_python_code(python_code)
        try:
            print('c', end='')
            is_successful, output = PythonREPL()(python_code)
            if is_successful:
                print('o', end='')
            else:
                print('e', end='')
        except Exception as e:
            print('f', end='')
            output = str(e)
        print(python_code)
        print()
        print(output)
        print("\n\n")
        
        # Use the new parse_steps function with error handling
        parsed_output = parse_steps(output)
        if not parsed_output:  # If no steps found, use the raw output
            messages.append({'role': 'user', 'content': output})
        else:
            messages.append({'role': 'user', 'content': '\n'.join(parsed_output)})
    print()
    return list_of_messages

#===============================================================================================================
#Everything above this line WORKS. Don't try to fix it at all.
#===============================================================================================================
def generate_solution_outline(question: str, info_set: List[Tuple[str, str]], n: int = 1) -> List[List[str]]:
    """
    Generates multiple comprehensive solution approach outlines for a given math question.
    """
    if not question or not isinstance(question, str):
        print("Warning: Invalid question input")
        return [[]]
    
    if not isinstance(info_set, list):
        print("Warning: Invalid info_set input")
        info_set = []
        
    print(f"Entering generate_solution_outline with question: {question}, info_set: {info_set}, n: {n}")
    try:
        prompt_intro = (
            "Please provide a detailed solution approach outline for the following math question. "
            "The outline should consist of clear, enumerated steps that logically "
            "progress towards the solution, but without including any computations or numerical values. "
            "Each step should specify what to do, which formulas or theorems to use, "
            "and what to plug into them, but should not perform any calculations or include any numbers. "
            "The outline should be similar to the example provided.\n\n"
        )
        print("Prompt intro prepared.")
    
        all_steps = []
        attempts = 0
        max_attempts = 5  # Prevent infinite loops
    
        while len(all_steps) < n and attempts < max_attempts:
            remaining = n - len(all_steps)
            print(f"Generating {remaining} solution outlines. Attempt {attempts + 1}/{max_attempts}")
            
            list_of_messages = []
            for i in range(remaining):
                print(f"Creating conversation {i+1}/{remaining}")
                conversation = [
                    {'role': 'user', 'content': f"{prompt_intro}\n\nQuestion: {question}"}
                ]
    
                if info_set:
                    print(f"Adding info_set to conversation {i+1}")
                    conversation.append({'role': 'user', 'content': "Here are some valuable facts and their reasoning, which may be useful to solve the question:\n"})
                    for info, reasoning in info_set:
                        conversation.append({'role': 'user', 'content': f"Fact: {info}\nReasoning: {reasoning}"})
    
                    conversation.append({'role': 'user', 'content': 
                        "Here is an example of a question, valuable information, and a good generated solution approach outline. Please follow the solution outline format as well:\n"
                        "Question:\nLet $ABCD$ be a tetrahedron such that $AB = CD = \\sqrt{41}$, "
                        "$AC = BD = \\sqrt{80}$, and $BC = AD = \\sqrt{89}$. There exists a point $I$ "
                        "inside the tetrahedron such that the distances from $I$ to each of the faces "
                        "of the tetrahedron are all equal. This distance can be written in the form "
                        "$\\frac{m \\sqrt{n}}{p}$, when $m$, $n$, and $p$ are positive integers, $m$ and "
                        "$p$ are relatively prime, and $n$ is not divisible by the square of any prime. "
                        "Find $m+n+p$.\n"
                        "Facts and Reasoning:\n"
                        "Fact: An isosceles tetrahedron can be inscribed in a rectangular box.\n"
                        "Reasoning: The symmetry of an isosceles tetrahedron allows it to fit perfectly inside a box with sides corresponding to its dimensions.\n"
                        "Solution approach outline:\n"
                        "1. Set up a coordinate system to model the tetrahedron within the box.\n"
                        "2. Identify the coordinates of the vertices based on the given side lengths.\n"
                        "3. Determine the volume of the tetrahedron using the scalar triple product.\n"
                        "4. Calculate the surface area of the tetrahedron by finding the areas of its faces using Heron's formula.\n"
                        "5. Use the formula relating the inradius, volume, and surface area of a tetrahedron to find the inradius.\n"
                        "6. Simplify the expression for the inradius to match the required form and compute $m + n + p$.\n"
                    })
                list_of_messages.append(conversation)
                print(f"Conversation {i+1} created.")
    
            # Generate responses using batch_message_generate
            print("Generating responses with batch_message_generate.")
            responses = batch_message_generate(list_of_messages)
            print("Responses generated.")
    
            for idx, conversation in enumerate(responses):
                print(f"Processing response {idx+1}/{len(responses)}")
                try:
                    solution_text = conversation[-1]['content'].strip()
                    steps = parse_steps(solution_text)
                    if steps and len(steps) <= 20:
                        print(f"Extracted steps: {steps}")
                        all_steps.append(steps)
                    else:
                        print("No steps extracted or exceeds step limit. Retrying...")
                except Exception as e:
                    print(f"Warning: Failed to process response {idx+1}: {str(e)}")
            attempts += 1
    
        if len(all_steps) < n:
            print(f"Only {len(all_steps)} solution outlines generated after {attempts} attempts.")
        else:
            print(f"Successfully generated {len(all_steps)} solution outlines.")
    
        print(f"generate_solution_outline returning {len(all_steps)} solution outlines.")
        return all_steps
    
    except Exception as e:
        print(f"Error in generate_solution_outline: {str(e)}")
        return [["1. Solve the problem."]]

def validate_solution_outline(question: str, outline: List[str], info_set: List[Tuple[str, str]] = None) -> Tuple[bool, List[str]]:
    """
    Validates if a solution outline is mathematically sound.
    """
    if not outline:
        print("Warning: Empty outline provided")
        return False, ["No solution steps provided"]
    
    if info_set is None:
        info_set = []
        
    print(f"Entering validate_solution_outline with question: {question}, outline: {outline}, info_set: {info_set}")
    try:
        is_valid = False
        missing_steps = []
        validation_prompt = [
            {'role': 'user', 'content': (
                "Please carefully analyze this mathematical solution outline and determine if it would definitely work when fully implemented.\n\n"
                "Question:\n{}\n\n"
                "Proposed solution outline:\n{}\n\n"
                "Your task:\n"
                "1. Carefully check if each step is mathematically valid, necessary, and sufficiently detailed.\n"
                "2. Verify that no crucial steps or details are missing.\n"
                "3. Confirm that the steps specify exactly which formulas or theorems to use and what to plug into them.\n"
                "4. Check if all required information is available to complete each step.\n"
                "5. Identify any potential mathematical obstacles or edge cases.\n\n"
                "Please respond in this exact format:\n"
                "DETAILED ANALYSIS: [Your step-by-step analysis]\n"
                "VALID: [Yes/No]\n"
                "CRITICAL ISSUES: [List any show-stopping problems]\n"
                "MISSING STEPS: [List any crucial missing steps or missing details]\n\n"
                "Be extremely rigorous and pedantic. A solution outline is only valid if you are 100% certain it would work when fully computed and includes every necessary detail. If there's any doubt, mark it as invalid."
            ).format(
                question,
                "\n".join(f"{i+1}. {step}" for i, step in enumerate(outline))
            )}
        ]
        
        # Generate validation response
        print("Generating validation response with batch_message_generate.")
        response = batch_message_generate([validation_prompt])[0][-1]['content']
        print("Validation response generated.")
    
        # Parse the response to determine validity and missing steps
        try:
            if 'VALID: Yes' in response and 'CRITICAL ISSUES: None' in response.replace('[]', 'None'):
                is_valid = True
                missing_steps = []
                print("Validation successful: VALID: Yes and CRITICAL ISSUES: None")
            else:
                # Extract missing steps from the response
                missing_section_start = response.find('MISSING STEPS:')
                if missing_section_start != -1:
                    missing_section = response[missing_section_start:].strip()
                    # Use parse_steps for consistent step parsing
                    missing_steps = parse_steps(missing_section)
                    if not missing_steps:  # If no numbered steps found, try bullet points
                        missing_steps = [step.lstrip('- ').strip() for step in missing_section.split('\n') 
                                       if step.strip() and step.strip().startswith('-')]
                    missing_steps = [step for step in missing_steps if step and step.lower() != 'none' and step != '[]']
                    if not missing_steps:  # If still no steps found, add a generic missing step
                        missing_steps = ["Solution approach needs refinement"]
                    print(f"Missing steps identified: {missing_steps}")
        except Exception as e:
            print(f"Error parsing validation response: {e}")
            missing_steps = ["Error validating solution outline"]
            return False, missing_steps
        
        print(f"validate_solution_outline returning: is_valid={is_valid}, missing_steps={missing_steps}")
        return is_valid, missing_steps
    except Exception as e:
        print(f"Error in validate_solution_outline: {str(e)}")
        missing_steps = ["Error validating solution outline"]
        return False, missing_steps

def make_problem_progress(question: str, info_set: List[Tuple[str, str]], missing_steps: List[str] = None) -> List[Tuple[str, str]]:
    """
    Advances problem-solving progress.
    """
    if info_set is None:
        info_set = []
    if missing_steps is None:
        missing_steps = []
        
    print(f"Entering make_problem_progress with question: {question}, info_set: {info_set}, missing_steps: {missing_steps}")
    try:
        if missing_steps:
            print("Addressing missing steps.")
            prompt = [
                {'role': 'user', 'content': (
                    "Given this mathematical problem and identified gaps in our solution approach, please provide detailed mathematical analysis to address the missing steps.\n\n"
                    "Question:\n{}\n\n"
                    "Current Information:\n{}\n\n"
                    "Missing Steps to Address:\n{}\n\n"
                    "Please:\n"
                    "1. Analyze each missing step systematically.\n"
                    "2. Identify specific mathematical relationships, theorems, or properties needed.\n"
                    "3. Derive intermediate results or establish key relationships.\n"
                    "4. Explain your reasoning for each fact discovered.\n\n"
                    "Format your response as a series of facts and their reasoning, each starting with '→ Fact: [your fact]'\n"
                    "Reasoning: [your reasoning]'.\n"
                    "For example:\n"
                    "→ Fact: The sum of angles in a triangle is 180 degrees.\n"
                    "Reasoning: In Euclidean geometry, the sum of the interior angles of any triangle is always 180 degrees.\n"
                    "→ Fact: The area of a circle is πr^2.\n"
                    "Reasoning: The area formula is derived from the definition of pi and the relationship between a circle's radius and its circumference.\n"
                ).format(
                    question,
                    "\n".join(f"Fact: {info}\nReasoning: {reasoning}" for info, reasoning in info_set),
                    "\n".join(f"- {step}" for step in missing_steps)
                )}
            ]
        else:
            print("Exploring new solution paths.")
            prompt = [
                {'role': 'user', 'content': (
                    "Please analyze this mathematical problem to discover new insights and potential solution paths.\n\n"
                    "Question:\n{}\n\n"
                    "Current Information:\n{}\n\n"
                    "Please:\n"
                    "1. Identify key mathematical properties or relationships not yet explored.\n"
                    "2. Apply relevant theorems or formulas that might yield useful results.\n"
                    "3. Consider alternative approaches or transformations of the problem.\n"
                    "4. Derive intermediate results that could lead to the solution.\n"
                    "5. Explain your reasoning for each fact discovered.\n\n"
                    "Format your response as a series of facts and their reasoning, each starting with '→ Fact: [your fact]'\n"
                    "Reasoning: [your reasoning]'.\n"
                    "For example:\n"
                    "→ Fact: The Pythagorean theorem relates the sides of a right triangle.\n"
                    "Reasoning: In a right-angled triangle, the square of the hypotenuse equals the sum of the squares of the other two sides.\n"
                    "→ Fact: Differentiation rules can be used to find the slope of a function.\n"
                    "Reasoning: Applying differentiation allows us to find the rate at which a function changes at any point.\n"
                ).format(
                    question,
                    "\n".join(f"Fact: {info}\nReasoning: {reasoning}" for info, reasoning in info_set)
                )}
            ]
        
        # Generate response
        print("Generating problem progress with batch_message_generate.")
        response = batch_message_generate([prompt])[0][-1]['content']
        print("Problem progress response generated.")

        # Extract new discoveries using consistent parsing
        print("Extracting new discoveries from response.")
        discoveries = []
        current_fact = None
        current_reasoning = []

        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            fact_match = re.match(r'^→?\s*Fact:\s*(.*)', line)
            reasoning_match = re.match(r'^Reasoning:\s*(.*)', line)
            if fact_match:
                if current_fact is not None:
                    discoveries.append((current_fact, ' '.join(current_reasoning).strip()))
                current_fact = fact_match.group(1).strip()
                current_reasoning = []
            elif reasoning_match and current_fact is not None:
                current_reasoning.append(reasoning_match.group(1).strip())
            elif current_reasoning:
                current_reasoning.append(line)
        # Don't forget the last discovery
        if current_fact is not None:
            discoveries.append((current_fact, ' '.join(current_reasoning).strip()))

        # Filter out empty or duplicate discoveries
        filtered_discoveries = [
            (fact, reasoning) for fact, reasoning in discoveries
            if fact and (fact, reasoning) not in info_set
        ]
        print(f"Filtered new discoveries: {filtered_discoveries}")
        
        # Extend info_set with new discoveries
        info_set.extend(filtered_discoveries)
        print(f"Updated info_set: {info_set}")
        
        return info_set
    except Exception as e:
        print(f"Error in make_problem_progress: {str(e)}")
        return info_set

def check_hallucinations(question: str, info_set: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Validates each piece of information in info_set and removes any that might be hallucinated or incorrect.
    
    Args:
        question (str): The original math question
        info_set (List[Tuple[str, str]]): List of facts and their reasoning to validate
            
    Returns:
        List[Tuple[str, str]]: Filtered info_set containing only verified correct statements
    """
    print(f"Entering check_hallucinations with question: {question}, info_set: {info_set}")
    
    if not info_set:
        print("info_set is empty. No hallucinations to check.")
        return []
        
    validation_prompt = [
        {'role': 'user', 'content': (
            "Please analyze each mathematical fact below and verify if it is 100% mathematically correct and directly derivable from the question. Do not make assumptions.\n\n"
            "Question:\n{}\n\n"
            "Facts to verify:\n{}\n\n"
            "For each fact, respond ONLY with the statement number and either VERIFIED or INCORRECT, followed by a brief explanation.\n"
            "Example format:\n"
            "[1] VERIFIED: Can be proven using given values\n"
            "[2] INCORRECT: Contains assumptions not supported by the question"
        ).format(
            question,
            "\n".join(f"[{i+1}] Fact: {info}\nReasoning: {reasoning}" for i, (info, reasoning) in enumerate(info_set))
        )}
    ]
    
    # Generate validation response
    print("Generating hallucination checks with batch_message_generate.")
    response = batch_message_generate([validation_prompt])[0][-1]['content']
    print("Hallucination check response generated.")

    # Parse response to identify verified statements
    print("Parsing hallucination check response.")
    verified_indices = []
    for i in range(len(info_set)):
        marker = f"[{i+1}]"
        if marker in response:
            statement_validation = response[response.find(marker):].split('\n')[0]
            if "VERIFIED" in statement_validation:
                verified_indices.append(i)
                print(f"Statement {i+1} verified.")
            else:
                print(f"Statement {i+1} not verified.")
    
    # Return only the verified statements
    verified_statements = [info_set[i] for i in verified_indices]
    print(f"Verified info_set: {verified_statements}")
    
    return verified_statements

def run_tool_integrated_reasoning(question: str, info_set: List[Tuple[str, str]], solution_outline: List[str]) -> int:
    """
    Executes tool-integrated reasoning.
    """
    start_time = time.time()
    max_execution_time = 300  # 5 minutes
    
    try:
        if not solution_outline:
            print("Warning: Empty solution outline")
            return 210
            
        print(f"Entering run_tool_integrated_reasoning with question: {question}, info_set: {info_set}, solution_outline: {solution_outline}")
        
        conversation = [
            {'role': 'system', 'content': (
                "You are writing code to solve a math problem. "
                "You are an intelligent assistant integrating natural language reasoning with programming to solve complex mathematical problems. "
                "Before writing any code, you must comment thoroughly on your reasoning for each step, and double-check your logic to ensure accuracy and correctness. "
                "Moreover, every value you get must be ran through sp.simplify() to check if it can be simplified."
            )},
            {'role': 'user', 'content': (
                "Question:\n{}\n\n"
                "Current Information:\n{}\n\n"
                "Solution Outline:\n{}\n\n"
                "Please provide your detailed reasoning (in comments) and the corresponding code for each step outlined above. "
                "Begin your response by importing sympy, and generate code in a single code block that will return the final answer."
            ).format(
                question,
                "\n".join(f"Fact: {info}\nReasoning: {reasoning}" for info, reasoning in info_set),
                "\n".join(f"{i+1}. {step}" for i, step in enumerate(solution_outline))
            )}
        ]
        
        max_attempts = 3
        for attempt in range(max_attempts):
            if time.time() - start_time > max_execution_time:
                print("Execution time limit exceeded")
                return 210
                
            print(f"Attempt {attempt+1} of {max_attempts}")
            # Generate response using the LLM
            print("Generating reasoning response with batch_message_generate.")
            response_messages = batch_message_generate([conversation])
            response_content = response_messages[0][-1]['content']
            print("Reasoning response generated.")
            # Delete the old code and error messages to save tokens
            if len(conversation) >= 4:
                potential_old_code = conversation[-3]['content']
                potential_error_message = conversation[-2]['content']
                if "code you provided" in potential_old_code and "error message" in potential_error_message:
                    del conversation[-2]
                    del conversation[-3]
                    print("Old code and error messages deleted from the conversation.")
            # Extract Python code from the response
            print("Extracting Python code from response.")
            python_code = extract_python_code(response_content)
            python_code = process_python_code(python_code)
            print(f"Extracted Python code:\n{python_code}")

            # Execute the extracted Python code and capture the output
            print("Executing extracted Python code.")
            try:
                is_successful, output = PythonREPL()(python_code)
                if is_successful:
                    print("Code execution successful.")
                    # Append the output to the conversation as a 'user' message with a heading
                    conversation.append({
                        'role': 'user',
                        'content': f"### Here are the results I got from running the code:\n\n{output}"
                    })
                    
                    # Add recap request to conversation
                    conversation.append({'role': 'user', 'content': 
                        "Please review our conversation and provide a complete recap of the solution, "
                        "keeping all mathematical expressions in a clear form. "
                        "Most importantly, state your final integer answer in this exact format on its own line:\n"
                        "FINAL ANSWER: [your integer answer]"
                    })
                    
                    # Generate recap response
                    recap_messages = batch_message_generate([conversation])
                    recap_content = recap_messages[0][-1]['content']
                    
                    # Extract final answer with more robust parsing
                    try:
                        # First try exact format
                        final_answer_match = re.search(r'FINAL ANSWER:\s*(\d+)', recap_content)
                        if final_answer_match:
                            return int(final_answer_match.group(1))
                        
                        # If not found, try parsing steps and look for answer in last step
                        steps = parse_steps(recap_content)
                        if steps:
                            # Look for numbers in the last step
                            numbers = re.findall(r'\d+', steps[-1])
                            if numbers:
                                return int(numbers[-1])
                        
                        # If still no answer found, look for any boxed text
                        boxed = extract_boxed_text(recap_content)
                        if boxed and boxed.isdigit():
                            return int(boxed)
                            
                        return 210  # Fallback answer
                    except:
                        return 210  # Fallback answer
                else:
                    print(f"Execution failed with output:\n{output}")
                    # Feed back the error into the conversation
                    conversation.append({'role': 'user', 'content': f"The code you provided failed to execute. The error message was:\n{output}\nPlease fix the code accordingly and ensure it runs correctly."})
            except Exception as e:
                print(f"Error during execution: {e}")
                # Feed back the exception into the conversation
                conversation.append({'role': 'user', 'content': f"The code you provided caused an exception:\n{e}\nPlease fix the code accordingly and ensure it runs correctly."})
            
        
        # After max_attempts, return 210
        print("Failed to get a valid result after maximum attempts.")
        return 210
        
    except Exception as e:
        print(f"Unexpected error in run_tool_integrated_reasoning: {e}")
        return 210

def predict_for_question(question: str) -> int:
    """
    Main prediction function.
    """
    start_time = time.time()
    max_execution_time = 600  # 10 minutes
    
    try:
        if not question or not isinstance(question, str):
            print("Warning: Invalid question input")
            return 210
            
        question += "\nIf the final answer is a number larger than 1 million, take modulo 1000."
        print(f"Starting prediction for question: {question}")

        info_set = []
        
        for iteration_number in range(1, 6):
            if time.time() - start_time > max_execution_time:
                print("Total execution time limit exceeded")
                return 210
                
            print(f"--- Iteration {iteration_number} ---")
            # Step 1: Generate solution outlines
            print("Generating solution outlines.")
            solution_outlines = generate_solution_outline(question, info_set, n=3)
            print(f"Generated {len(solution_outlines)} solution outlines.")
            
            for idx, outline in enumerate(solution_outlines, start=1):
                print(f"Validating solution outline {idx}: {outline}")
                # Step 2: Validate each solution outline
                is_valid, missing_steps = validate_solution_outline(question, outline, info_set)
                if is_valid:
                    print("Valid solution outline found. Running tool_integrated_reasoning...")
                    return run_tool_integrated_reasoning(question, info_set, outline)
                else:
                    # Step 3: Make problem progress with missing steps
                    if missing_steps:
                        print(f"Missing steps identified: {missing_steps}. Making progress.")
                        info_set = make_problem_progress(question, info_set, missing_steps)
                    else:
                        print("No specific missing steps identified. Making general progress.")
            
            print("Making general problem progress.")
            info_set = make_problem_progress(question, info_set, [])

            # Step 4: Check for hallucinations in info_set
            print("Checking for hallucinations in info_set.")
            info_set = check_hallucinations(question, info_set)
            print(f"Info set after hallucination check: {info_set}")
        
        # If no valid approach is found after all iterations
        print("No valid solution outline validated after all iterations. Returning fallback answer 210.")
        return 210
        
    except Exception as e:
        print(f"Unexpected error in predict_for_question: {e}")
        return 210
