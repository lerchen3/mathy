import os
import multiprocessing
import warnings
import random
import re
from collections import Counter
from transformers import set_seed, AutoTokenizer
from vllm import LLM, SamplingParams

def model_worker(model_path, gpu_ids, input_queue, output_queue):
    """
    Worker function to load a model and handle batch generation requests.
    """
    # Set environment variables specific to this process
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    try:
        # Initialize the LLM model
        llm = LLM(
            model_path,
            dtype="half",
            max_num_seqs=64,
            max_model_len=32768 if 'qwq-32b-preview-awq' in model_path else 8192,
            trust_remote_code=True,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.96,
        )

        # Get the tokenizer
        tokenizer = llm.get_tokenizer()

        # We will dynamically override max_tokens in calls if needed.
        # Default sampling_params as baseline (will be overridden by caller logic).
        default_sampling_params = SamplingParams(
            temperature=0,
            min_p=0.01,
            skip_special_tokens=True,
            max_tokens=32768 if 'qwq-32b-preview-awq' in model_path else 8192,
        )

        while True:
            # Wait for a batch of (prompts, sampling_params) from the input queue
            data = input_queue.get()
            if data == "TERMINATE":
                break  # Exit the loop to terminate the worker

            if isinstance(data, dict):
                batch = data.get("batch", [])
                sampling_params = data.get("sampling_params", default_sampling_params)
            else:
                # Backward compatibility if single list is sent
                batch = data
                sampling_params = default_sampling_params

            if not isinstance(batch, list):
                batch = [batch]

            outputs = llm.generate(batch, sampling_params)
            output_texts = [output.outputs[0].text for output in outputs]

            # Put the list of results into the output queue
            output_queue.put(output_texts)

    except Exception as e:
        # In case of error, put the exception message into the output queue
        output_queue.put([f"Error: {str(e)}"])

def initialize_workers(model1_path, gpu_ids1, model2_path, gpu_ids2):
    """
    Initializes and starts worker processes for both models.
    Returns the input and output queues for both models.
    """
    input_queue1 = multiprocessing.Queue()
    output_queue1 = multiprocessing.Queue()

    input_queue2 = multiprocessing.Queue()
    output_queue2 = multiprocessing.Queue()

    p1 = multiprocessing.Process(target=model_worker, args=(model1_path, gpu_ids1, input_queue1, output_queue1))
    p2 = multiprocessing.Process(target=model_worker, args=(model2_path, gpu_ids2, input_queue2, output_queue2))

    p1.start()
    p2.start()

    print("Worker processes started and models initialized.")

    return (input_queue1, output_queue1, p1), (input_queue2, output_queue2, p2)

def terminate_workers(input_queue1, input_queue2, p1, p2):
    """
    Sends termination signals to worker processes and joins them.
    """
    input_queue1.put("TERMINATE")
    input_queue2.put("TERMINATE")
    p1.join()
    p2.join()
    print("Worker processes terminated.")

def batch_conversations_generate(conversations, tokenizer, input_queue1, output_queue1, 
                                 input_queue2, output_queue2, model='model1', max_tokens=2048) -> list:
    """
    Generate responses for a batch of conversations using the specified model.

    Args:
        conversations (list[list[dict]]): A list of conversations, where each conversation is a list of messages (dicts with 'role' and 'content').
        tokenizer: Tokenizer to prepare prompts.
        model (str): 'model1' or 'model2'. Determines which model to use.
        max_tokens (int): Max tokens to generate (override).

    Returns:
        list[list[dict]]: The updated conversations with assistant responses appended.
    """
    conversations_texts = [
        tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        for conversation in conversations
    ]

    # Create custom sampling_params
    sampling_params = SamplingParams(
        temperature=0,
        min_p=0.01,
        skip_special_tokens=True,
        max_tokens=max_tokens,
    )

    if model == 'model1':
        input_queue1.put({"batch": conversations_texts, "sampling_params": sampling_params})
        outputs = output_queue1.get()
    else:
        input_queue2.put({"batch": conversations_texts, "sampling_params": sampling_params})
        outputs = output_queue2.get()

    # Append responses
    for i, conversation in enumerate(conversations):
        conversation.append({'role': 'assistant', 'content': outputs[i]})

    return conversations


def extract_boxed_text(text):
    pattern = r'oxed{(.*?)}'
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
    return most_common[0][0] % 1000 if most_common else 210

def get_n_prompts(n, thoughts):
    return random.sample(thoughts, n)

def parse_current_approach_and_path(text):
    # We expect something like:
    # CURRENT APPROACH: <some text>
    # ANALYSIS: <...>
    # ON A GOOD PATH: yes/no
    # We'll try a simple regex parse
    ca_match = re.search(r'CURRENT APPROACH:\s*(.*?)\n', text, re.DOTALL)
    if not ca_match:
        return None, None
    current_approach = ca_match.group(1).strip()

    path_match = re.search(r'ON A GOOD PATH\s*:\s*(yes|no)', text, re.IGNORECASE)
    if not path_match:
        return None, None
    path = path_match.group(1).lower()
    if path not in ['yes', 'no']:
        return None, None

    return current_approach, path

def run_analysis_with_model2(conversations, tokenizer, input_queue1, output_queue1, input_queue2, output_queue2):
    """
    For each conversation in conversations, use Qwen-32b (model2) to produce a text with the requested format.
    Prompt format (we will create a system message that instructs model2 to produce the required output):
    We will feed the entire conversation so far and then as a system message request the analysis.
    """
    # We'll create a new conversation for model2 input.
    # According to instructions:
    # "CURRENT APPROACH: <subset text>\n ANALYSIS (of ...)\n ON A GOOD PATH <yes/no>."
    # We must give model2 something that leads it to produce this format.
    # We'll just instruct it directly. Assume we just show the conversation and ask model2 to comply:

    # We'll try to extract only relevant info from the conversation:
    # The instructions: 
    # "Use Qwen-32b (model2), batch-size-(however many), on the conversations in active_conversations.
    # [you’ll have to create a prompt for this], output:
    #  “CURRENT APPROACH: <use a cohesive subset of the text given, remove unnecessary details> \n
    #   ANALYSIS (of whether the approach shows promise in finding an answer) <analysis> \n
    #   ON A GOOD PATH <yes / no>.”
    # For simplicity, we'll provide a system message that instructs model2 to read the conversation and produce that format.

    # We'll just pass the entire conversation as if user said it, and a system message instructing the format.
    # Then model2 returns a single assistant message with the format.

    model2_input = []
    for conv in conversations:
        # Convert the conversation to a text block
        conv_text = ""
        for msg in conv:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                conv_text += f"[System]: {content}\n"
            elif role == 'user':
                conv_text += f"[User]: {content}\n"
            else:
                conv_text += f"[Assistant]: {content}\n"

        # Add a system prompt at the end instructing model2 what to do:
        # We'll ask model2:
        # "Please read the conversation above and produce the following output:
        # CURRENT APPROACH: <cohesive subset of the text that represents the current approach>
        # ANALYSIS (of whether the approach shows promise in finding an answer) <analysis here>
        # ON A GOOD PATH <yes/no>."
        # Return only this format.
        new_conv = [
            {'role': 'user', 'content': conv_text.strip()},
            {'role': 'system', 'content': "Please analyze the above conversation. Extract the latest attempt at a solution approach. Then produce exactly:\nCURRENT APPROACH: <approach>\nANALYSIS (of whether the approach shows promise in finding an answer) <analysis>\nON A GOOD PATH <yes/no>."}
        ]
        model2_input.append(new_conv)

    # Run model2 on all these
    analyzed_convs = batch_conversations_generate(model2_input, tokenizer, input_queue1, output_queue1, input_queue2, output_queue2, model='model2', max_tokens=2048)

    # Extract results
    results = []
    for ac in analyzed_convs:
        # The assistant's last message is the model2 output
        analysis_text = ac[-1]['content']
        current_approach, path = parse_current_approach_and_path(analysis_text)
        if current_approach is None or path is None:
            # error
            results.append(('error', None))
        else:
            results.append((path, current_approach))
    return results


def predict_for_question(question: str,
                         tokenizer,
                         input_queue1, output_queue1,
                         input_queue2, output_queue2) -> int:

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

    # 0.
    active_conversations1 = []
    active_conversations2 = []
    extracted_answers = []

    def create_conversations_from_prompts(prompts, question):
        # Each conversation is system-> user(question)
        # We'll just do: [ {role: system, content: prompt}, {role: user, content: question} ]
        # Then generate with model1
        convs = []
        for pr in prompts:
            convs.append([
                {'role': 'system', 'content': pr},
                {'role': 'user', 'content': question}
            ])
        return convs

    # Steps outline:
    # 1) Get random subset of size 8
    prompts_1 = get_n_prompts(8, thoughts)
    convs_1 = create_conversations_from_prompts(prompts_1, question)

    # 2) run batch generate with model1 (QwQ-32b), max_tokens=2048
    convs_1 = batch_conversations_generate(convs_1, tokenizer, input_queue1, output_queue1, input_queue2, output_queue2, model='model1', max_tokens=2048)

    # 3) append them to active_conversations1
    active_conversations1.extend(convs_1)

    # 3a) run batch generate on another 8 random prompts for active_conversations2
    prompts_2 = get_n_prompts(8, thoughts)
    convs_2 = create_conversations_from_prompts(prompts_2, question)
    convs_2 = batch_conversations_generate(convs_2, tokenizer, input_queue1, output_queue1, input_queue2, output_queue2, model='model1', max_tokens=2048)
    active_conversations2.extend(convs_2)

    # 3b) Check if any in active_conversations1 have oxed{}:
    def process_conversations_for_answers_and_filter(convs):
        # Extract answers
        # If found, append answer to extracted_answers
        # Then run analysis with model2, parse result
        # Discard "no" and errors, keep "yes" but replace last message content
        # If error, delete conversation
        keep_convs = []
        to_analyze = []
        indexed = []

        # First check for answers
        for i, c in enumerate(convs):
            full_text = " ".join(m['content'] for m in c if m['role'] == 'assistant')
            ans = extract_boxed_text(full_text)
            if ans != -1:
                extracted_answers.append(ans)
            # We'll analyze all convs anyway
            to_analyze.append(c)
            indexed.append(i)

        if not to_analyze:
            return convs

        results = run_analysis_with_model2(to_analyze, tokenizer, input_queue1, output_queue1, input_queue2, output_queue2)

        for i, (path, approach) in enumerate(results):
            idx = indexed[i]
            if path == 'error':
                # delete this conversation
                continue
            if path == 'no':
                # discard
                continue
            if path == 'yes':
                # replace last assistant message with CURRENT APPROACH
                # The last message is assistant. Replace its content:
                if len(convs[idx]) > 0:
                    # Replace last assistant message content with approach
                    # Last message should be assistant reply from model1
                    # We'll find the last assistant:
                    for rev_i in range(len(convs[idx]) - 1, -1, -1):
                        if convs[idx][rev_i]['role'] == 'assistant':
                            convs[idx][rev_i]['content'] = approach
                            break
                    keep_convs.append(convs[idx])
        return keep_convs

    active_conversations1 = process_conversations_for_answers_and_filter(active_conversations1)

    # Check if we have >=5 answers
    if len(extracted_answers) >= 5:
        return select_answer(extracted_answers)

    # Steps 4 through 17: pattern described:
    # 4) 
    # 4a) run batch generate on all in active_conversations1 (model1)
    # 4b) do same as 3b but with active_conversations2
    # 5) same as 4 but flipped (active_conversations2 generation and active_conversations1 analysis)
    # 6 same as 4, etc...
    # Continue this pattern up to 17.
    # Pattern seems: even steps - generate on set1, analyze set2; odd steps (after initial) - generate on set2, analyze set1.
    # We'll define a helper function to do these steps.

    def generation_step(convs, model):
        if not convs:
            return convs
        return batch_conversations_generate(convs, tokenizer, input_queue1, output_queue1, input_queue2, output_queue2, model=model, max_tokens=2048)

    def analysis_step(convs):
        return process_conversations_for_answers_and_filter(convs)

    # We will run from step 4 to step 17:
    # Steps:
    # 4: generate on active_conversations1 (model1), analyze active_conversations2
    # 5: generate on active_conversations2 (model1), analyze active_conversations1
    # 6: generate on active_conversations1 (model1), analyze active_conversations2
    # 7: generate on active_conversations2 (model1), analyze active_conversations1
    # ...
    # It keeps flipping. We have a total of 14 more steps (4 to 17 inclusive is 14 steps).
    # Pattern: even steps: generate on set1, analyze set2
    #          odd steps: generate on set2, analyze set1
    # We'll just implement a loop:
    # total rounds: steps 4 to 17 => 14 iterations
    # i in range(4,18)
    for step_num in range(4, 18):
        if step_num % 2 == 0:
            # even step: generate on set1, analyze set2
            active_conversations1 = generation_step(active_conversations1, 'model1')
            active_conversations2 = analysis_step(active_conversations2)
        else:
            # odd step: generate on set2, analyze set1
            active_conversations2 = generation_step(active_conversations2, 'model1')
            active_conversations1 = analysis_step(active_conversations1)

        # Check answers count
        if len(extracted_answers) >= 5:
            return select_answer(extracted_answers)

    # After finishing all steps, do final selection:
    return select_answer(extracted_answers)


if __name__ == '__main__':
    set_seed(42)
    warnings.simplefilter('ignore')

    model1_path = '/kaggle/input/m/shelterw/qwen2.5/transformers/qwq-32b-preview-awq/1'
    gpu_ids1 = '0,1'  # Assign GPUs 0 and 1 to the first model

    model2_path = '/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1'
    gpu_ids2 = '2,3'  # Assign GPUs 2 and 3 to the second model

    (input_queue1, output_queue1, p1), (input_queue2, output_queue2, p2) = initialize_workers(
        model1_path, gpu_ids1, model2_path, gpu_ids2
    )

    tokenizer = AutoTokenizer.from_pretrained(model1_path)

    # Example question
    question = "What is 123+456?"

    answer = predict_for_question(question, tokenizer, input_queue1, output_queue1, input_queue2, output_queue2)
    print("Final answer:", answer)

    terminate_workers(input_queue1, input_queue2, p1, p2)
