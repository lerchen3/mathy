import os
import multiprocessing
import warnings
from transformers import set_seed, AutoTokenizer
from vllm import AsyncEngine, AsyncLLMEngine
import re
from typing import List, Tuple
import statistics
import time
import asyncio

def extract_boxed_text(text: str) -> str:
    """
    Extracts the content within \boxed{} from the given text.
    """
    match = re.search(r'\\boxed\{([^}]+)\}', text)
    if match:
        return match.group(1).strip()
    return ""

def model_worker(model_path, gpu_ids, input_queue, output_queue):
    """
    Worker function to load a model and handle batch generation requests.
    """
    # Set environment variables specific to this process
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    try:
        # Initialize AsyncEngine instead of LLM
        engine = AsyncEngine(
            model_path,
            dtype="half",
            max_num_seqs=64,
            max_model_len=16384,
            trust_remote_code=True,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.96,
        )

        # Get the tokenizer
        tokenizer = engine.get_tokenizer()

        # Define sampling parameters
        sampling_params = SamplingParams(
            temperature=0,
            min_p=0.01,
            skip_special_tokens=True,
            max_tokens=16384
        )

        async def process_batch(batch):
            if not isinstance(batch, list):
                batch = [batch]

            # Record prompt processing time
            start_time = time.time()
            generator = await engine.generate(batch, sampling_params)
            prompt_time = time.time() - start_time
            
            # Track token generation times
            token_times = []
            outputs = []
            
            async for request_output in engine.stream_generate(batch, sampling_params):
                token_start = time.time()
                outputs.append(request_output)
                token_times.append(time.time() - token_start)

            final_texts = [output.outputs[0].text for output in outputs]
            return {
                'texts': final_texts,
                'prompt_time': prompt_time,
                'token_times': token_times
            }

        while True:
            # Wait for a batch of prompts from the input queue
            batch = input_queue.get()
            if batch == "TERMINATE":
                break  # Exit the loop to terminate the worker

            # Run the async function in the event loop
            result = asyncio.run(process_batch(batch))
            output_queue.put(result)

    except Exception as e:
        # In case of error, put the exception message into the output queue
        output_queue.put({'error': str(e)})

def initialize_workers(model1_path, gpu_ids1, model2_path, gpu_ids2):
    """
    Initializes and starts worker processes for both models.
    Returns the input and output queues for both models.
    """
    # Create multiprocessing queues for communication
    input_queue1 = multiprocessing.Queue()
    output_queue1 = multiprocessing.Queue()

    input_queue2 = multiprocessing.Queue()
    output_queue2 = multiprocessing.Queue()

    # Create multiprocessing processes for each model
    p1 = multiprocessing.Process(target=model_worker, args=(model1_path, gpu_ids1, input_queue1, output_queue1))
    p2 = multiprocessing.Process(target=model_worker, args=(model2_path, gpu_ids2, input_queue2, output_queue2))

    # Start the processes
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

def batch_conversations_generate(conversations, tokenizer, input_queue1, output_queue1, input_queue2, output_queue2, model='model1') -> Tuple[list, float, List[float]]:
    """
    Generate responses for a batch of conversations using the specified model.

    Args:
        conversations (list[list[dict]]): A list of conversations, where each conversation is a list of messages.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to prepare prompts.
        input_queue1 (Queue): Input queue for model1.
        output_queue1 (Queue): Output queue for model1.
        input_queue2 (Queue): Input queue for model2.
        output_queue2 (Queue): Output queue for model2.
        model (str): 'model1' or 'model2'. Determines which model to use. Defaults to 'model1'.

    Returns:
        tuple[list[list[dict]], float, List[float]]: The updated conversations with assistant responses appended, prompt time, and token generation times.
    """
    # Prepare the prompts by applying the tokenizer's chat template
    conversations_texts = [
        tokenizer.apply_chat_template(
            conversation=conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        for conversation in conversations
    ]

    if model == 'model1':
        # Send to model1
        input_queue1.put(conversations_texts)
        output = output_queue1.get()
    elif model == 'model2':
        # Send to model2
        input_queue2.put(conversations_texts)
        output = output_queue2.get()

    if 'error' in output:
        raise Exception(output['error'])

    responses = output['texts']
    prompt_time = output['prompt_time']
    token_times = output['token_times']

    # Append the responses to the respective conversations
    response_index = 0
    for conversation in conversations:
        conversation.append({'role': 'assistant', 'content': responses[response_index]})
        response_index += 1

    return conversations, prompt_time, token_times

if __name__ == '__main__':
    set_seed(42)
    warnings.simplefilter('ignore')

    # Define model paths and assign GPUs
    model1_path = '/kaggle/input/m/shelterw/qwen2.5/transformers/qwq-32b-preview-awq/1'
    gpu_ids1 = '0,1'  # Assign GPUs 0 and 1 to the first model

    model2_path = '/kaggle/input/qwen2.5/transformers/32b-instruct-awq/1'
    gpu_ids2 = '2,3'  # Assign GPUs 2 and 3 to the second model

    # Initialize worker processes
    (input_queue1, output_queue1, p1), (input_queue2, output_queue2, p2) = initialize_workers(
        model1_path, gpu_ids1, model2_path, gpu_ids2
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model1_path)

    question_text = 'A peacock is a ten-digit positive integer that uses each digit exactly once. Compute the number of peacocks that are exactly twice another peacock.'
    batch_sizes = [1, 2, 4, 6, 8, 10, 12, 14, 16]
    results = []

    for batch_size in batch_sizes:
        prompt_times = []
        all_token_times = []
        
        # Run 3 times for each batch size to get average
        for _ in range(3):
            _, prompt_time, token_times = batch_conversations_generate(
                conversations=[
                    [{'role': 'assistant', 
                      'content': question_text
                     }] for _ in range(batch_size)
                ],
                tokenizer=tokenizer,
                input_queue1=input_queue1,
                output_queue1=output_queue1,
                input_queue2=input_queue2,
                output_queue2=output_queue2,
                model='model1'
            )
            prompt_times.append(prompt_time)
            all_token_times.append(token_times)

        avg_prompt_time = statistics.mean(prompt_times)
        avg_token_time = statistics.mean([t for times in all_token_times for t in times])
        
        results.append({
            'batch_size': batch_size,
            'avg_prompt_time': avg_prompt_time,
            'avg_token_time': avg_token_time,
            'tokens_per_second': 1 / avg_token_time
        })

    # Print results in a formatted table
    print("\nBatch Processing Results:")
    print("-" * 80)
    print(f"{'Batch Size':^10} | {'Prompt Time (s)':^15} | {'Token Time (s)':^15} | {'Tokens/Second':^15}")
    print("-" * 80)
    for r in results:
        print(f"{r['batch_size']:^10} | {r['avg_prompt_time']:^15.4f} | {r['avg_token_time']:^15.4f} | {r['tokens_per_second']:^15.2f}")

    # Terminate worker processes gracefully
    terminate_workers(input_queue1, input_queue2, p1, p2)