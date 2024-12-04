import os
import multiprocessing
import warnings
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

        # Define sampling parameters
        sampling_params = SamplingParams(
            temperature=0,
            min_p=0.01,
            skip_special_tokens=True,
            max_tokens=32768 if 'qwq-32b-preview-awq' in model_path else 8192,
        )

        while True:
            # Wait for a batch of prompts from the input queue
            batch = input_queue.get()
            if batch == "TERMINATE":
                break  # Exit the loop to terminate the worker

            if not isinstance(batch, list):
                # If a single prompt is sent instead of a list, make it a list
                batch = [batch]

            # Perform inference
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

def batch_conversations_generate(conversations, tokenizer, input_queue1, output_queue1, 
                                 input_queue2, output_queue2, model='model1') -> list:
    """
    Generate responses for a batch of conversations using the specified model.

    Args:
        conversations (list[list[dict]]): A list of conversations, where each conversation is a list of messages (dicts with 'role' and 'content').
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to prepare prompts.
        input_queue1, output_queue1: Queues for model1.
        input_queue2, output_queue2: Queues for model2.
        model (str): 'model1' or 'model2'. Determines which model to use.

    Returns:
        list[list[dict]]: The updated conversations with assistant responses appended.
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

    responses = []

    if model == 'model1':
        # Send to model1
        input_queue1.put(conversations_texts)
        output1 = output_queue1.get()
        responses.extend(output1)

    elif model == 'model2':
        # Send to model2
        input_queue2.put(conversations_texts)
        output2 = output_queue2.get()
        responses.extend(output2)

    # Append the responses to the respective conversations
    response_index = 0
    for conversation in conversations:
        conversation.append({'role': 'assistant', 'content': responses[response_index]})
        response_index += 1

    return conversations

if __name__ == '__main__':
    set_seed(42)
    warnings.simplefilter('ignore')

    # Define model paths and assign GPUs
    model1_path = '/kaggle/input/m/shelterw/qwen2.5/transformers/qwq-32b-preview-awq/1'
    gpu_ids1 = '0,1'  # Assign GPUs 0 and 1 to the first model

    model2_path = '/kaggle/input/m/qwen-lm/qwen2.5/transformers/32b-instruct-awq/1'
    gpu_ids2 = '2,3'  # Assign GPUs 2 and 3 to the second model

    # Initialize worker processes
    (input_queue1, output_queue1, p1), (input_queue2, output_queue2, p2) = initialize_workers(
        model1_path, gpu_ids1, model2_path, gpu_ids2
    )

    # Initialize a tokenizer (assuming both models use the same tokenizer)
    # If different, initialize separate tokenizers for each model
    tokenizer = AutoTokenizer.from_pretrained(model1_path)

    #WRITE CODE HERE!!!

    #==========


    #==========

    # Terminate worker processes gracefully
    #terminate_workers(input_queue1, input_queue2, p1, p2)