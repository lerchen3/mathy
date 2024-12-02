import threading
import time

class TimeoutError(Exception):
    """Custom exception to indicate a timeout."""
    pass

def batch_message_generate(list_of_messages) -> list[list[dict]]:
    timeout_event = threading.Event()

    def timeout():
        print("Operation timed out after 8 minutes.")
        timeout_event.set()

    timer = threading.Timer(480, timeout)  # 480 seconds = 8 minutes
    timer.start()
    
    try:
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
            if timeout_event.is_set():
                raise TimeoutError("batch_message_generate timed out.")
            messages.append({'role': 'assistant', 'content': single_request_output.outputs[0].text})
            print(messages[-1])

    except TimeoutError as te:
        print(te)
        # Handle the timeout accordingly, e.g., clean up resources or return partial results
        # For example, you might return the messages processed so far
    finally:
        timer.cancel()
    
    return list_of_messages