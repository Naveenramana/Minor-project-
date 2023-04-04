import openai
import pandas
import threading
import numpy
import concurrent.futures


# Setting the api key for openai

openai.api_key = "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"


def call_openai_api(prompt, thread_id, conversation_type):
    chat_gpt_response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "user", "content": str(prompt)}]
    )
    
    message_response = chat_gpt_response['choices'][0]['message']['content']

    return (thread_id, message_response, conversation_type)
    

if __name__ == "__main__":
    thread_number = 2
    threads = []
    
    prompts = [
        "I want you to act like 2 people in a conversation with yourself. \
        One person is trying to scam the other and one is a normal human being on the phone. I want you to write 'attacker' \
        when the attacker is speaking and 'victim' when the victim is speaking. Give me the response in a json format, \
        every time the same person speaks in the same json give it a increasing numeric in the end of the key.",
        
        "I want you to act like 2 people in a conversation with yourself. \
        One person is trying to help the other (the opposite of voice phishing) and one is a normal human being on the phone. \
        I want you to write 'attacker' \
        when the helper is speaking and 'victim' when the recipient is speaking. Give me the response in a json format, \
        every time the same person speaks in the same json give it a increasing numeric in the end of the key."  
    ]
    
    prompts_dictionary = {"vishing": prompts[0], "normal": prompts[1]}
    
    responses = [None] * len(prompts)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(call_openai_api, prompts_dictionary[list(prompts_dictionary.keys())[i]], i, list(prompts_dictionary.keys())[i]): i for i in range(len(prompts_dictionary))}

        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            try:
                thread_id, response, conversation_type = future.result()
                responses[index] = (response, conversation_type)
            except Exception as e:
                print(f"Thread {index} failed: {e}")

    # Print the responses
    for i, (response, conversation_type) in enumerate(responses):
        print(f"Thread {i} Assistant: {response}, {conversation_type}")


