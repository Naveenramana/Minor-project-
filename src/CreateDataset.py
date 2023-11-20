import openai
import pandas
import concurrent.futures
import string
import json
import random
import asyncio


# Setting the api key for openai
with open("G:\Dissertation_Project\Environment\open-api-secret-key.json", "r") as key:
    key_data = json.load(key)
    openai.api_key = key_data["key"]


def generate_random_id(length=8):
    # Define the characters to use (digits and uppercase and lowercase letters)
    characters = string.ascii_letters + string.digits

    # Generate a random string of the specified length
    random_id = ''.join(random.choice(characters) for _ in range(length))

    return random_id


def call_openai_api(prompt, thread_id, conversation_type):
    chat_gpt_response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "user", "content": str(prompt)}]
    )

    message_response = chat_gpt_response['choices'][0]['message']['content']

    return (thread_id, message_response, conversation_type)


def store_conversation(conversation_json):

    # Retrieving the conversation from Chat GPT response.
    conversation = conversation_json['conversation']
    conversation_type = conversation_json['conversation_type']

    existing_csv_file = "Data\Custom_Datasets\conversation_datasets_GPT.csv"
    existing_dataframe = pandas.read_csv(existing_csv_file)
    existing_dataframe_columns = existing_dataframe.columns

    # Creating a random id for this conversation
    conversation_id = generate_random_id(length=14)

    new_conversation_dataframe = form_new_dataframe(
        conversation_id, conversation, existing_dataframe_columns, conversation_type)

    updated_dataframe = pandas.concat(
        [existing_dataframe, new_conversation_dataframe], ignore_index=False)
    updated_dataframe.to_csv(path_or_buf=existing_csv_file, index=False)


def form_new_dataframe(conversation_id, conversation, existing_dataframe_columns, conversation_type):

    attacker_helper_sentence_list = []
    victim_sentence_list = []

    # Converting string of json into json
    conversation = json.loads(conversation)

    for iteration in conversation.keys():
        iteration_json = conversation[iteration]

        attacker_helper_sentence = iteration_json[
            "Attacker"] if conversation_type == "vishing" else iteration_json["Helper"]

        victim_sentence = iteration_json["Victim"]

        attacker_helper_sentence_list.append(attacker_helper_sentence)
        victim_sentence_list.append(victim_sentence)

    # Creating the binary representation of vishing or not vishing

    conversation_type_numeric = 1 if conversation_type == "vishing" else 0

    data = [conversation_id, attacker_helper_sentence_list,
            victim_sentence_list, conversation_type_numeric]

    new_row = {column: value for column, value in zip(
        existing_dataframe_columns, data)}

    conversation_dataframe = pandas.DataFrame(
        new_row, columns=existing_dataframe_columns)
    return conversation_dataframe


def execute_functionality():
    prompts = [
        'I want you to act like 2 people in a conversation with yourself.  \
        I want you to only give me the conversation. \
        The conversation will include a person which is trying to phone scam the other person. \
        A normal human being of an age which would probably be targeted by phone scammers. \
        I want the numbers to match when one answers to the other. I want those numbers to express how the conversation went back and forth. \
        I want the whole conversation to be at the most 10 sentences long. \
        I want this formatting for the answer in JSON: { "1": "{"Attacker": "Sentence", "Victim": "Sentence"}", "2": "{"Attacker": "Sentence", "Victim": "Sentence"}"} \
        I want you to be careful and give me all the json answers in correct json format (with double quotes). \
        I want you to be extra careful to not miss any quotes that might mess the json \
        I want you to avoid writing descriptions of physical or psychological actions the people in the conversation might do.',

        'I want you to act like 2 people in a conversation with yourself.  \
        I want you to only give me the conversation. \
        The conversation will include a person which is calling for some legitimate reason from a bank or another institution that is generally the reason phone scammers call victims. \
        A normal human being of an age which would probably be targeted by phone scammers. \
        I want the numbers to match when one answers to the other. I want those numbers to express how the conversation went back and forth. \
        I want the whole conversation to be at the most 10 sentences long. \
        I want this formatting for the answer in JSON: { "1": "{"Helper": "Sentence", "Victim": "Sentence"}", "2": "{"Helper": "Sentence", "Victim": "Sentence"}"} \
        I want you to be careful and give me all the json answers in correct json format (with double quotes).\
        I want you to be extra careful to not miss any quotes that might mess the json \
        I want you to avoid writing descriptions of physical or psychological actions the people in the conversation might do.'

    ]
    try:
        prompts_dictionary = {"vishing": prompts[0], "normal": prompts[1]}

        responses = [None] * len(prompts)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(call_openai_api, prompts_dictionary[list(prompts_dictionary.keys())[
                i]], i, list(prompts_dictionary.keys())[i]): i for i in range(len(prompts_dictionary))}

            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                try:
                    thread_id, response, conversation_type = future.result()
                    responses[index] = (response, conversation_type)
                except Exception as e:
                    print(f"Thread {index} failed: {e}")

        # Print the responses
        for i, (response, conversation_type) in enumerate(responses):
            # print(f"Thread {i} Assistant: {response}, {conversation_type}")

            temp_json = {"conversation": response,
                         "conversation_type": conversation_type}

            store_conversation(temp_json)

        print({
            "statusCode": 200,
            "statusText": "Success",
            "message": "Workflow completed successfully"
        })
    except Exception as e:
        print({
            "statusCode": 500,
            "statusText": "Fail",
            "message": str(e)
        })


async def call_execute_functionality():
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, execute_functionality)


async def functionality_runner(threads):
    tasks = [call_execute_functionality()
             for _ in range(threads)]
    await asyncio.gather(*tasks)


if __name__ == "__main__":

    threads = 4

    asyncio.run(functionality_runner(threads))
