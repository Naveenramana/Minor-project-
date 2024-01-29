import pandas as pd
# nltk.download('punkt')
# nltk.download('stopwords')


def read_csv_and_return_info(file_path):
    # Read the CSV file and store it in a pandas DataFrame
    df = pd.read_csv(file_path)

    # Get the columns of the dataset
    columns = df.columns

    # Get the size of the dataset (number of rows and columns)
    dataset_size = df.shape
    return df, columns, dataset_size


def search_dataframe(df, column_name, specific_value, specific_columns=None):
    if specific_columns is None:
        specific_columns = []

    # Filter the dataframe based on the specific value in the specified column
    filtered_df = df[df[column_name] == specific_value]

    result_dict = {}

    for column in filtered_df.columns:
        if column in specific_columns:
            result_dict[column] = filtered_df[column].tolist()
        else:
            unique_value = filtered_df[column].unique()
            result_dict[column] = unique_value[0] if len(
                unique_value) == 1 else unique_value.tolist()

    return result_dict


def execute_complete_enhance_workflow(csv_file_path, output_file_path):
    dataframe, columns, size = read_csv_and_return_info(csv_file_path)

    # Setting the conversation columns to be used further down the road.
    conversation_columns = ['Attacker_Helper', 'Victim']

    result_data = []

    for conversation_id in dataframe['Conversation_ID'].unique():

        # Searching for the exact conversation_dictionary
        conversation_dictionary = search_dataframe(
            dataframe, 'Conversation_ID', conversation_id, conversation_columns)

        # print("Conversation Dictionary -> {}".format(conversation_dictionary))
        # print("\n")

        # Creating New Lists for each iteration
        attacker_helper_temp_list = []
        victim_temp_list = []

        counter = 0

        for attacker_helper_s, victim_s in zip(conversation_dictionary['Attacker_Helper'], conversation_dictionary['Victim']):
            conversation_data = {}

            attacker_helper_temp_list.append(str(attacker_helper_s))
            victim_temp_list.append(str(victim_s))

            conversation_data['Conversation_ID'] = conversation_dictionary['Conversation_ID'] + \
                "_" + str(counter)

            conversation_data['Attacker_Helper'] = attacker_helper_temp_list.copy()

            conversation_data['Victim'] = victim_temp_list.copy()

            conversation_data['Conversation_Type'] = conversation_dictionary['Conversation_Type']

            counter += 1

            result_data.append(conversation_data)

    # Convert result_data to a DataFrame
    result_dataframe = pd.DataFrame(result_data, columns=columns)
    result_dataframe['Conversation_Type'] = result_dataframe['Conversation_Type'].astype(
        int)
    # Save the enhanced DataFrame to a CSV file
    result_dataframe.to_csv(output_file_path, index=False)

    return result_dataframe


if __name__ == '__main__':
    file_path = 'Data/Custom_Datasets/conversation_datasets_GPT.csv'
    output_file_path = 'Data/Custom_Datasets/conversations_dataset_enhanced_GPT.csv'
    resulting_dataframe = execute_complete_enhance_workflow(
        file_path, output_file_path=output_file_path)
