import pandas as pd
import re
import nltk
# nltk.download('punkt')
nltk.download('stopwords')
from greek_stemmer import stemmer
from nltk.stem import PorterStemmer
from langdetect import detect
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



def read_csv_and_return_info(file_path):
    # Read the CSV file and store it in a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Get the columns of the dataset
    columns = df.columns
    
    # Get the size of the dataset (number of rows and columns)
    dataset_size = df.shape
    return df, columns, dataset_size


def search_dataframe(df, column_name, specific_value, specific_columns=None):

    if specific_columns == None:
        specific_columns = []
        
    # Filter the dataframe based on the specific value in the specified column
    filtered_df = df[df[column_name] == specific_value]
    
    result_dict = {}
    
    for column in filtered_df.columns:
        if column in specific_columns:
            result_dict[column] = filtered_df[column].tolist()
        else:
            unique_value = filtered_df[column].unique()
            result_dict[column] = unique_value[0] if len(unique_value) == 1 else unique_value.tolist()

    return result_dict


def preprocess_strings(input_data):
    # Replace numbers with 'X' and remove all non-letter characters, excluding spaces
    def process_string(s):
        s = re.sub(r'\d', 'X', s)
        s = ''.join(c for c in s if c.isalpha() or c == 'X' or c == ' ')
        s = s.lower()
        return s

    # Check if the input is a single string or a list of strings
    if isinstance(input_data, str):
        return process_string(input_data)
    elif isinstance(input_data, list):
        return [process_string(s) for s in input_data]
    else:
        raise TypeError("Input data must be a string or a list of strings")


def stem_strings(input_data):

    english_stemmer = PorterStemmer()

    def stem_string(s):
        language = detect(s)
        words = nltk.word_tokenize(s)
        
        if language == 'en':
            stemmed_words = [english_stemmer.stem(word).lower() for word in words]
        elif language == 'el':
            stemmed_words = [stemmer.stem_word(word, 'vbg').lower() for word in words]
        else:
            raise ValueError("Unsupported language detected")
        
        return ' '.join(stemmed_words)

    if isinstance(input_data, str):
        return stem_string(input_data)
    elif isinstance(input_data, list):
        return [stem_string(s) for s in input_data]
    else:
        raise TypeError("Input data must be a string or a list of strings")


def remove_stopwords(input_data):

    # Define stopwords for both languages
    english_stopwords = set(stopwords.words("english"))
    greek_stopwords = set(stopwords.words("greek"))

    def process_text(text):
        language = detect(text)
        if language == "en":
            stopwords_set = english_stopwords
        elif language == "el":
            stopwords_set = greek_stopwords
        else:
            raise ValueError("Unsupported language detected")

        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in stopwords_set]

        return " ".join(filtered_words)

    if isinstance(input_data, str):
        return process_text(input_data)
    elif isinstance(input_data, list):
        return [process_text(text) for text in input_data]
    else:
        raise TypeError("Input data must be a string or a list of strings")


def execute_complete_preprocess_workflow(csv_file_path):

    # file_path = 'Data\Custom_Datasets\Test_dataset.csv'
    dataframe, columns, size = read_csv_and_return_info(csv_file_path)
    
    # Setting the conversation columns to be used further down the road.
    conversation_columns = ['Attacker_Helper', 'Victim']

    # For each conversation we preprocess the conversation and we store it in a dictionary
    for conversation_id in dataframe['Conversation_ID'].unique():
        conversation_dictionary = search_dataframe(dataframe, 'Conversation_ID', conversation_id, conversation_columns)
        
        for conversation_column in conversation_columns:
            # Removing special characters and numbers
            conversation_dictionary[conversation_column] = preprocess_strings(conversation_dictionary[conversation_column])
            # Removing stopwords
            conversation_dictionary[conversation_column] = remove_stopwords(conversation_dictionary[conversation_column])
            # Executing stemming
            conversation_dictionary[conversation_column] = stem_strings(conversation_dictionary[conversation_column])
            
        
    return pd.DataFrame.from_dict(conversation_dictionary)
        

if __name__ == '__main__':

    file_path = 'Data\Custom_Datasets\Test_dataset.csv'
    dataframe, columns, size = read_csv_and_return_info(file_path)
    
    # Setting the conversation columns to be used further down the road.
    conversation_columns = ['Attacker_Helper', 'Victim']

    # For each conversation we preprocess the conversation and we store it in a dictionary
    for conversation_id in dataframe['Conversation_ID'].unique():
        conversation_dictionary = search_dataframe(dataframe, 'Conversation_ID', conversation_id, conversation_columns)
        
        for conversation_column in conversation_columns:
            # Removing special characters and numbers
            conversation_dictionary[conversation_column] = preprocess_strings(conversation_dictionary[conversation_column])
            # Removing stopwords
            conversation_dictionary[conversation_column] = remove_stopwords(conversation_dictionary[conversation_column])
            # Executing stemming
            conversation_dictionary[conversation_column] = stem_strings(conversation_dictionary[conversation_column])
            
        
        print(pd.DataFrame.from_dict(conversation_dictionary))
        
    
    