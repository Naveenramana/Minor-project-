from SpeechToText import SpeechToText
from PreprocessDataset import *
import threading
from queue import Queue
from nltk.tokenize import word_tokenize
import nltk
from pyspark.ml.classification import *
from pyspark.sql import SparkSession
import findspark


def initializer():

    print("\n\n-----------------------INITIALIZATION RUNNING----------------------\n")

    findspark.init()
    print("Spark location => " + findspark.find())

    # Initialize SparkSession with necessary configurations
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName('Spark') \
        .config("spark.driver.memory", "15g") \
        .config("spark.hadoop.home.dir", "H:/HADOOP/") \
        .config("spark.hadoop.conf.dir", "H:/HADOOP/etc/hadoop/") \
        .getOrCreate()

    # Get SparkContext from the SparkSession
    sc = spark.sparkContext

    print("\n----------------------INITIALIZATION COMPLETED----------------------\n")


def load_prediction_model(model_id):
    models = {
        "LogisticRegression_TFIDF": "G:\\Dissertation_Project\\src\\Models\\Trained_Models\\LogisticRegression\\bestModel",
        "RandomForest_TFIDF": "G:\\Dissertation_Project\\src\\Models\\Trained_Models\\RandomForest\\bestModel",
        "GradientBoosted_TFIDF": "G:\\Dissertation_Project\\src\\Models\\Trained_Models\\GradientBoostedTrees\\bestModel"
    }

    print("<--LOADING PREDICTION MODEL : {} , From location : {}-->\n".format(
        model_id, models[model_id]))

    if not isinstance(model_id, str):
        raise TypeError(model_id + " must be of type str.")

    if not model_id in models.keys():
        raise ValueError("model_id " + model_id + " does not exist.")

    try:
        match (model_id):
            case "LogisticRegression_TFIDF":
                model = LogisticRegressionModel.load(models[model_id])
                return model

            case "RandomForest_TFIDF":
                model = RandomForestClassificationModel.load(models[model_id])
                return model

            case "GradientBoosted_TFIDF":
                model = GBTClassificationModel.load(models[model_id])
                return model

            case _:
                model = LogisticRegressionModel.load(
                    models["LogisticRegression_TFIDF"])
                return model

    except FileNotFoundError as e:
        print(e)
        raise


def process_stream(private_key_file_path, processed_transcripts, preprocessing_mode):

    stt = SpeechToText(private_key_file_path, 1)

    for transcript, non_modified_transcript in stt.recognize_speech_stream():

        print(transcript)
        # If transcript is a list of strings, join them into a single string
        if isinstance(transcript, list) and all(isinstance(s, str) for s in transcript):
            transcript = ' '.join(transcript)

        # Check if transcript is a non-empty string
        if isinstance(transcript, str) and len(transcript) > 0:
            transcript_words = word_tokenize(transcript)

        ############################### Further Preprocessing of choice. ##################################

        # Add the one-hot encoded vectors to the queue
        processed_transcripts.put((transcript_words, non_modified_transcript))


if __name__ == "__main__":

    # This map is for help only not for usage
    preprocessing_modes = {
        "TF-IDF": 1
    }

    # Initializing spark and other things necessary
    initializer()

    private_key_file_path = 'Environment\speech-to-text.json'

    processed_transcripts = Queue()

    model = load_prediction_model("RandomForest_TFIDF")

    ################################################## TESTING ###########################################################

    ############################################## END OF TESTING ########################################################

    # Start the SpeechToText thread
    # stt_thread = threading.Thread(target=process_stream, args=(
    #     private_key_file_path, processed_transcripts, 1))
    # stt_thread.start()

    # try:
    #     while stt_thread.is_alive():
    #         # Check if there is a preprocessed transcript in the queue
    #         if not processed_transcripts.empty():
    #             (transcript, non_modified_transcript) = processed_transcripts.get()
    #             print("Received preprocessed transcript:\n", transcript)
    #             print("Received non_modified_transcript:",
    #                   non_modified_transcript)

    # except KeyboardInterrupt:
    #     print("Stopping...")

    # stt_thread.join()
