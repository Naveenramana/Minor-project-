from tensorflow.keras.models import load_model
import sys
import findspark
from pyspark.sql import SparkSession
from pyspark.ml.classification import *
from SpeechToText import SpeechToText
from PreprocessDataset import *
import threading
from queue import Queue
from multiprocessing import Process
from nltk.tokenize import word_tokenize
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

sys.path.append("G:\Dissertation_Project")
global stop_threads


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
    sc.setLogLevel("WARN")

    try:
        from src.CustonTransformers import FlattenTransformer
    except ImportError as e:
        print(f"Error importing FlattenTransformer: {e}")

    print("\n----------------------INITIALIZATION COMPLETED----------------------\n")

    return spark


def load_prediction_model(model_id):
    models = {
        "LogisticRegression_TFIDF": "G:\\Dissertation_Project\\src\\Models\\Trained_Models\\LogisticRegression\\bestModel",
        "RandomForest_TFIDF": "G:\\Dissertation_Project\\src\\Models\\Trained_Models\\RandomForest\\bestModel",
        "GradientBoosted_TFIDF": "G:\\Dissertation_Project\\src\\Models\\Trained_Models\\GradientBoostedTrees\\bestModel",
        "SupportVectorMachine_TFIDF": "G:\\Dissertation_Project\\src\\Models\\Trained_Models\\SupportVectorMachine\\bestModel",
        "NeuralNetwork_TFIDF": "G:\\Dissertation_Project\\src\Models\\Trained_Models\\NeuralNetwork_TFIDF\\NeuralNetwork_TFIDF.keras",
        "LSTM_NeuralNetwork_TFIDF": "G:\\Dissertation_Project\\src\\Models\\Trained_Models\\LSTM_NeuralNetwork_TFIDF\\LSTM_NeuralNetwork_TFIDF.keras"
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

            case "SupportVectorMachine_TFIDF":
                model = LinearSVCModel.load(model[model_id])
                return model

            case "NeuralNetwork_TFIDF":
                model = load_model(models[model_id])
                return model

            case "LSTM_NeuralNetwork_TFIDF":
                model = load_model(models[model_id])
                return model

            case _:
                model = LogisticRegressionModel.load(
                    models["LogisticRegression_TFIDF"])
                return model

    except FileNotFoundError as e:
        print(e)
        raise


def process_stream(private_key_file_path, output_queue, CHANNELS, RATE, device_index):

    stt = SpeechToText(private_key_file_path, CHANNELS, RATE, device_index)

    global stop_threads
    while not stop_threads:

        for transcript, non_modified_transcript in stt.recognize_speech_stream():

            # If transcript is a list of strings, join them into a single string
            if isinstance(transcript, list) and all(isinstance(s, str) for s in transcript):
                transcript = ' '.join(transcript)

            # Check if transcript is a non-empty string
            if isinstance(transcript, str) and len(transcript) > 0:
                transcript_words = word_tokenize(transcript)

            # Add the one-hot encoded vectors to the queue
            output_queue.put((transcript_words, non_modified_transcript))


if __name__ == "__main__":
    stop_threads = False

    # This map is for help only not for usage
    preprocessing_modes = {
        "TF_IDF": 1,
        "Neural_Network_TF_IDF": 2,
        "LSTM_Neural_Network_TF_IDF": 3
    }

    # Initializing spark and other things necessary
    # spark = initializer()

    private_key_file_path = 'Environment\speech-to-text.json'

    microphone_queue = Queue()
    loopback_queue = Queue()

    # model = load_prediction_model("LogisticRegression_TFIDF")
    preprocessing_mode = 1

    ################################################## TESTING ###########################################################

    # Start the SpeechToText thread for the first stream.
    # stt_thread_microphone = threading.Thread(target=process_stream, args=(
    #     private_key_file_path, microphone_queue, 1, 44100, 1))
    # stt_thread_microphone.daemon = True
    # stt_thread_microphone.start()
    # print("Opening MIC Thread -- SUCCESS\n")

    # Start the SpeechToText thread for the second stream.
    stt_thread_loopback = threading.Thread(target=process_stream, args=(
        private_key_file_path, loopback_queue, 2, 48000, 1))
    stt_thread_loopback.daemon = True
    stt_thread_loopback.start()
    print("Opening LOOP Thread -- SUCCESS\n")

    try:
        while stt_thread_loopback.is_alive():
            # stt_thread_microphone.is_alive() and
            # Check if there is a preprocessed transcript in the queue
            # if not microphone_queue.empty():
            #     (transcript_mic, non_modified_transcript_mic) = microphone_queue.get()
            #     print("Microphone preprocessed transcript:\n", transcript_mic)
            #     print("Microphone bare transcript:",
            #           non_modified_transcript_mic)

            if not loopback_queue.empty():
                (transcript_loop, non_modified_transcript_loop) = loopback_queue.get()
                print("Loopback preprocessed transcript:\n", transcript_loop)
                print("Loopback bare transcript:",
                      non_modified_transcript_loop)

            ############################### Further Preprocessing of choice. ##################################

            if (preprocessing_mode == 1):
                # TF - IDF
                pass
            elif (preprocessing_mode == 2):
                # Neural Network specific together with TF - IDF
                pass

    except KeyboardInterrupt:
        print("Stopping...")
        stop_threads = True
        # stt_thread_microphone.join()
        stt_thread_loopback.join()
        # try:
        #     # Attempt to stop Spark
        #     spark.stop()
        #     print("Spark stopped successfully.")
        # except Exception as e:
        #     # Handle the error or ignore it
        #     print("Error stopping Spark:", e)
        print("Threads have been stopped and joined successfully.")
