from SpeechToText import SpeechToText
from PreprocessDataset import *
from multiprocessing import Queue
from multiprocessing import Process
from threading import Thread
import findspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.ml.classification import *
from pyspark.ml import PipelineModel
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from scipy.sparse import hstack
import pandas
import socket
import logging
import json
import sys

sys.path.append("G:\Dissertation_Project")


# Setting up logging
log_format = "%(asctime)s - %(name)s - %(message)s"
logging.basicConfig(filename="G:\\Dissertation_Project\\Logs\\performance_logs.log",
                    level=logging.INFO, format=log_format)
logger = logging.getLogger("Dissertation_Project")
logger.setLevel(logging.INFO)

py4j_logger = logging.getLogger('py4J')
py4j_logger.setLevel(logging.ERROR)


def initializer():

    print("\n\n-----------------------INITIALIZATION RUNNING----------------------\n")

    # Finding spark
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

    print("\n----------------------INITIALIZATION COMPLETED----------------------\n")

    return spark


def load_prediction_model(model_id):
    models = {
        "LogisticRegression_TFIDF": "G:\\Dissertation_Project\\src\\Models\\Trained_Models\\LogisticRegression\\bestModel",
        "RandomForest_TFIDF": "G:\\Dissertation_Project\\src\\Models\\Trained_Models\\RandomForest\\bestModel",
        "GradientBoosted_TFIDF": "G:\\Dissertation_Project\\src\\Models\\Trained_Models\\GradientBoostedTrees\\bestModel",
        "SupportVectorMachine_TFIDF": "G:\\Dissertation_Project\\src\\Models\\Trained_Models\\SupportVectorMachine\\bestModel",
        "NeuralNetwork_TFIDF": "G:\\Dissertation_Project\\src\Models\\Trained_Models\\NeuralNetwork_TFIDF\\NeuralNetwork_TFIDF.keras",
        "LSTM_NeuralNetwork_TFIDF": "G:\\Dissertation_Project\\src\\Models\\Trained_Models\\LSTM_NeuralNetwork_TFIDF\\LSTM_NeuralNetwork_TFIDF.keras",
        "NeuralNetwork_TEST_TFIDF": "G:\\Dissertation_Project\\src\Models\\Trained_Models\\NeuralNetwork_TFIDF\\NN_TEST_TFIDF.keras"
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
                model = LinearSVCModel.load(models[model_id])
                return model

            case "NeuralNetwork_TFIDF":
                from src.CustomNNMetrics import F1Score
                model = load_model(models[model_id], custom_objects={
                                   'F1Score': F1Score})
                return model

            case "LSTM_NeuralNetwork_TFIDF":
                from src.CustomNNMetrics import F1Score
                model = load_model(models[model_id], custom_objects={
                                   'F1Score': F1Score})
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

    for transcript, non_modified_transcript in stt.recognize_speech_stream():
        # If transcript is a list of strings, join them into a single string
        if isinstance(transcript, list) and all(isinstance(s, str) for s in transcript):
            transcript = ' '.join(transcript)
        # Check if transcript is a non-empty string
        if isinstance(transcript, str) and len(transcript) > 0:
            transcript_words = word_tokenize(transcript)

        stemmed_words = stem_strings(transcript_words, 'en')
        output_queue.put(stemmed_words)


def run_process_stream(private_key_file_path, output_queue, CHANNELS, RATE, device_index):
    try:
        process_stream(private_key_file_path, output_queue,
                       CHANNELS, RATE, device_index)
    except KeyboardInterrupt:
        print(f"Stopping process for device {device_index}")
    except Exception as e:
        print(f"Exception in process for device {device_index}: {e}")


def connect_to_middleware_server(host, port):
    socket_connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    socket_connection.connect((host, port))
    return socket_connection


def send_data_to_server(socket_connection, data):
    message = json.dumps(data) + '\n'
    socket_connection.sendall(message.encode())


def initiate_spark_streaming(preprocessing_mode):

    # Do not open connection with middleware server if preprocessing mode is not 1 since streaming will not be required
    if (preprocessing_mode != 1):
        return

    try:
        # Initializing spark and other things necessary
        spark = initializer()

        # Loading the preprocessing pipeline
        pipeline_path = "G:\\Dissertation_Project\\src\\Models\\Pipelines\\Prediction_Pipeline"
        pipeline_model = PipelineModel.load(pipeline_path)

        # Loading the prediction model
        prediction_model = load_prediction_model("GradientBoosted_TFIDF")
        logger.info("Using prediction model: {}".format(prediction_model))

        # Define the schema of the data
        schema = StructType([
            StructField("Attacker_Helper", ArrayType(StringType())),
            StructField("Victim", ArrayType(StringType()))
        ])

        raw_stream_df = spark.readStream \
            .format("socket") \
            .option("host", "localhost") \
            .option("port", 9999) \
            .load()

        # Parse the JSON strings
        parsed_df = raw_stream_df.select(
            from_json(col("value"), schema).alias("data")).select("data.*")

        def process_and_log_batch(batch_df, batch_id):
            logger.info(
                f"Batch ID: {batch_id}, --Data has been received--")
            if batch_df:
                # Log the batch
                batch_string = batch_df._jdf.showString(1, 50, False)
                logger.info(f"Batch ID: {batch_id}, Data:\n{batch_string}")
                # Print to console
                batch_df.show(truncate=False)

        preprocessed_df = pipeline_model.transform(parsed_df)

        # Prediction
        predictions = prediction_model.transform(
            preprocessed_df).select("Prediction", "probability")
        query = predictions.writeStream \
            .outputMode("append") \
            .foreachBatch(process_and_log_batch) \
            .start()

        # Await termination to keep the streaming application running
        query.awaitTermination()

    except KeyboardInterrupt:
        print(
            "Keyboard Interrupt received. Stopping the streaming queries and Spark session.")
        if query:
            query.stop()

        if spark:
            spark.stop()


def run_main_application(preprocessing_mode):
    private_key_file_path = 'Environment\speech-to-text.json'

    microphone_queue = Queue()
    loopback_queue = Queue()

    process_microphone = Process(target=run_process_stream, args=(
        private_key_file_path, microphone_queue, 1, 44100, 1,))

    process_loopback = Process(target=run_process_stream, args=(
        private_key_file_path, loopback_queue, 1, 44100, 2,))

    process_microphone.start()
    process_loopback.start()

    if (preprocessing_mode == 1):
        # Connect to middleware server
        socket_connection = connect_to_middleware_server('localhost', 9999)

        while True:
            if (not microphone_queue.empty()) and (not loopback_queue.empty()):
                microphone_data = microphone_queue.get()
                loopback_data = loopback_queue.get()

                # constructing the request object
                data_for_prediction = {
                    "Attacker_Helper": loopback_data,
                    "Victim": microphone_data
                }

                send_data_to_server(
                    socket_connection=socket_connection, data=data_for_prediction)

                logger.info("Sending data to Middleware server. Data: {}".format(
                    data_for_prediction))

    elif (preprocessing_mode == 2):

        prediction_model = load_prediction_model("NeuralNetwork_TFIDF")
        logger.info("Using prediction model: {}".format(prediction_model))

        while True:
            if not microphone_queue.empty() and not loopback_queue.empty():
                microphone_data = microphone_queue.get()
                loopback_data = loopback_queue.get()

                data_for_logs = {
                    "attacker_helper": loopback_data,
                    "victim": microphone_data
                }

                logger.info(
                    "Received: {}, implementing prediction.".format(data_for_logs))

                # Concatenate texts
                concatenated_microphone_data = ' '.join(microphone_data)
                concatenated_loopback_data = ' '.join(loopback_data)

                # Handle preprocessing for the neural network
                num_features = 200

                # Initialize HashingVectorizer to do the hashing trick
                hashing_vectorizer_ah = HashingVectorizer(
                    n_features=num_features, alternate_sign=False)
                hashing_vectorizer_v = HashingVectorizer(
                    n_features=num_features, alternate_sign=False)

                # Implementing the hashing trick
                V_hashed_features = hashing_vectorizer_v.fit_transform(
                    [concatenated_microphone_data])

                AH_hashed_features = hashing_vectorizer_ah.fit_transform(
                    [concatenated_loopback_data])

                # Initialize TfidfTransformer
                tfidf_transformer = TfidfTransformer(
                    use_idf=True, norm=None, smooth_idf=True)

                # Apply IDF scaling
                AH_tfidf_scaled = tfidf_transformer.fit_transform(
                    AH_hashed_features)
                V_tfidf_scaled = tfidf_transformer.fit_transform(
                    V_hashed_features)

                # Combining features
                combined_features = hstack(
                    [AH_tfidf_scaled, V_tfidf_scaled])

                # Convert to dense array and flatten
                combined_features_dense = combined_features.toarray().flatten()

                # Reshape the flattened array to (1, 400)
                combined_features_reshaped = np.reshape(
                    combined_features_dense, (1, -1))

                predictions = prediction_model.predict(
                    combined_features_reshaped)
                print("Predictions", predictions)

                if ((1-predictions[0][0]) > 0.5):
                    print("Scam detected with probability: {}".format(
                        (1-predictions[0][0])))
                    logger.info("Scam detected with probability: {}".format(
                        (1-predictions[0][0])))
                else:
                    print("Normal conversation detected with probability: {}".format(
                        predictions[0][0]))
                    logger.info("Normal conversation detected with probability: {}".format(
                        predictions[0][0]))

    elif (preprocessing_mode == 3):
        prediction_model = load_prediction_model("LSTM_NeuralNetwork_TFIDF")
        logger.info("Using prediction model: {}".format(prediction_model))

        while True:
            if not microphone_queue.empty() and not loopback_queue.empty():
                microphone_data = microphone_queue.get()
                loopback_data = loopback_queue.get()

                data_for_logs = {
                    "attacker_helper": loopback_data,
                    "victim": microphone_data
                }

                logger.info(
                    "Received: {}, implementing prediction.".format(data_for_logs))

                # Concatenate texts
                concatenated_microphone_data = ' '.join(microphone_data)
                concatenated_loopback_data = ' '.join(loopback_data)

                # Handle preprocessing for the neural network
                num_features = 200

                # Initialize HashingVectorizer to do the hashing trick
                hashing_vectorizer_ah = HashingVectorizer(
                    n_features=num_features, alternate_sign=False)
                hashing_vectorizer_v = HashingVectorizer(
                    n_features=num_features, alternate_sign=False)

                # Implementing the hashing trick
                V_hashed_features = hashing_vectorizer_v.fit_transform(
                    [concatenated_microphone_data])

                AH_hashed_features = hashing_vectorizer_ah.fit_transform(
                    [concatenated_loopback_data])

                # Initialize TfidfTransformer
                tfidf_transformer = TfidfTransformer(
                    use_idf=True, norm=None, smooth_idf=True)

                # Apply IDF scaling
                AH_tfidf_scaled = tfidf_transformer.fit_transform(
                    AH_hashed_features)
                V_tfidf_scaled = tfidf_transformer.fit_transform(
                    V_hashed_features)

                # Combining features
                combined_features = hstack(
                    [AH_tfidf_scaled, V_tfidf_scaled])

                # Convert to dense array and flatten
                combined_features_dense = combined_features.toarray().flatten()

                # Reshape the flattened array to (1, 400)
                combined_features_reshaped = np.reshape(
                    combined_features_dense, (1, -1))

                combined_features_final = combined_features_reshaped[:,
                                                                     np.newaxis, :]

                predictions = prediction_model.predict(
                    combined_features_final)

                print("Predictions", predictions)

                if ((1-predictions[0][0]) > 0.5):
                    print("Scam detected with probability: {}".format(
                        (1-predictions[0][0])))
                    logger.info("Scam detected with probability: {}".format(
                        (1-predictions[0][0])))
                else:
                    print("Normal conversation detected with probability: {}".format(
                        predictions[0][0]))
                    logger.info("Normal conversation detected with probability: {}".format(
                        predictions[0][0]))


if __name__ == "__main__":

    # For help, not to be used in code
    preprocessing_modes = {
        "TF_IDF": 1,
        "NN_TF_IDF": 2,
        "LSTM_NN_TF_IDF": 3
    }

    # Define preprocessing mode
    preprocessing_mode = 3

    # Starting streaming thread
    spark_streaming_thread = Thread(
        target=initiate_spark_streaming, args=[preprocessing_mode])
    spark_streaming_thread.start()

    # Run main application
    run_main_application(preprocessing_mode=preprocessing_mode)
