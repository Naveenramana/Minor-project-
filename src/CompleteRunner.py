from tensorflow.keras.models import load_model
import sys
import findspark
from pyspark.sql import SparkSession
from pyspark.ml.classification import *
from pyspark.sql.functions import *
from pyspark.ml import PipelineModel
import logging
import ctypes
import pandas

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from pyspark.sql.types import StructType, StructField, StringType


from scipy.sparse import hstack
import nltk
# nltk.download('punkt')
# nltk.download('stopwords')

sys.path.append("G:\Dissertation_Project")


def initializer():

    print("\n\n-----------------------INITIALIZATION RUNNING----------------------\n")

    log_format = "%(asctime)s - %(name)s - %(message)s"
    logging.basicConfig(filename="G:\\Dissertation_Project\\Logs\\performance_logs.log",
                        level=logging.INFO, format=log_format)
    logger = logging.getLogger("Dissertation_Project")
    logger.setLevel(logging.INFO)

    py4j_logger = logging.getLogger('py4J')
    py4j_logger.setLevel(logging.ERROR)

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

    return spark, logger


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


def raise_alert_on_scam_detection(prediction, probability_vector):
    # Check if the prediction is 1 and the second element in the probability vector is greater than 0.9
    if prediction == 1 and probability_vector[1] > 0.9:
        # Windows alert
        ctypes.windll.user32.MessageBoxW(
            0, "High Probability Alert!", "Alert", 1)
        return True
    return False


if __name__ == "__main__":

    spark = None
    query = None
    console_query = None

    preprocessing_modes = {
        "TF_IDF": 1,
        "NN_TF_IDF": 2,
        "LSTM_NN_TF_IDF": 3
    }

    try:
        preprocessing_mode = 2
        # Initializing spark and other things necessary
        spark, logger = initializer()

        # Loading the preprocessing pipeline
        pipeline_path = "G:\\Dissertation_Project\\src\\Models\\Pipelines\\Prediction_Pipeline"
        pipeline_model = PipelineModel.load(pipeline_path)

        # Loading the prediction model
        prediction_model = load_prediction_model("NeuralNetwork_TFIDF")

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

        if preprocessing_mode == 1:

            def process_and_log_batch(batch_df, batch_id):
                logger.info(
                    f"Batch ID: {batch_id}, --Data has been received--")

                if batch_df:
                    # Log the batch
                    batch_string = batch_df._jdf.showString(1, 20, False)
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

        elif preprocessing_mode == 2:

            @pandas_udf('array<float>', PandasUDFType.SCALAR)
            def predict_udf(feature_series):
                # Convert the feature_series (which is a Pandas Series of lists) to a 2D numpy array
                feature_array = np.stack(feature_series)
                # Get predictions from the Keras model, assuming it returns probabilities
                predictions = prediction_model.predict(feature_array)
                # Return predictions as a Pandas Series of arrays
                return pandas.Series(list(predictions))

            def process_and_log_batch(batch_df, batch_id):

                logger.info(
                    f"Batch ID: {batch_id}, --Data has been received--")

                if batch_df:
                    # This will create a column of arrays where each array contains class probabilities
                    predictions_df = batch_df.withColumn(
                        'probabilities', predict_udf(col('combined_features')))

                    # Log the batch
                    batch_string = predictions_df._jdf.showString(1, 20, False)
                    logger.info(f"Batch ID: {batch_id}, Data:\n{batch_string}")

                    # Print to console
                    predictions_df.show(truncate=False)

            preprocessed_df = pipeline_model.transform(parsed_df)

            query = preprocessed_df.writeStream \
                .outputMode("append") \
                .foreachBatch(process_and_log_batch) \
                .start()
        else:
            pass

        # Await termination to keep the streaming application running
        query.awaitTermination()

    except KeyboardInterrupt:
        print(
            "Keyboard Interrupt received. Stopping the streaming queries and Spark session.")
        if query:
            query.stop()

        if spark:
            spark.stop()
