from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.types import FloatType
from pyspark.sql.functions import flatten, udf, col
from pyspark import keyword_only


class FlattenTransformer(Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):

    def __init__(self, inputCol=None, outputCol=None):
        super(FlattenTransformer, self).__init__()
        self._setDefault(inputCol=inputCol, outputCol=outputCol)

    def _transform(self, df):
        # Ensure that the input column is set.
        if self.getInputCol() is None:
            raise ValueError("Input column must be defined")
        # Ensure that the output column is set.
        if self.getOutputCol() is None:
            raise ValueError("Output column must be defined")

        # Apply the flattening operation.
        return df.withColumn(self.getOutputCol(), flatten(df[self.getInputCol()]))


class KerasModelTransformer(Transformer, HasInputCol, HasOutputCol):

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None, model=None):
        super(KerasModelTransformer, self).__init__()
        self._setDefault(inputCol=None, outputCol=None)
        self.model = model  # Your Keras model

    def _transform(self, dataset):
        # Assuming the input is a Dense Vector
        def model_udf(feature):
            prediction = self.model.predict([feature.toArray()])
            return float(prediction[0])

        # Convert to a SQL function
        keras_predict_udf = udf(model_udf, FloatType())

        # Apply the model's predict method as a new column in the DataFrame
        return dataset.withColumn(self.getOutputCol(), keras_predict_udf(col(self.getInputCol())))
