from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import flatten


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
