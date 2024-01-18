from keras import backend as K
import keras
import tensorflow as tf
# Custom F1Score metric


@keras.saving.register_keras_serializable()
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.round(y_pred)
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        self.f1_score = 2 * ((p * r) / (p + r + K.epsilon()))
        return self.f1_score

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()
