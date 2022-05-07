import tensorflow_addons as tfa
import tensorflow as tf

class F1(tfa.metrics.F1Score):
    def __init__(self, num_classes, average=None, threshold=None, name="f1_score", dtype=None):
        super().__init__(num_classes, average, threshold, name, dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        C = tf.shape(y_true)[-1]
        y_true = tf.reshape(y_true, (-1,C))
        y_pred = tf.reshape(y_pred, (-1,C))
        super().update_state(y_true, y_pred, sample_weight)