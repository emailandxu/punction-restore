
import tensorflow as tf


class PositionWiseFFN(tf.keras.layers.Layer):
    def __init__(self,
                 dff,
                 d_model,
                 dropout,
                 name="point_wise_ffn",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(name=name, **kwargs)
        self.ln = tf.keras.layers.LayerNormalization(gamma_regularizer=kernel_regularizer,
                                                     beta_regularizer=bias_regularizer)
        self.conv1 = tf.keras.layers.Conv1D(filters=dff, kernel_size=1, activation='relu',
                                            kernel_regularizer=kernel_regularizer,
                                            bias_regularizer=bias_regularizer)
        self.conv2 = tf.keras.layers.Conv1D(filters=d_model, kernel_size=1,
                                            kernel_regularizer=kernel_regularizer,
                                            bias_regularizer=bias_regularizer)
        self.do1 = tf.keras.layers.Dropout(dropout)
        self.do2 = tf.keras.layers.Dropout(dropout)
        self.res_add = tf.keras.layers.Add()

    def call(self, inputs, training=False, **kwargs):
        outputs = self.ln(inputs, training=training)
        outputs = self.conv1(outputs, training=training)
        outputs = self.do1(outputs, training=training)
        outputs = self.conv2(outputs, training=training)
        outputs = self.do2(outputs, training=training)
        outputs = self.res_add([inputs, outputs])
        return outputs

    def get_config(self):
        conf = super(PositionWiseFFN, self).get_config()
        conf.update(self.conv1.get_config())
        conf.update(self.conv2.get_config())
        conf.update(self.do1.get_config())
        conf.update(self.do2.get_config())
        conf.update(self.ln.get_config())
        conf.update(self.res_add.get_config())
        return conf
