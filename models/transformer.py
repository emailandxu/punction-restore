import tensorflow as tf
from .layers.encoder import EncoderLayer
from models.utils.utils import positional_encoding
from .activations import GLU
from .layers.positional_encoding import PositionalEncoding, PositionalEncodingConcat
from .layers.multihead_attention import MultiHeadAttention, RelPositionMultiHeadAttention
from .layers.position_wise_ffn import PositionWiseFFN

L2 = tf.keras.regularizers.l2(1e-6)

class Encoder(tf.keras.Model):
    def __init__(self,name, num_layers, embedding_dim, num_heads, dff, vocab_size, 
                 maximum_position_encoding=256, dropout=0.1, classes=4):
        super(Encoder, self).__init__()
        self.embeding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.embeding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.embeding_dim)
        self.enc_layer = [EncoderLayer(embedding_dim, num_heads, dff, dropout)
                          for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout)
        # self.bn = tf.keras.layers.BatchNormalization()
        # self.gru = tf.keras.layers.Bidirectional(GRU(int(embedding_dim/2), return_sequences=True))
        # self.fully_layer = tf.keras.layers.Dense(embedding_dim/2)
        # self.res = tf.keras.layers.Add()
        self.class_layer = tf.keras.layers.Dense(classes)



    def call(self, inputs, training=None, mask=None, **kwargs):
        x = inputs
        
        seq_len = tf.shape(x)[1]
        x = self.embeding(x)
        embeding_out = x
        x *= tf.math.sqrt(tf.cast(self.embeding_dim, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training = training)

        for i in range(self.num_layers):
            x = self.enc_layer[i](x, training, mask)
        
        # x = self.res([x,  embeding_out])
        # output = self.gru(x)
        # x = tf.reshape(x, [-1, x.shape()[1]*x.shape()[-1]])
        # x = self.fully_layer(x)
        # x = self.dropout(x, training = training)
        output = self.class_layer(x)

        return output

    def init_build(self):
        x = tf.keras.Input(shape=[None], dtype=tf.int16)
        self(x, states=None, return_state=True, training=False)
        return self

    # @tf.function(input_signature=[tf.TensorSpec([None], dtype=tf.int32)])
    def predict_pb(self, sentence):
        input_len = tf.expand_dims(tf.shape(sentence)[0], axis=0)
        input = tf.expand_dims(sentence, axis=0)
        output = self([input, input_len], training=False)
        output = tf.argmax(output, axis=-1)
        output = tf.squeeze(output)
        return output
