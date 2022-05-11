import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard as OfficialTensorBoard
from util.prutils import get_test_samples, add_sign_by_mask

class TensorBoard(OfficialTensorBoard):
    def __init__(self, log_dir='logs', histogram_freq=0, write_graph=True, write_images=False, update_freq='epoch', profile_batch=2, embeddings_freq=0, embeddings_metadata=None, config=None, vocab=None, **kwargs):
        self.config = config
        self.vocab = vocab
        super().__init__(log_dir, histogram_freq, write_graph, write_images, update_freq, profile_batch, embeddings_freq, embeddings_metadata, **kwargs)
        
    def on_epoch_end(self, epoch, logs=None):
        with self._val_writer.as_default():

            texts = get_test_samples(self.config, n=10)
            text = random.choice(texts)

            self.model.trainable=False
            output = self.model(np.array( [self.vocab(c) for c in text.split("\t")[0].split(" ")])[np.newaxis, :])
            output = tf.argmax(tf.squeeze(output), -1).numpy()

            label = np.array(list(map(int, text.split("\t")[1].split(" "))))
            chars = np.array(text.strip().split("\t")[0].split(" "))
            self.model.trainable=True
            
            label_text = add_sign_by_mask(chars, label)
            hypo_text = add_sign_by_mask(chars, output)
            tf.summary.text("label", label_text, step=epoch)
            tf.summary.text("hypo", hypo_text, step=epoch)

        return super().on_epoch_end(epoch, logs)