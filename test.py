#%%
import tensorflow as tf
from configs.config import Config
from models.transformer import Encoder as Model
from vocab.vocab import Vocab
from util.prutils import add_sign_by_mask, get_test_samples
import numpy as np

#%%
config = Config("./configs/config.yml")
# config = Config("saved_weights/20220509/config.yml")

#%%

vocab = Vocab(**config.vocab_config)
model = Model(**config.model_config, vocab_size=len(vocab)).init_build(config.running_config["weights_path"])

#%%
for text in get_test_samples(config, n=10):
  output = model(np.array( [vocab(c) for c in text.split("\t")[0].split(" ")])[np.newaxis, :])
  output = tf.argmax(tf.squeeze(output), -1).numpy()

  label = np.array(list(map(int, text.split("\t")[1].split(" "))))
  chars = np.array(text.strip().split("\t")[0].split(" "))
  
  # print(output)
  # print("-"*15)

  print("hypo:", add_sign_by_mask(chars, output))
  print("-"*15)

  print("label:", add_sign_by_mask(chars, label))

  print("-"*30)
#%%