#%%
from multiprocessing import cpu_count
from callbacks.csv_logger import FloatCutCSVLogger
from dataset.pr_dataset import Dataset
import tensorflow as tf
import tensorflow_addons as tfa
from configs.config import Config
from metrics.f1 import F1
from models.transformer import Encoder as Model
from optimizers import get_optimizer
from vocab.vocab import Vocab
#%%
config = Config("./configs/config.yml")

#%%
vocab = Vocab(**config.vocab_config)
dataset = Dataset(**config.dataset_config, vocab=vocab)

#%%
devices = ["GPU:0", "GPU:1", "GPU:2", "GPU:3"] if config.running_config["use_multi_gpu"] else ["GPU:0"]
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
BATCH_SIZE = config.running_config["batch_size"] * strategy.num_replicas_in_sync

#%%
train_ds = dataset.train.tfds().map(lambda x,y: (x, tf.one_hot(y, 4)),num_parallel_calls=cpu_count())
dev_ds = dataset.dev.tfds().map(lambda x,y: (x, tf.one_hot(y, 4)),num_parallel_calls=cpu_count())
train_ds = train_ds.batch(BATCH_SIZE).prefetch(10)
dev_ds = dev_ds.batch(BATCH_SIZE).prefetch(10)

with strategy.scope():
    model = Model(**config.model_config, vocab_size=len(vocab)).init_build()
    model.summary()
    model.compile(
        optimizer=get_optimizer(config), 
        loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=True, alpha = 0.25, gamma  = 2.0),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(),
            F1(num_classes=4)
        ],
        run_eagerly=config.running_config["run_eagerly"]
    )

    model.fit(
        train_ds,
        validation_data=dev_ds,
        epochs=config.running_config["epochs"],
        steps_per_epoch=config.running_config["steps_per_epoch"],
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                f"saved_weights/my_model",
                monitor='val_loss',
                mode="min",
                save_best_only=True,
                save_weights_only=True
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='./logs'
            ),
            FloatCutCSVLogger("./log.tsv")
        ]
    )
