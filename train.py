#%%
from multiprocessing import cpu_count
from callbacks.csv_logger import FloatCutCSVLogger
from callbacks.tensorboard import TensorBoard
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
print(config)

#%%
vocab = Vocab(**config.vocab_config)
dataset = Dataset(**config.dataset_config, vocab=vocab)

#%%
devices = ["GPU:0", "GPU:1", "GPU:2", "GPU:3"] if config.running_config["use_multi_gpu"] else ["GPU:0"]
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
BATCH_SIZE = config.running_config["batch_size"] * strategy.num_replicas_in_sync

#%%
train_ds = dataset.train.tfds().map(lambda x,y: (x, tf.one_hot(y, config.model_config["classes"])),num_parallel_calls=cpu_count())
dev_ds = dataset.dev.tfds().map(lambda x,y: (x, tf.one_hot(y, config.model_config["classes"])),num_parallel_calls=cpu_count())
train_ds = train_ds.batch(BATCH_SIZE).prefetch(10).repeat(100)
dev_ds = dev_ds.batch(BATCH_SIZE).prefetch(10).repeat(100)


#%%
with strategy.scope():
    model = Model(**config.model_config, vocab_size=len(vocab)).init_build()
    model.summary()

    if config.running_config["load_weights"]:
        weights_path = config.running_config["weights_path"]
        model.load_weights(weights_path)
        print(f"load weights from {weights_path}")

    model.compile(
        optimizer=get_optimizer(config), 
        loss=tfa.losses.SigmoidFocalCrossEntropy(from_logits=True, alpha = 0.25, gamma  = 2.0),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(),
            F1(num_classes=config.model_config["classes"])
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
                config.running_config["weights_path"],
                monitor='val_loss',
                mode="min",
                save_best_only=True,
                save_weights_only=True
            ),
            TensorBoard(
                log_dir='./logs',
                config=config,
                vocab=vocab,
            ),
            FloatCutCSVLogger("./log.tsv")
        ]
    )
