import tensorflow as tf
from .schedules import TransformerLRSchedule

def get_optimizer(config):
    init_steps = config.optimizer_config['init_steps']
    step = tf.Variable(init_steps)
    lr_scheduler = TransformerLRSchedule(
        d_model=config.model_config["embedding_dim"],
        init_steps=init_steps,
        warmup_steps=config.optimizer_config['warmup_steps'],
        max_lr=config.optimizer_config['max_lr']
    )
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_scheduler,
        beta_1=config.optimizer_config['beta1'],
        beta_2=config.optimizer_config['beta2'],
        epsilon=config.optimizer_config['epsilon']
    )
    return optimizer