from tensorflow.keras.callbacks import CSVLogger

class FloatCutCSVLogger(CSVLogger):
    def __init__(self, filename, separator=',', append=False, accuracy_for_float=4):
        super().__init__(filename, separator, append)
        self.accuracy_for_float = accuracy_for_float

    def on_epoch_end(self, epoch, logs=None):
        new_logs = {}
        type_logs = {}
        for k,v in logs.items():
            if isinstance(v, float):
                factor = 10 ** self.accuracy_for_float
                v = int(v * factor) / factor
            new_logs[k] = v
            type_logs[k] = type(v)
        return super().on_epoch_end(epoch, new_logs)