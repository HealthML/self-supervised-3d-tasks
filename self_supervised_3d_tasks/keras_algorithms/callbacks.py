from tensorflow.python.keras.callbacks import Callback
import numpy as np

class NaNLossError(Exception):
    def __init__(self, *args):
        if len(args) >= 2:
            self.epoch = args[0]
            self.batch = args[1]
        elif len(args) > 0:
            self.epoch = args[0]
        else:
            self.epoch = None
            self.batch = None

    def __str__(self):
        if self.epoch is not None and self.batch is not None:
            return f"NaNLoss occured in e:{self.epoch}/b:{self.batch}"
        else:
            return f"NaNLoss occured."


class TerminateOnNaN(Callback):
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if loss is not None:
            if np.isnan(loss) or np.isinf(loss):
                print(f'Batch %d: Invalid loss: {loss}, terminating training' % (batch))
                self.model.stop_training = True
                raise NaNLossError()
