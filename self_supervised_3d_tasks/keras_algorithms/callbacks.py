from tensorflow.keras import callbacks


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


class TerminateOnNaN(callbacks.TerminateOnNaN):
    def on_batch_end(self, batch, logs=None):
        super(TerminateOnNaN, self).on_batch_end(batch, logs)
        raise NaNLossError()
