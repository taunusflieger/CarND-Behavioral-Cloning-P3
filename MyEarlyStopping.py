from keras.callbacks import EarlyStopping

class MyEarlyStooping(EarlyStopping):
    def __init__(self, threshold, min_epochs, **kwargs):
        super(MyEarlyStooping, self).__init__(**kwargs)
        self.threshold = threshold # threshold for validation loss
        self.min_epochs = min_epochs # min number of epochs to run

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return

        # implement your own logic here
        if (epoch >= self.min_epochs) & (current >= self.threshold):
            self.stopped_epoch = epoch
            self.model.stop_training = True