import tensorflow.keras as kr
from tensorflow.keras.layers import BatchNormalization
BatchNormalization._USE_V2_BEHAVIOR = False

class SubnetFactory:
    def __init__(self, ch_hidden):
        self.ch = ch_hidden

    def __call__(self, ch_in, ch_out):
        net = kr.models.Sequential([
                kr.layers.Conv2D(self.ch, 3, padding='same'),
                BatchNormalization(),
                kr.layers.ReLU(),

                kr.layers.Conv2D(self.ch, 3, padding='same'),
                BatchNormalization(),
                kr.layers.ReLU(),

                kr.layers.Conv2D(ch_out, 3, padding='same'),
            ], name='subnet')

        return net

