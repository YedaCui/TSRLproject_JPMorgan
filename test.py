import tensorflow as tf

from tf.keras.layers import Dense, Flatten, Conv2D
from tf.keras import Model

class Mymodel(Model):
    def __init__(self):
        super().__init__()
        self.conv2d = Conv2D(32, 3, activation = "relu")
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv2d(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)


        


