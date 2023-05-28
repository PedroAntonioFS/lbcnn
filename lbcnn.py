import tensorflow as tf
from tensorflow.python.keras import activations
from tensorflow.python.keras.utils import conv_utils
from abc import ABC, abstractmethod

class LBCSubLayer(ABC):
    @abstractmethod
    def calculate(self, x, weights):
        pass

class TestSubLayer(LBCSubLayer):
    
    def calculate(self, x, weights):
        return x

class LBC(tf.keras.layers.Layer):
    
    def __init__(self, rank, anchor_weights, sublayer1, sublayer2, strides=1, padding='valid', activation='relu'):
        super(LBC, self).__init__()
        self.rank = rank
        self.kernel_size = 1
        self.anchor_weights = anchor_weights
        self.filters = anchor_weights.shape[0]
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.activation = activations.get(activation)
        self.sublayer1 = sublayer1
        self.sublayer2 = sublayer2
        

    def build(self, input_shape):
        kernel_shape = [self.kernel_size,self.filters]
        self.kernel = self.add_weight("kernel", shape=kernel_shape)

    def call(self, inputs):
        y = self.sublayer1.calculate(inputs, self.anchor_weights)
        y = self.activation(y)
        y = self.sublayer2.calculate(y, self.kernel)
        return y