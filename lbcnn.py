import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.utils import conv_utils
from abc import ABC, abstractmethod

class LBCSubLayer(ABC):
    @abstractmethod
    def calculate(self, x, weights):
        pass

class TestSubLayer(LBCSubLayer):
    
    def calculate(self, x, weights):
        return x

class SubLayerLBC2D(LBCSubLayer):
    def __init__(self, strides=1, padding='SAME'):
        self.strides = [strides, strides, strides, strides]
        self.padding = padding

    def calculate(self, x, weights):
        feature_map = tf.raw_ops.Conv2D(input=x,filter=weights,strides=self.strides, padding=self.padding)
        return feature_map

class LBC(tf.keras.layers.Layer):
    
    def __init__(self, rank, anchor_weights, filters, kernel_size, sub_layer1, sub_layer2, strides=1, padding='valid', activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer=None, activity_regularizer=None, kernel_constraint=None, trainable=True, name=None, **kwargs):
        super(LBC, self).__init__( trainable=trainable, name=name, activity_regularizer=regularizers.get(activity_regularizer), **kwargs)
        self.rank = rank
        self.kernel_size = kernel_size
        self.anchor_weights = tf.Variable(initial_value=anchor_weights, trainable=False)
        self.filters = filters
        self.intermediary_filters = anchor_weights.shape[-1]
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.sublayer1 = sub_layer1
        self.sublayer2 = sub_layer2
        
    def build(self, input_shape):
        kernel_shape = self.kernel_size + (self.intermediary_filters, self.filters)
        self.kernel = self.add_weight(name='kernel', shape=kernel_shape, initializer=self.kernel_initializer, regularizer=self.kernel_regularizer, constraint=self.kernel_constraint, trainable=True, dtype=self.dtype)

    def call(self, inputs):
        y = self.sublayer1.calculate(inputs, self.anchor_weights)
        y = self.activation(y)
        y = self.sublayer2.calculate(y, self.kernel)
        return y
    
class LBC2D(LBC):
    
    def __init__(self, anchor_weights, filters, strides=1, padding='valid', kernel_initializer='glorot_uniform', kernel_regularizer=None, activity_regularizer=None, kernel_constraint=None, **kwargs):
        sub_layer1 = SubLayerLBC2D(strides=strides, padding=padding.upper())
        sub_layer2 = SubLayerLBC2D(strides=1, padding='SAME')
        self.validate_anchor_weights(anchor_weights)
        super(LBC2D, self).__init__(rank=2, anchor_weights=anchor_weights, filters=filters, kernel_size=(1,1), sub_layer1=sub_layer1, sub_layer2=sub_layer2, strides=strides, padding=padding, activation='relu', kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, **kwargs)

    def validate_anchor_weights(self, anchor_weights):
        if anchor_weights.max() > 1 or anchor_weights.max() < -1:
            raise ValueError("Anchor weights must only have -1, 0 or 1 values!")