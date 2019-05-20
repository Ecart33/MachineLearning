import numpy as np
import random
import math

'''
Network([Input([...]), Convolute([...]), Flatten(), Dense([...]), (Dense[...]]) will throw error that 
'AttributeError: 'Convolute' object has no attribute 'size'' because it doesn't recognize 'ifinstance(layer, convolute)'
'''
class Network(object):
    def __init__(self, layers):
        self.layers = layers
        self.channels = [layers[0].channels]
        self.shapes = [layers[0].shape]
        for layer, index in zip(layers, range(len(layers))):
            if isinstance(layer, Flatten):
                self.setflattensize(layer, index)
            #if isinstance(layer, Dense):
               # layer.weights = np.random.randn(self.layers[index-1].size, layer.size) 
            if index>0:
                if self.channels[-1]*layer.channels == 0:
                    self.channels.append(1)
                else:
                    self.channels.append(self.channels[-1]*layer.channels)
                if isinstance(layer, Convolute):
                    self.shapes.append(((self.shapes[-1][0]-layer.filter_shape[0]+1)/layer.pool_shape[0], (self.shapes[-1][1]-layer.filter_shape[1]+1)/layer.pool_shape[1]))
                else:
                    self.shapes.append(layer.size)
    
    def setflattensize(self, layer, index):
        layer.size = self.shapes[index-1][0]*self.shapes[index-1][1]*self.channels[index-1]




class Convolute(object):
    def __init__(self, channels, filter_shape, pool_shape, stride_length=1):
        self.channels = channels
        self.filter_shape = filter_shape
        self.filters = [np.random.randn(filter_shape[0], filter_shape[1]) for i in range(channels)]
        self.biases = [random.random() for i in range (channels)]
        self.stride_length = stride_length
        self.pool_shape = pool_shape

class Dense(object):
    def __init__(self, size):
        self.size = size
        self.weights = []
        self.biases = np.random.randn(size, 1)
        self.channels = 0

class Flatten(object):
    def __init__(self):
        self.size = 0
        self.channels = 0

class Input(object):
    def __init__(self, shape, channels):
        self.shape = shape
        self.channels = channels

net = Network([Input((30, 30), 3), Convolute(3, (3, 3), (2, 2)), Dense(100), Dense(10)])
print(net.shapes)