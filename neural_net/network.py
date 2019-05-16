import numpy as np
import random
import math
from scipy import signal
import skimage.measure

class Network(object):

    def __init__(self, layers):
        self.num_layers = len(layers)
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

    def sgd(self, epochs, batch_size, learning_rate, training_data, test_data=None):
        accuracy = []
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+batch_size]
                            for k in range(0, len(training_data), batch_size)]
            for batch in mini_batches:
                weight_gradients = [np.zeros(w.shape) for w in self.weights]
                bias_gradients = [np.zeros(b.shape) for b in self.biases]
                for x,y in batch:
                    weight_gradient, bias_gradient = self.backpropagate(x,y)
                    weight_gradients = [wgs+wg for wgs, wg in zip(weight_gradients, weight_gradient)]
                    bias_gradients = [bgs+bg for bgs, bg in zip(bias_gradients, bias_gradient)]
                self.update_weights_biases(weight_gradients, bias_gradients, learning_rate, batch_size)
            if test_data:
                e = self.progress(test_data)
                m = len(test_data)
                print(e)
                print(m)
                print('Epoch {0}: {1}%'.format(epoch+1, e/m))
                accuracy.append(e/m)
            else:
                print('Epoch {0} complete'.format(epoch))
        print(accuracy)

    def progress(self, training_data):
        #predictions = [(np.argmax(self.feedforward(x)), y) for x,y in training_data]
        #print(predictions)
        #prediction for specifically 1 neuron output layer
        predictions = [(round(self.feedforward(x)[0][0]), y) for x,y in training_data]
        return sum([int(x==y) for x,y in predictions])

    def update_weights_biases(self, weight_gradient, bias_gradient, learning_rate, batch_size):
        self.weights = [w-(learning_rate/batch_size)*wg for w, wg in zip(self.weights, weight_gradient)]
        self.bias = [b-(learning_rate/batch_size)*bg for b, bg in zip(self.biases, bias_gradient)]

    def backpropagate(self, input_layer, training_output):
        weight_gradient = [np.zeros(w.shape) for w in self.weights]
        bias_gradient = [np.zeros(b.shape) for b in self.biases]
        activations, zs = self.feedforward(input_layer, backprop=True)
        errors = [self.cost_derivative(activations[-1], training_output) * sigmoid_derivative(zs[-1])]
        bias_gradient[-1]=errors[0]
        weight_gradient[-1]= np.dot(errors[0], activations[-2].transpose())
        for i in range(2, self.num_layers):
            #transpose b/c connecting same layer so rows and columns should be switched
            #usually weight rows represent next layer and columns represent current layer
            error = np.dot(self.weights[-i+1].transpose(), errors[i-2]) * sigmoid_derivative(zs[-i])
            errors.append(error)
            bias_gradient[-i] = errors[i-2]
            weight_gradient[-i] = np.dot(errors[i-1], activations[-i-1].transpose())
        return weight_gradient, bias_gradient

    def cost_derivative(self, a, y):
        #quadratic derivative
        #return (a-y)
        #cross entropy derivative
        return -((y/a)-(1-y)/(1-a))

    def feedforward(self, input, backprop=False):
        activations = [np.reshape(input, (input.size, 1))]
        z = []
        for w, b in zip(self.weights, self.biases):
            if backprop:
                z.append(np.dot(w, activations[-1]) + b)
            activations.append(sigmoid(np.dot(w, activations[-1]) + b))
        if backprop:
            return activations, z
        else:
            return activations[-1]


'''
----------------------------------Work on FCNN is being delayed----------------
class FCNN(object):
    #Convoluional Neural Net in frequency domain
    


    #ConvolutionalNetwork([ConvolutionalLayer(...), ConnectedLayer(...), OutputLayer(...)])
    def __init__(self, layers):
        self.layers = layers

    #do everything in fourier domain
    def forwardpass(self, input_layer):
        #array of all activations
        activations = []
        for layer in self.layers:
            if isinstance(layer, ConvolutionalLayer):
                #hadamard filters and input for each feature map
                #then make pooling layer by taking subset of fourier domain
                #append to activations array
            elif isinstance(layer, ConnectedLayer) || isinstance(OutputLayer):
                #standard dot weights and activations then add bias
                #append to activations array

#calculate local gradient in layer class?
'''

class ConvolutionalNetwork(object):

    #params for class are layers described by class e.g. ConvolutionalNetwork([ConvolutionalLayer[...], ConnectedLayer[...], OutputLayer[...]])
    def __init__(self, layers):
        self.layers = layers
        for layer in self.layers:
            if isinstance(layer, Flatten):
                setflattensize(layer)
            if isinstance(layer, ConnectedLayer):
                layer.weights = np.random.randn(self.layers[self.layers.index(layer)-1].size, layer.size)
    
    def setflattensize(self, layer):
        layer.size = self.layers[self.layers.index(layer)-1].pool_shape.size*self.layers[self.layers.index(layer)-1].feature_maps

    def forwardpass(self, input_activations):
        activations = [input_activations]
        for layer in self.layers:
            if isinstance(layer, ConvolutionalLayer):
                feature_maps = []
                pool_layers = []
                for feature_map_index in len(layer.feature_maps):
                    feature_maps.append(sigmoid(signal.correlate2d(activations[:-1], layer.filters[feature_map_index], mode='valid') + layer.biases[feature_map_index]))
                for feature_map in feature_maps:
                    pool_layers.append(skimage.measure.block_reduce(feature_map, layer.pool_shape, np.max))
                activations.append((feature_maps, pool_layers))
            elif isinstance(layer, ConnectedLayer):
                #previous layer into 1 colomn
                a = activations[-1]
                if isinstance(self.layers[self.layers.index(layer)-1], ConvolutionalLayer):
                    a = np.zeros((0, 1))
                    for pool_maps in activations[-1]:
                        a = np.append(a, np.reshape(pool_maps, (pool_maps.size, 1)), axis=0)
                activations.append(sigmoid(np.dot(layer.weights, a) + layer.biases))
        return activations
    
    def backwardpass(self, input_layer, training_output):
        ####PSUEDO CODE
        #input x gradient = signal.convolve2d(filter, loss from last layer) zero padding full convolution
        #filter f gradient = signal.correlate(input x, loss from last layer)
        #convolve2d is reversed correlate 
        activations = self.forwardpass(input_layer)
        layer_partial_derivative = []
        for layer, activation in zip(self.layers[::-1], activations[::-1]):
            if isinstance(layer, ConvolutionalLayer):
                # deal with backprop of max layer either with matrix or bypass?????
                for feature_map in len(activation):
                    input_gradient = signal.convolve2d(layer.filters[feature_map], layer_partial_derivative[-1])
                    layer_partial_derivative.append(input_gradient)
                    #messy if multiple inputs for each feature map
                    #also will break if there is no layer before have to fix in some way
                    filter_gradient = signal.correlate(activations[activations.idex(activation)-1], layer_partial_derivative[-1])
            elif isinstance(layer, ConnectedLayer):
                #given delta loss/delta activation 
                #chain rule here this might not be exact but close
                weight_gradient = sigmoid_derivative*layer_partial_derivative[-1]*activations[activations.idex(activation)-1]
                bias_gradient = sigmoid_derivative*layer_partial_derivative
                previous_layer_loss = sigmoid_derivative*layer_partial_derivative[-1]*layer.weights
                #normal backprop
            elif isinstance(layer, OutputLayer):
                layer_partial_derivative.append(self.cost_derivative(activation, training_output))
            
        def cost_derivative(self, a, y):
        #quadratic derivative
        #return (a-y)
        #cross entropy derivative
            return -((y/a)-(1-y)/(1-a))






class ConvolutionalLayer(object):
    #ConvolutionalLayer(feature_maps=5, filter_size=[2,2], stride_length=1, pool_type="L2")
    def __init__(self, feature_maps, filter_shape, stride_length=1, pool_shape, pool_type):
        self.feature_maps = feature_maps
        self.filters = [np.random.randn(filter_shape) for i in range(feature_maps)]
        self.biases = [random.random() for i in range (feature_maps)]
        self.stride_length = stride_length
        self.pool_shape = pool_shape
        self.pool_type = pool_type


class ConnectedLayer(object):
    #ConnectedLayer(size=100)
    def __init__(self, size, weights):
        self.size = size
        self.weights = []
        self.biases = np.random.randn(size, 1)

class Flatten(object):
    def __init__(self):
        self.size = 0


class OutputLayer(object):
    #OutputLayer(size=10, type="softmax")
    def __init__(self, shape):
        self.size = size

class InputLayer(object):
    def __init__(self, size):
        self.output_size = size


def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))

def sigmoid(z):
    return 1/(1+np.exp(-z))
