import numpy as np
import random
import math
from scipy import signal
import skimage.measure

#rgb input is 3 differnet channels treat as 3 different feature maps in MLP treat as 3 different networks?

class MLPNetwork(object):

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

#NOTE potentially one class can encompass mlp as well as convolutional because would just mean that instead of 
#ConvolutionalNetwork([Input([...]), Convolute([...]), Flatten(), Dense([...]), (Dense[...]]) 
#it would be ConvolutionalNetwork([Input([...]), Flatten([...]), Dense([...]), (Dense[...]])

class Network(object):

    #params for class are layers described by class e.g. ConvolutionalNetwork([Input([...]), Convolute([...]), Flatten(), Dense([...]), (Dense[...]]) 
    #__init__ and setflattensize functions initilize network structures
    def __init__(self, layers):
        self.layers = layers
        self.channels = [layers[0].channels]
        self.shapes = [layers[0].shape]
        for layer, index in zip(layers, range(len(layers))):
            if layer.id == 3:
                self.setflattensize(layer, index)
            if layer.id == 0:
                #column, rows
                layer.weights = np.random.randn(int(layer.size), int(self.layers[index-1].size))
            #get list of channels and shapes/sizes that correspond with each layer
            if index>0:
                if self.channels[-1]*layer.channels == 0:
                    self.channels.append(1)
                else:
                    self.channels.append(self.channels[-1]*layer.channels)
                if layer.id == 1:
                    self.shapes.append((int((self.shapes[-1][0]-layer.filter_shape[0]+1)/layer.pool_shape[0]), int((self.shapes[-1][1]-layer.filter_shape[1]+1)/layer.pool_shape[1])))
                else:    
                    self.shapes.append(layer.size)
        for layer, index in zip(layers, range(len(layers))):
            if layer.id == 1:
                self.layers.insert(index+1, Pool())
    
    def setflattensize(self, layer, index):
        #set size of flattened convolutional pool layer
        #get shape straight from input layer
        layer.size = self.shapes[index-1][0]*self.shapes[index-1][1]*self.channels[index-1]


    def forwardpass(self, input_activations):
        activations = [input_activations]
        for layer in self.layers:
            if layer.id == 1:
                feature_maps = []
                pool_layers = []
                #k defined here since it is independent of channels of layer
                #k reduces a 2d feature map then turns it into a 3d array
                k = lambda x: (lambda y :  np.reshape(y, (y.shape[0], y.shape[1], 1)))(skimage.measure.block_reduce(x, layer.pool_shape, np.max))
                for i in range(layer.channels):
                    #l convolutes a 2d array and the layer filters and biases then turns it into 3d array
                    l = lambda x : (lambda y : np.reshape(y, (y.shape[0], y.shape[1], 1)))(sigmoid(signal.correlate2d(x, layer.filters[i], mode='valid') + layer.biases[i])) 
                    fm = np.concatenate([l(activations[-1][...,i]) for i in range(activations[-1].shape[-1])], axis=2)
                    feature_maps.append(fm)
                    pool_layers.append(np.concatenate([k(fm[...,i]) for i in range(fm.shape[-1])], axis=2))
                activations.append(np.concatenate(feature_maps, axis=2))
                activations.append(np.concatenate(pool_layers, axis=2))
            elif layer.id == 0:
                activations.append(sigmoid(np.dot(layer.weights, activations[-1]) + layer.biases))
            elif layer.id == 3:
                #flatten previous layer (either input layer or convolutional layer) into a column vector 
                activations.append(np.reshape(activations[-1], (activations[-1].size, 1)))
        return activations
'''
    def backwardpass(self, input_layer, training_output):
        ####PSUEDO CODE
        #input x gradient = signal.convolve2d(filter, loss from last layer) zero padding full convolution
        #filter f gradient = signal.correlate(input x, loss from last layer)
        #convolve2d is reversed correlate 
        activations = self.forwardpass(input_layer)
        layer_partial_derivative = []
        #iterate backwards over activations
        for layer, activation in zip(self.layers, activations[::-1]):
            if layer.id == 1:
                # deal with backprop of max layer either with matrix or bypass?????
                for feature_map in len(activation):
                    input_gradient = signal.convolve2d(layer.filters[feature_map], layer_partial_derivative[-1])
                    layer_partial_derivative.append(input_gradient)
                    #messy if multiple inputs for each feature map
                    #also will break if there is no layer before have to fix in some way
                    filter_gradient = signal.correlate(activations[activations.index(activation)-1], layer_partial_derivative[-1])
                    bias_gradient = sigmoid_derivative*layer_partial_derivative
            elif layer.id == 2:
                
            elif layer.id == 0:
                #given delta loss/delta activation 
                #chain rule here this might not be exact but close
                weight_gradient = sigmoid_derivative*layer_partial_derivative[-1]*activations[activations.index(activation)-1]
                bias_gradient = sigmoid_derivative*layer_partial_derivative
                previous_layer_loss = sigmoid_derivative*layer_partial_derivative[-1]*layer.weights
                #normal backprop
            
        def cost_derivative(self, a, y):
        #quadratic derivative
            qd = (a-y)
            #cross entropy derivative
            ced =  -((y/a)-(1-y)/(1-a))
            return ced

'''


# self.id identifies what type of layer
# 0 is Dense 
# 1 is Convolutional
# 2 is Pool
# 3 is Flatten
# 4 is Input 
class Pool(object):
    def __init__(self):
        self.id = 2

#NOTE add an id element to use instead of ifinstance since ifinstance fails for some import cases
class Convolute(object):
    #Convolutional layer
    #channels is amount of feature maps
    #filter shape is the shape of the filter in a tuple e.g. (5, 5)
    #pool_shape is the shape of the pool layer in a tuple e.g. (4,4) 
    #pool_type is the type of pooling NOTE use string or integer to decide what type of pooling
    #stride_length is the length of the stride in the convolution
    def __init__(self, channels, filter_shape, pool_shape, stride_length=1):
        self.channels = channels
        self.filter_shape = filter_shape
        self.filters = [np.random.randn(filter_shape[0], filter_shape[1]) for i in range(channels)]
        self.biases = [random.random() for i in range (channels)]
        self.stride_length = stride_length
        self.pool_shape = pool_shape
        self.id = 1

class Dense(object):
    #fully connected layer
    #size is the amount of neurons in the dense layer
    #biases are initialized randomly
    #weights are initialized in the convulutionallayer __init__ by the flatten layer
    def __init__(self, size):
        self.size = size
        self.weights = []
        self.biases = np.random.randn(size, 1)
        self.channels = 0
        self.id = 0

class Flatten(object):
    #flatten buffer between convolutional layer and fully connected layer
    #size is the size of the previous layer in a single column vector and is initialized in convolutionallayer __init__
    def __init__(self):
        self.size = 0
        self.channels = 0
        self.id = 3 

class Input(object):
    #input layer
    #shape is the shape of the images being inputted
    #channels is the amount of channels. e.g. grayscale has 1 channel and rgb has 3 channels
    def __init__(self, shape, channels):
        self.shape = shape
        self.channels = channels
        self.id = 4

def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))

def sigmoid(z):
    return 1/(1+np.exp(-z))

net = Network([Input((30, 30), 3), Convolute(3, (3, 3), (2, 2)), Flatten(), Dense(100), Dense(10)])
arc = net.forwardpass(np.random.randn(30, 30, 3))
