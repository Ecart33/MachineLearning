import numpy as np
import random
import math
from scipy import signal
import skimage.measure
import json

class Network(object):

    #params for class are layers described by class e.g. Network([Input([...]), Convolute([...]), Flatten(), Dense([...]), (Dense[...]]) 
    #__init__ and setflattensize functions initilize network structures
    def __init__(self, layers):
        self.loss = []
        self.layers = layers
        self.channels = [layers[0].channels]
        self.shapes = [layers[0].shape]
        for layer, index in zip(layers, range(len(layers))):
            if layer.id == 1:
                self.layers.insert(index+1, Pool(layer.pool_shape, layer.act_derivative, layer.act_type))
        for layer, index in zip(layers, range(len(layers))):
            if layer.id == 3:
                self.setflattensize(layer, index)
            if layer.id == 0:
                #column, rows
                layer.weights = np.random.randn(int(layer.size), int(self.layers[index-1].size))
                layer.weight_gradients = np.zeros(layer.weights.shape)
            #get list of channels and shapes/sizes that correspond with each layer
            if index>0:
                if self.channels[-1]*layer.channels == 0:
                    self.channels.append(1)
                else:
                    self.channels.append(layer.channels*self.channels[-1])
                if layer.id == 1:
                    self.shapes.append((self.shapes[-1][0]-layer.filter_shape[0]+1, self.shapes[-1][1]-layer.filter_shape[1]+1))
                if layer.id == 2:    
                    self.shapes.append((int(self.shapes[-1][0]/layer.pool_shape[0]), int(self.shapes[-1][1]/layer.pool_shape[1])))
                if layer.id == 0 or layer.id == 3:    
                    self.shapes.append(layer.size)
        #weights initialization
        for layer, index in zip(self.layers[1:], range(1, len(self.layers)+1)):
            if index != len(layers)-1:
                if layer.id == 0:
                    layer.weights = layer.weights*np.sqrt(12/(self.shapes[index-1]*self.channels[index-1] + self.shapes[index+1]*self.channels[index+1]))
                if layer.id == 1:
                    layer.filters = [f*np.sqrt(12/(self.shapes[index-1][0]*self.shapes[index-1][1]*self.channels[index-1] + self.shapes[index+1][0]*self.shapes[index+1][1]*self.channels[index+1])) for f in layer.filters]
            else:
                layer.weights = layer.weights*np.sqrt(12/(self.shapes[index-1]*self.channels[index-1]))

    
    def setflattensize(self, layer, index):
        #set size of flattened convolutional pool layer
        #get shape straight from input layer
        layer.size = self.shapes[index-1][0]*self.shapes[index-1][1]*self.channels[index-1]


    def forwardpass(self, input_activations, backprop=False):
        activations = [input_activations]
        zs = [input_activations]
        for layer in self.layers[1:]:
            f = layer.feedforward(activations[-1])
            for i in f[0]:
                zs.append(i)
            for i in f[1]:
                activations.append(i)
            activations = [i for i in activations if i is not None]
            zs = [i for i in zs if i is not None]

        if not backprop:
                return activations
        else:
            return activations, zs
        



    def backwardpass(self, input_layer, training_output, threshold):
        a_z = self.forwardpass(input_layer, backprop=True)
        layer_partial_derivative = []
        gradient = self.cost_derivative(a_z[0][-1], training_output)*self.layers[-1].act_derivative(a_z[self.layers[-1].act_type][-1])
        if np.linalg.norm(gradient) > threshold:
            layer_partial_derivative.append(gradient*threshold/np.linalg.norm(gradient))
        else:
            layer_partial_derivative.append(gradient)
        self.layers[-1].weight_gradients = self.layers[-1].weight_gradients + np.dot(layer_partial_derivative[-1], a_z[0][-2].transpose())
        self.layers[-1].bias_gradients = self.layers[-1].bias_gradients + layer_partial_derivative[-1]

        for layer, index in zip(self.layers[len(self.layers)-2:0:-1], [self.layers.index(i) for i in self.layers[len(self.layers)-2:0:-1]]):
            try:
                w = self.layers[index+1].weights
            except:
                w = None
            try:
                w2 = self.layers[index+2].weights
            except:
                w2 = None
            try:
                l = self.layers[index+1].id
            except:
                l = None
            b = layer.backprop(w, w2, a_z, index, layer_partial_derivative[-1], l)
            [layer_partial_derivative.append(i) for i in b]
            layer_partial_derivative = [i for i in layer_partial_derivative if i is not None]
        self.loss.append(self.cel(a_z[0][-1], layer_partial_derivative[0]))
        
    def cel(self, a, y):
        offset = 1e-7
        return y*np.log(a+offset)+(1-y)*np.log(1-(a-offset))



    def update_weights_biases(self, learning_rate, batch_size, threshold):
        for layer in self.layers:
            if layer.id == 0:
                layer.weights = layer.weights - (learning_rate/batch_size)*layer.weight_gradients
                layer.biases = layer.biases - (learning_rate/batch_size)*layer.bias_gradients
                layer.weight_gradients = np.zeros(layer.weight_gradients.shape)
                layer.bias_gradients = np.zeros(layer.bias_gradients.shape)
            if layer.id == 1:
                layer.filters = [f-(learning_rate/batch_size)*fg for f,fg in zip(layer.filters, layer.filter_gradients)]
                layer.biases = [b-(learning_rate/batch_size)*bg for b, bg in zip(layer.biases, layer.bias_gradients)]
                layer.filter_gradients = [np.zeros(i.shape) for i in layer.filter_gradients]
                layer.bias_gradients = [0 for i in layer.bias_gradients]
    # epochs is number of epoochs to train for
    # batch_size is the size of the mini_batches
    # learning_rate is the learning rate for the sgd
    # training_data is an array containing tuples of (input_array, training_output) to train on
    # test data is an array containing tuples of (input_array, training_output) to test on
    def train(self, epochs, batch_size, learning_rate, cycle_length, threshold, training_data, test_data=None):
        learning_rate_r = learning_rate
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+batch_size] for k in range(0, len(training_data), batch_size)]
            for batch in mini_batches:
                for input_data, output_data in batch:
                    self.backwardpass(input_data, output_data, threshold)
                self.update_weights_biases(learning_rate_r, batch_size, threshold)
            learning_rate_r = self.update_learning_rate(learning_rate, i, cycle_length)  
            if test_data:
                print('epoch {0} --------- loss: {1} --------- learning rate: {2} --------- test accuracy: {3}'.format( \
                    i, np.linalg.norm(-sum(self.loss)/len(training_data)), learning_rate_r, self.test_progress(test_data)/len(test_data)))
            else:
                print('epoch {0} --------- loss: {1} --------- learning rate: {2}'.format( \
                    i, np.linalg.norm(-sum(self.loss)/len(training_data)), learning_rate_r))
            self.loss = []
            

    def update_learning_rate(self, learning_rate, epoch, cycle_length):
       value_scaled = (epoch % cycle_length)/(cycle_length-1)
       theta = value_scaled*math.pi/2
       return learning_rate*math.cos(theta)
    
    def test_progress(self, training_data):
        right = 0
        for i in range(len(training_data)):
            right = right + int(np.argmax(self.forwardpass(training_data[i][0])[-1]) == np.argmax(training_data[i][1]))     
        return right

    def cost_derivative(self, a, y):
        #quadratic derivative
        qd = (a-y)
        #cross entropy derivative
        offset = 1e-7
        ced =  -((y/(a+offset))-(1-y)/(1-(a-offset)))
        return ced





# self.id identifies what type of layer
# 0 is Dense 
# 1 is Convolutional
# 2 is Pool
# 3 is Flatten
# 4 is Input 

activation_functions = {
    'sigmoid': lambda y: sigmoid(y),
    'ReLu': lambda y: y*(y>0)+0.01*y*(y<=0),
    'softmax': lambda y: np.exp(y-np.max(y))/np.sum(np.exp(y-np.max(y)))
}
activation_derivative = {
    'sigmoid': lambda y: sigmoid_derivative(y),
    'ReLu': lambda y: 1*(y>0)+0.01*(y<=0),
    'softmax': lambda y: y*(1-y)
}

class Pool(object):
    def __init__(self, shape, activation_derivative, act_type):
        self.id = 2
        self.channels = 1
        self.pool_shape = shape
        self.act_derivative = activation_derivative
        self.act_type = act_type
    
    def backprop(self, w, w2, a_z, i, d, l):
        if l == 3:
            flattened = np.dot(w2.transpose(), d)*self.act_derivative(a_z[self.act_type][i+1])
            return [np.reshape(flattened, a_z[0][i].shape)]
        else:
            return [None]
    
    def feedforward(self, a):
        return [None], [None]

class Convolute(object):
    #Convolutional layer
    #channels is amount of feature maps
    #filter shape is the shape of the filter in a tuple e.g. (5, 5)
    #pool_shape is the shape of the pool layer in a tuple e.g. (4,4) 
    #pool_type is the type of pooling NOTE use string or integer to decide what type of pooling
    #stride_length is the length of the stride in the convolution
    def __init__(self, channels, filter_shape, pool_shape, activation_type, stride_length=1):
        self.channels = channels
        self.filter_shape = filter_shape
        self.filters = [np.random.randn(filter_shape[0], filter_shape[1]) for i in range(channels)]
        self.filter_gradients = [np.zeros(i.shape) for i in self.filters]
        self.biases = [0 for i in range (channels)]
        self.bias_gradients = [0 for i in self.biases]
        self.stride_length = stride_length
        self.pool_shape = pool_shape
        self.act_func = activation_functions[activation_type]
        self.act_derivative = activation_derivative[activation_type]
        self.act_type = int(activation_type!='softmax')
        self.id = 1

    def backprop(self, w, w2, a_z, i, d, l):
        filter_gradients = []
        bias_gradients = []
        input_gradients = []
        current_partial_derivates = [] 
        for feature_map in range(a_z[0][i].shape[-1]):                     
            l = a_z[0][i+1]
            pool_repeat = l[...,feature_map].repeat(self.pool_shape[0], axis=0).repeat(self.pool_shape[1], axis=1)
            dPoolC = np.equal(a_z[0][i][...,feature_map], pool_repeat).astype(int)
            # dL/dOut
            dCout = dPoolC*d[...,feature_map].repeat(self.pool_shape[0], axis=0).repeat(self.pool_shape[1], axis=1)
            dCout = self.act_derivative(a_z[self.act_type][i][...,feature_map])*dCout
            current_partial_derivates.append(np.reshape(dCout, (dCout.shape[0], dCout.shape[1], 1)))
            if feature_map%self.channels == 0:
                input_gradient = signal.convolve2d(self.filters[0], dCout)
                input_gradients.append(np.reshape(input_gradient, (input_gradient.shape[0], input_gradient.shape[1], 1)))
            if feature_map<self.channels:
                filter_gradients.append(signal.correlate2d(a_z[0][i-1][...,0], dCout, mode='valid'))
                bias_gradients.append(np.sum(a_z[1][i][...,0]*dCout))
        for f, i  in zip(filter_gradients, range(len(filter_gradients))):
            self.filter_gradients[i] = self.filter_gradients[i] + f
        for b, i  in zip(bias_gradients, range(len(bias_gradients))):
            self.bias_gradients[i] = self.bias_gradients[i] + b
        return [np.concatenate(current_partial_derivates, axis=2), np.concatenate(input_gradients, axis=2)]

    def feedforward(self, a):
        feature_maps = []
        pool_layers = []
        #k defined here since it is independent of channels of layer
        #k reduces a 2d feature map then turns it into a 3d array
        k = lambda x: (lambda y :  np.reshape(y, (y.shape[0], y.shape[1], 1)))(skimage.measure.block_reduce(x, self.pool_shape, np.max))
        for i in range(self.channels):
            #l convolutes a 2d array and the layer filters and biases then turns it into 3d array
            l = lambda x : (lambda y : np.reshape(y, (y.shape[0], y.shape[1], 1)))(signal.correlate2d(x, self.filters[i], mode='valid')+ self.biases[i])             
            fm = np.concatenate([l(a[...,i]) for i in range(a.shape[-1])], axis=2)
            feature_maps.append(fm)
            pool_layers.append(np.concatenate([k(fm[...,i]) for i in range(fm.shape[-1])], axis=2))
        z = np.concatenate(feature_maps, axis=2)
        zp = np.concatenate(pool_layers, axis=2)
        return [z, zp], [self.act_func(z), self.act_func(zp)]

class Dense(object):
    #fully connected layer
    #size is the amount of neurons in the dense layer
    #biases are initialized to 0
    #weights are initialized based on layer structures
    def __init__(self, size, activation_type):
        self.size = size
        self.weights = None
        self.weight_gradients = None
        self.biases = np.zeros((size, 1))
        self.bias_gradients = np.zeros(self.biases.shape)
        self.channels = 1
        self.act_func = activation_functions[activation_type]
        self.act_derivative = activation_derivative[activation_type]
        self.act_type = int(activation_type!='softmax')
        self.id = 0
    
    def backprop(self, w, w2, a_z, i, d, l):
        lpd = np.dot(w.transpose(), d)*self.act_derivative(a_z[self.act_type][i])
        weight_gradient = np.dot(lpd, a_z[0][i-1].transpose())
        bias_gradient = lpd
        self.weight_gradients = self.weight_gradients + weight_gradient
        self.bias_gradients = self.bias_gradients + bias_gradient 
        return [lpd]

    def feedforward(self, a):
        z = np.dot(self.weights, a) + self.biases
        return [z], [self.act_func(z)]


class Flatten(object):
    #flatten goes before dense layer
    #size is the size of the previous layer in a single column vector and is initialized in convolutionallayer __init__
    def __init__(self):
        self.size = 0
        self.channels = 0
        self.id = 3 
        self.act_type = 0

    def backprop(self, w, w2, a_z, i, d, l):
        return [None]

    def feedforward(self, a):
        return [[np.reshape(a, (a.size, 1))] for i in range(2)]

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



#saves network model to json file
def save_to_json(network, filename):
    layer_json = []
    for layer in network.layers:
        layer_dict = {}
        for i in layer.__dict__:
            if type(layer.__dict__[i]).__name__ == 'ndarray': 
                layer.__dict__[i] = layer.__dict__[i].tolist()
            if type(layer.__dict__[i]).__name__ != 'function': 
                layer_dict[i] = layer.__dict__[i]

        layer_json.append(layer_dict)
    data = {}
    data['layers'] = layer_json
    with open(filename, 'w+') as outfile:
        json.dump(data, outfile)

#loads network model from json file
def load_from_json(net, filename):
    with open(filename) as json_file:
        data = json.load(json_file)
        for net_layer, json_layer in zip(net.layers, data['layers']):
            for i in json_layer:
                if type(json_layer[i]).__name__ == 'list':
                    json_layer[i] = np.array(json_layer[i])
                net_layer.__dict__[i] = json_layer[i]

