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
                    ##NOTE NOTE NOTE NOTE find way to add convo layer to self.shapes or else itll fuck up index
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
       #     print('LAYER ACTIVATION SIZES ----- {0}'.format([np.linalg.norm(a) for a in activations[1:]]))
        #    print('last layer:--->:: {0}'.format(activations[-1]))
            return activations, zs
        



    def backwardpass(self, input_layer, training_output):
        ####PSUEDO CODE
        #input x gradient = signal.convolve2d(filter, loss from last layer) zero padding full convolution
        #filter f gradient = signal.correlate(input x, loss from last layer)
        #convolve2d is reversed correlate 
        a_z = self.forwardpass(input_layer, backprop=True)
        layer_partial_derivative = []
        layer_partial_derivative.append(self.cost_derivative(a_z[0][-1], training_output)*self.layers[-1].act_derivative(a_z[self.layers[-1].act_type][-1]))
        self.layers[-1].weight_gradients = self.layers[-1].weight_gradients + np.dot(layer_partial_derivative[-1], a_z[0][-2].transpose())
        self.layers[-1].bias_gradients = self.layers[-1].bias_gradients + layer_partial_derivative[-1]

   #     print('origin story- -- - - - - : ;: ;-; {0}'.format(layer_partial_derivative[-1]))

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
            #print('indexx {0}'.format(index))
            #print(len(self.layers))
            b = layer.backprop(w, w2, a_z, index, layer_partial_derivative[-1], l)
            [layer_partial_derivative.append(i) for i in b]
            layer_partial_derivative = [i for i in layer_partial_derivative if i is not None]
            #print(layer_partial_derivative[-1].shape)
            #print('a: {0} \n pd: {1}'.format(activations[-1], layer_partial_derivative[0]))
        self.loss.append(self.cel(a_z[0][-1], layer_partial_derivative[0]))
        #print('loss-----------:::: {0}'.format(np.linalg.norm(self.cel(a_z[0][-1], layer_partial_derivative[0]))))
        
    def cel(self, a, y):
        offset = 1e-7
        return y*np.log(a+offset)+(1-y)*np.log(1-(a-offset))



    def update_weights_biases(self, learning_rate, batch_size, threshold):
        for layer in self.layers:
            if layer.id == 0:

              #  print('BEFORE!!!!!! weight: {0} ------- bias: {1}'.format(np.linalg.norm(layer.weight_gradients), np.linalg.norm(layer.bias_gradients)))
                if np.linalg.norm(layer.weight_gradients) > threshold:
                    layer.weight_gradients = layer.weight_gradients*threshold/np.linalg.norm(layer.weight_gradients)
                if np.linalg.norm(layer.bias_gradients) > threshold:
                    layer.bias_gradients = layer.bias_gradients*threshold/np.linalg.norm(layer.bias_gradients)
               # print('AFTER!!!!!! weight: {0} ------- bias: {1}'.format(np.linalg.norm(layer.weight_gradients), np.linalg.norm(layer.bias_gradients)))
 
     #           print('weights: ---- {0} -- biases: {1}'.format(np.linalg.norm(layer.weight_gradients), np.linalg.norm(layer.bias_gradients)))
                layer.weights = layer.weights - (learning_rate/batch_size)*layer.weight_gradients
                layer.biases = layer.biases - (learning_rate/batch_size)*layer.bias_gradients
                layer.weight_gradients = np.zeros(layer.weight_gradients.shape)
                layer.bias_gradients = np.zeros(layer.bias_gradients.shape)
            if layer.id == 1:
                ''''
                for i in layer.filter_gradients:
                    if np.linalg.norm(i) > threshold:
                        layer.filter_gradients[layer.filter_gradients.index(i)] = i*threshold/np.linalg.norm(i)
                for b in layer.bias_gradients:
                    if b > threshold/(layer.filter_shape[0]*layer.filter_shape[1]):
                        layer.bias_gradients[layer.bias_gradients.index(b)] = threshold/(layer.filter_shape[0]*layer.filter_shape[1])
                #print('filters: {0} ------- bias: {1}'.format([np.linalg.norm(i) for i in layer.filter_gradients], np.linalg.norm(layer.bias_gradients)))
                '''
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
                    self.backwardpass(input_data, output_data)
                self.update_weights_biases(learning_rate_r, batch_size, threshold)
            learning_rate_r = self.update_learning_rate(learning_rate, i, cycle_length)  
            print(np.linalg.norm(-sum(self.loss)/len(training_data)))
            self.loss = []
            print(learning_rate_r)
            if test_data:
                print('Epoch {0}: {1}'.format(i, self.test_progress(test_data)/len(test_data)))

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
        return qd





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
    'softmax': lambda y: softmax_derivative(y)
}

def softmax_derivative(y):
    sd = np.zeros((y.size, y.size))
    for i in range(sd.shape[0]):
        for j in range(sd.shape[1]):
            sd[i][j] = y[i]*((i==j)-y[j])
    a = np.reshape(np.sum(sd, axis=1), y.shape)
    return a


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

#NOTE add an id element to use instead of ifinstance since ifinstance fails for some import cases
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
        self.biases = [random.random() for i in range (channels)]
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
            #print(layer_partial_derivative[-1].shape)
                    
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
    #biases are initialized randomly
    #weights are initialized in the convulutionallayer __init__ by the flatten layer
    def __init__(self, size, activation_type):
        self.size = size
        self.weights = None
        self.weight_gradients = None
        self.biases = np.random.randn(size, 1)
        self.bias_gradients = np.zeros(self.biases.shape)
        self.channels = 1
        self.act_func = activation_functions[activation_type]
        self.act_derivative = activation_derivative[activation_type]
        #print(int(activation_type=='softmax'))
        self.act_type = int(activation_type!='softmax')
        self.id = 0
    
    def backprop(self, w, w2, a_z, i, d, l):
        lpd = np.dot(w.transpose(), d)*self.act_derivative(a_z[self.act_type][i])
       # print('lpd: ---- > {0}'.format(np.linalg.norm(lpd)))
        weight_gradient = np.dot(lpd, a_z[0][i-1].transpose())
        bias_gradient = lpd
        self.weight_gradients = self.weight_gradients + weight_gradient
        self.bias_gradients = self.bias_gradients + bias_gradient 
        return [lpd]

    def feedforward(self, a):
        z = np.dot(self.weights, a) + self.biases
        return [z], [self.act_func(z)]


class Flatten(object):
    #flatten buffer between convolutional layer and fully connected layer
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

#net = Network([Input((30, 30), 3), Convolute(1, (7,7), (5, 5)), Flatten(), Dense(100), Dense(10)])
#print(net.layers[5].weights.shape)
#net.backwardpass(np.random.randn(30, 30, 3), np.random.randn(10, 1))

from data_loader import load_data_sign
data = load_data_sign()
random.shuffle(data)


net = Network([Input((64, 64), 1), Convolute(5, (3, 3), (2, 2), 'ReLu'), Convolute(3, (5, 5), (3, 3), 'ReLu'), Flatten(), Dense(750, 'ReLu'), Dense(10, 'softmax')])
#net = Network([Input((64, 64), 1), Flatten(), Dense(1000, 'ReLu'), Dense(100, 'ReLu'), Dense(10, 'softmax')])
#print(net.channels)
#print(net.shapes)
net.train(400, 50, 1, 5, 10, data[:-100], data[-100:])
    
#print([(np.random.randn(96, 96, 3).shape, round(random.uniform(0, 1))) for i in range(1)])
'''net.layers[1].filters[0] = np.array([[1, 1], [1, 1]])

i = np.array([[[1], [0], [2], [3]], [[0], [1], [3], [2]], [[1], [1], [1], [1]], [[2], [3], [0], [1]]])
print(net.layers[1].filters[0].shape)
print(signal.correlate2d(np.reshape(i, (i.shape[0], i.shape[1])), net.layers[1].filters[0], mode='valid'))
w = net.forwardpass(i)[1]'''
#print(np.reshape(w, (w.shape[0], w.shape[1])))
#def vect(i):
#    e = np.zeros((2, 1))
#    e[i] = 1
#    return e
#print(net.test_progress([(np.random.randn(96, 96, 3), vect(round(random.uniform(0, 1)))) for i in range(100)]))
#print(net.test_progress(data[100:200]))

