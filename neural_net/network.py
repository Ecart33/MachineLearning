import numpy as np
import random

class Network(object):

    def __init__(self, layers):
        self.num_layers = len(layers)
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layers[:-1], layers[1:])]

    #takes one argument (z). i is object type
    def sigmoid(i, z):
        return 1.0/(1.0+np.exp(-z))

    def sgd(self, training_data, epochs, mini_batch_size, learning_rate):
        n = len(training_data)
        for j in xrange(epochs)
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n)]
            for mini_batch in mini_batches:
                self.update_weights_and_biases(mini_batch, learning_rate)
        




    #given activations for input layer and weights and biases, returns output layer
    #if backprop specified, returns list of all z vectors (weight*activation + bias)
    #and activations for all layers
    def feedforward(self, inputLayer, backprop=False):
        a = inputLayer
        l = np.empty([1,0])
        if backprop:
            zs = []
            activations = [inputLayer]
        for w, b in zip(self.weights, self.biases):
            f = np.matmul(w,a)
            for i in (f[:,None]+b):
                l = np.append(l, self.sigmoid(i))
                if backprop:
                    zs.append(i)
            a = l
            if backprop:
                activations.append(a)
            l = np.empty([1,0])
        if backprop:
            return activations, zs
        else:
            return a

    #cost function for single training example
    def cost_single(self, output, training_output):
        output_differnces = []
        for neuron, t_neuron in zip(output, training_output):
            output_differnces.append((neuron-t_neuron)^2)
        return sum(output_differnces)/2

    #update weights and biases
    def update_weights_and_biases(self, mini_batch, learning_rate):
        #initiate empty matrices of weights and biases
        n_w = [np.zeros(w.shape) for w in self.weights]
        n_b = [np.zeros(b.shape) for b in self.biases]
        for input, output in mini_batch:
            #(dn_w, dn_b) is gradient of cost function
            #layer by layer arrays of gradient weights and biases
            dn_w, dn_b = self.backprop(input, output)
            #set n_w and n_b to contain all gradients of weights and biases
            n_w = [nw+dnw for nw, dnw in zip(n_w, dn_w)]
            n_b = [nb+dnb for nb, dnb in zip(n_b, dn_b)]
        #update weights and biases based off of gradient descent formulas
        self.weights = [w-(learning_rate/len(mini_batch))*nw for
                        w, nw in zip(self.weights, n_w)]
        self.biases = [b-(learning_rate/len(mini_batch))*nb for
                        b, nb in zip(self.biases, n_b)]

    #figure out gradient of cost function
    def backprop(input, output):
        n_w = [np.zeros(w.shape) for w in self.weights]
        n_b = [np.zeros(b.shape) for b in self.biases]

        #feedforward
        activations, zs = self.feedforward(input, backprop=True)

        #calculate error for each layer
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])

        n_b[-1] = delta
        n_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            #sp is error for layer
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            #gradients
            n_b[-l] = delta
            n_w[-l] = np.dot(delta, activations[-l-1].transpose())
        #return gradients of each layer
        return (nabla_b, nabla_w)


    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

    #inverse of sigmoid function
    def sigmoid_prime(z):
        return sigmoid(z)*(1-sigmoid(z))
