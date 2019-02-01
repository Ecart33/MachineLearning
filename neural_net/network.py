import numpy as np
import random

class Network(object):

    def __init__(self, layers):
        self.num_layers = len(layers)
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

    def sgd(self, epochs, batch_size, learning_rate, training_data):
        for epoch in range(epochs+1):
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
            print('Epoch {0}: {1}/{2}'.format(epoch, self.progress(training_data), len(training_data)))

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
        errors = [self.cost_derivative(activations[-1], training_output) * self.sigmoid_derivative(zs[-1])]
        bias_gradient[-1]=errors[0]
        weight_gradient[-1]= np.dot(errors[0], activations[-2].transpose())
        for i in range(2, self.num_layers):
            #transpose b/c connecting same layer so rows and columns should be switched
            #usually weight rows represent next layer and columns represent current layer
            error = np.dot(self.weights[-i+1].transpose(), errors[i-2]) * self.sigmoid_derivative(zs[-i])
            errors.append(error)
            bias_gradient[-i] = errors[i-2]
            weight_gradient[-i] = np.dot(errors[i-1], activations[-i-1].transpose())
        return weight_gradient, bias_gradient

    def cost_derivative(self, a, y):
        return (a-y)

    def feedforward(self, input, backprop=False):
        activations = [np.reshape(input, (input.size, 1))]
        z = []
        for w, b in zip(self.weights, self.biases):
            if backprop:
                z.append(np.dot(w, activations[-1]) + b)
            activations.append(self.sigmoid(np.dot(w, activations[-1]) + b))
        if backprop:
            return activations, z
        else:
            return activations[-1]

    def sigmoid_derivative(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))

from data_loader import load_data
e = load_data()
net = Network([27648, 100, 100, 1])
net.sgd(30, 20, 1.5, e)
