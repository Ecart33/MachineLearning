import numpy as np
import random

class Network(object):

    def __init__(self, layers):
        self.num_layers = len(layers)
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layers[:-1], layers[1:])]

    def sgd(self, epochs, batch_size, learning_rate, training_data):
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

    def update_weights_biases(self, weight_gradient, bias_gradient, learning_rate, batch_size):
        self.weights = [w-learning_rate/batch_size*wg for w, wg in zip(self.weights, weight_gradient)]
        self.bias = [b-learning_rate/batch_size*bg for b, bg in zip(self.biases, bias_gradient)]

    def backpropagate(self, input_layer, training_output):
        weight_gradient = [np.zeros(w.shape) for w in self.weights]
        bias_gradient = [np.zeros(b.shape) for b in self.biases]
        activations, zs = self.feedforward(input_layer, backprop=True)
        errors = [self.sigmoid_derivative(zs[-1])*self.cost_derivative(activations[-1], training_output)]
        bias_gradient[-1]=errors[-1]
        weight_gradient[-1]=activations[-2]*errors[-1]
        for i in range(2, self.num_layers):
            error = np.dot(self.weights[-i+1].transpose(), errors[-i+1])*self.sigmoid_derivative(zs[-i])
            errors.insert(0, error)
            bias_gradient[-i] = error
            weight_gradient[-i] = activations[-i-1]*error
        return weight_gradient, bias_gradient

    def cost_derivative(self, a, y):
        return a-y

    def feedforward(self, input, backprop=False):
        activations = [np.asarray(input)]
        z = []
        for w, b in zip(self.weights, self.biases):
            if backprop:
                z.append(np.dot(w, activations[-1].transpose()) + b.transpose())
            activations.append(self.sigmoid(np.dot(w, activations[-1].transpose()) + b.transpose()))
        if backprop:
            return activations, z
        else:
            return activations[-1]

    def sigmoid_derivative(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
