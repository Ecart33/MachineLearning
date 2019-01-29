import numpy as np
import random

class Network(object):

    def __init__(self, layers):
        self.num_layers = len(layers)
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(layers[:-1], layers[1:])]

    def sgd(self, epochs, batch_size, learning_rate, training_data):
        for epoch in epochs:
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+batch_size]
                            for k in range(0, len(training_data)-batch_size, batch_size]
            for batch in mini_batches:
                for x,y in batch:
                    weight_gradient, bias_gradient = self.backpropagate(x,y)
                    self.update_weights_biases(weight_gradient, bias_gradient, learning_rate)

    def update_weights_biases(self, weight_gradient, bias_gradient, learning_rate):
        self.weights = [w-learning_rate*wg for w, wg in zip(self.weights, weight_gradient)]
        self.bias = [w-learning_rate*wg for w, wg in zip(self.weights, weight_gradient)]

    def backpropagate(self, input_layer, training_output):
        weight_gradient = [np.zeros(w.shape()) for w in sef.weights]
        bias_gradient = [np.zeros(b.shape()) for b in sef.biases]
        activations, zs = self.feedforward(input_layer)
        errors = [sigmoid_derivative(zs[-1])*cost_derivative]
        bias_gradient[-1]=errors[-1]
        weight_gradient[-1]=activations[-2]*errors[-1]
        for i in range(2, self.num_layers):
            error = np.dot(self.weights[-i+1].transpose(), errors[-i+1])*sigmoid_derivative(zs[-i])
            errors.insert(0, error)
            bias_gradient[-i] = error
            weight_gradient[-i] = activations[-i-1]*error
        return weight_gradient, bias_gradient

    def cost_derivative(self, a, y):
        return a-y

    def feedforward(self, input, backprop=False):
        activations = [input]
        z = []
        for w, b in zip(self.weights, self.biases):
            if backprop:
                z.append(np.dot(w, activations[-1].transpose()) + b.transpose()))
            activations.append(self.sigmoid(np.dot(w, activations[-1].transpose()) + b.transpose()))
        if backprop:
            return activations, z
        else:
            return activations[-1]

    def sigmoid_derivative(self, z):
        sigmoid(z)*(1-sigmoid(z))

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
