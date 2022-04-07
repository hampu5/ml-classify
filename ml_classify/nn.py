import numpy as np

def sigmoid(x): 1/(1 + np.exp(-x))

def relu(x): np.maximum(x, 0, x)

class Network(object):
    def __init__(self, *args, **kwargs):
        input_size = kwargs.get('input_size', None)
        hidden_size = kwargs.get('hidden_size', None)
        output_size = kwargs.get('output_size', None)
        self.weights = [
            np.random.rand(input_size, hidden_size).T,
            np.random.rand(hidden_size, output_size).T
        ]
        self.biases = [
            np.zeros(hidden_size),
            np.zeros(output_size)
        ]
        self.activation = kwargs.get('activation', sigmoid)
        # self.activation_derivative = kwargs.get('activation_derivative', sigmoid)
    
    def feedforward(self, a):
        """Return the output of the network for an input vector a"""
        for w, b in zip(self.weights, self.biases):
            a = np.dot(w, a) + b
            # print(a)
            self.activation(a)
        return a
    
    def backpropagation(self, x, y):
        """Return a list of gradients for each layer"""
        gradients = []
        # feedforward
        activations = [x]
        for w, b in zip(self.weights, self.biases):
            a = np.dot(w, activations[-1]) + b
            activations.append(self.activation(a))
        # backward pass
        delta = activations[-1] - y
        gradients.append(np.dot(delta, activations[-2].T) / len(x))
        for l in range(2, len(self.weights) + 1):
            delta = np.dot(self.weights[-l + 1].T, delta) * self.activation_derivative(activations[-l])
            gradients.append(np.dot(delta, activations[-l - 1].T) / len(x))
        return gradients
    
    def fit(self, X, y, epochs=10, learning_rate=0.1):
        """Fit the network to the training data"""
        for epoch in range(epochs):
            for x, y in zip(X, y):
                activations = [x]
                for w, b in zip(self.weights, self.biases):
                    a = np.dot(w, activations[-1]) + b
                    activations.append(self.activation(a))
                gradients = self.backpropagation(activations, y)
                for w, g in zip(self.weights, gradients):
                    w -= learning_rate * g



X_train = np.array([0.1, 0.1, 0.1, 0.1])
y_train = np.array([1])

nn = Network(input_size=4, hidden_size=3, output_size=1, activation=sigmoid)

print(X_train)

data = nn.fit(X_train, y_train)

print(data)