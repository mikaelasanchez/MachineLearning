import numpy as np
from scipy.optimize import minimize


class Neural_Network(object):

    def configureNN(self, inputSize, hiddenSize, outputSize, W1=np.array([0]), W2=np.array([0]),
                    maxiter=20, lambd=0.1):
        # parameters
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.hiddenSize = hiddenSize

        # initialize weights / random by default
        if (not W1.any()):
            self.W1 = np.random.randn(
                self.hiddenSize,
                self.inputSize + 1)  # weight matrix from input to hidden layer
        else:
            self.W1 = W1
        if (not W2.any()):
            self.W2 = np.random.randn(
                self.outputSize,
                self.hiddenSize + 1)  # weight matrix from hidden to output layerself.W2 = W2
        else:
            self.W2 = W2

        # maximum number of iterations for optimization algorithm
        self.maxiter = maxiter
        # regularization penalty
        self.lambd = lambd

    def addBias(self, X):
        # adds a column of ones to the beginning of an array
        if (X.ndim == 1): return np.insert(X, 0, 1)
        return np.concatenate((np.ones((len(X), 1)), X), axis=1)

    def delBias(self, X):
        # deletes a column from the beginning of an array
        if (X.ndim == 1): return np.delete(X, 0)
        return np.delete(X, 0, 1)

    def unroll(self, X1, X2):
        # unrolls two matrices into one vector
        return np.concatenate((X1.reshape(X1.size), X2.reshape(X2.size)))

    def sigmoid(self, s):
        # activation function
        return 1 / (1 + np.exp(-s))

    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s * (1 - s)

    def forward(self, X):
        # forward propagation through our network
        X = self.addBias(X)
        self.z = np.dot(
            X,
            self.W1.T)  # dot product of X (input) and first set of 3x2 weights
        self.z2 = self.sigmoid(self.z)  # activation function
        self.z2 = self.addBias(self.z2)
        self.z3 = np.dot(
            self.z2,
            self.W2.T)  # dot product of hidden layer (z2) and second set of 3x1 weights
        o = self.sigmoid(self.z3)  # final activation function
        return o

    def backward(self, X, y, o):
        # backward propgate through the network
        self.o_delta = o - y  # error in output

        self.z2_error = self.o_delta.dot(
            self.W2
        )  # z2 error: how much our hidden layer weights contributed to output error
        self.z2_delta = np.multiply(self.z2_error, self.sigmoidPrime(
            self.z2))  # applying derivative of sigmoid to z2 error
        self.z2_delta = self.delBias(self.z2_delta)

        self.W1_delta += np.dot(
            np.array([self.z2_delta]).T, np.array([self.addBias(X)]))  # adjusting first set (input --> hidden) weights
        self.W2_delta += np.dot(
            np.array([self.o_delta]).T, np.array([self.z2]))  # adjusting second set (hidden --> output) weights

    def cost(self, nn_params, X, y):
        # computing how well the function does. Less = better
        self.W1_delta = 0
        self.W2_delta = 0
        m = len(X)

        o = self.forward(X)
        J = -1 / m * sum(sum(y * np.log(o) + (1 - y) * np.log(1 - o)));  # cost function
        reg = (sum(sum(np.power(self.delBias(self.W1), 2))) + sum(
            sum(np.power(self.delBias(self.W2), 2)))) * (self.lambd / (2 * m));  # regularization: more precise
        J = J + reg;

        for i in range(m):
            o = self.forward(X[i])
            self.backward(X[i], y[i], o)
        self.W1_delta = (1 / m) * self.W1_delta + (self.lambd / m) * np.concatenate(
            (np.zeros((len(self.W1), 1)), self.delBias(self.W1)), axis=1)
        self.W2_delta = (1 / m) * self.W2_delta + (self.lambd / m) * np.concatenate(
            (np.zeros((len(self.W2), 1)), self.delBias(self.W2)), axis=1)

        grad = self.unroll(self.W1_delta, self.W2_delta)

        return J, grad

    def train(self, X, y):
        # using optimization algorithm to find best fit W1, W2
        nn_params = self.unroll(self.W1, self.W2)
        results = minimize(self.cost, x0=nn_params, args=(X, y),
                           options={'disp': True, 'maxiter': self.maxiter}, method="L-BFGS-B", jac=True)

        self.W1 = np.reshape(results["x"][:self.hiddenSize * (self.inputSize + 1)],
                             (self.hiddenSize, self.inputSize + 1))

        self.W2 = np.reshape(results["x"][self.hiddenSize * (self.inputSize + 1):],
                             (self.outputSize, self.hiddenSize + 1))

    def saveWeights(self):
        # sio.savemat('myWeights.mat', mdict={'W1': self.W1, 'W2' : self.W2})
        np.savetxt('data/TrainedW1.in', self.W1, delimiter=',')
        np.savetxt('data/TrainedW2.in', self.W2, delimiter=',')

    def predict(self, X):
        o = self.forward(X)
        i = np.argmax(o)
        o = o * 0
        o[i] = 1
        return o

    def predictClass(self, X):
        # printing out the number of the class, starting from 1
        print("Predicted class out of", self.outputSize, "classes based on trained weights: ")
        print("Input: \n" + str(X))
        print("Class number: " + str(np.argmax(np.round(self.forward(X))) + 1))

    def accuracy(self, X, y):
        # printing out the accuracy
        p = 0
        m = len(X)
        for i in range(m):
            if (np.all(self.predict(X[i]) == y[i])): p += 1

        print('Training Set Accuracy: {:.2f}%'.format(p * 100 / m))