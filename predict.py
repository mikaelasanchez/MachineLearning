import numpy as np
import display
from NN import Neural_Network

NN = Neural_Network()

# Loading data
X = np.loadtxt("data/X.in", comments="#", delimiter=",", unpack=False)
y = np.loadtxt("data/y.in", comments="#", delimiter=",", unpack=False)
trW1 = np.loadtxt("data/TrainedW1.in", comments="#", delimiter=",", unpack=False)
trW2 = np.loadtxt("data/TrainedW2.in", comments="#", delimiter=",", unpack=False)

# Configuring settings of Neural Network:
NN.configureNN(400, 25, 10,
               W1 = trW1,
               W2 = trW2)

# Predicting a class number of given input
testNo = 3402; # any number between 0 and 4999 to test
NN.predictClass(X[testNo])
# Display output
display.displayData(1, X, testNo, 'Predicted class: ' + str(np.argmax(np.round(NN.forward(X[testNo]))) + 1))