# This code trains the Neural Network. In the end, you end up
# with best-fit parameters (weights W1 and W2) for the problem in folder 'data'
# and can use them to predict in predict.py
import numpy as np
import display
from NN import Neural_Network

NN = Neural_Network()

# Loading data
X = np.loadtxt("data/X.in", comments="#", delimiter=",", unpack=False)
y = np.loadtxt("data/y.in", comments="#", delimiter=",", unpack=False)
W1 = np.loadtxt("data/W1.in", comments="#", delimiter=",", unpack=False)
W2 = np.loadtxt("data/W2.in", comments="#", delimiter=",", unpack=False)

# Display inputs
sel = np.random.permutation(len(X));
sel = sel[0:100];
display.displayData(5, X, sel, 'TrainingData');

# Configuring settings of Neural Network:
#
# inputSize, hiddenSize, outputSize = number of elements
#                      in input, hidden, and output layers
# (optional) W1, W2  = random by default
# (optional) maxiter = number of iterations you allow the
#                      optimization algorithm.
#                      By default, set to 20
# (optional) lambd   = regularization penalty. By
#                      default, set to 0.1
#
NN.configureNN(400, 25, 10,
               W1 = W1,
               W2 = W2)

# Training Neural Network on our data
# This step takes 12 mins in Repl.it or 20 sec on your
# computer
NN.train(X, y)

# Saving Weights in the file
NN.saveWeights()

# Checking the accuracy of Neural Network
sel = np.random.permutation(5000)[1:1000]
NN.accuracy(X[sel], y[sel])