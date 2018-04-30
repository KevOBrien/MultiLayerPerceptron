import os
import sys

import numpy as np

projectDir = os.path.dirname(os.path.realpath(__file__))


class DataSet:
    # DataSet class makes for easier integration with MLP class.

    def __init__(self, data, trainSplit=1.0):
        # 'data' = 2-D list of [inputs, labels]
        # 'trainSplit' = fraction of data to be used for train/test sets
        trainSize = int(len(data) * trainSplit)
        testSize = len(data) - trainSize
        inputs, labels = [], []
        for i in range(len(data)):
            inputs.append(data[i][0])
            labels.append(data[i][1])

        # Dicts holding the inputs, labels, and size of
        # the train and test data
        self.trainData, self.testData = {}, {}
        self.trainData['inputs'] = inputs[:trainSize]
        self.trainData['labels'] = labels[:trainSize]
        self.trainData['size'] = trainSize
        self.testData['inputs'] = inputs[trainSize:]
        self.testData['labels'] = labels[trainSize:]
        self.testData['size'] = testSize

        # Number of inputs and outputs required for the MLP
        try:
            self.numInputs = len(inputs[0])
        except TypeError:
            self.numInputs = 1
        try:
            self.numOutputs = len(labels[0])
        except TypeError:
            self.numOutputs = 1


class MLP:

    def __init__(self, numInputs=0, numHidden=0, numOutputs=0, outputActivation=None, hiddenActivation=None, loadModel=False):
        # if 'loadModel' = True, no other parameters need to be provided
        if loadModel:
            self.load()
        else:
            self.build(numInputs, numHidden, numOutputs, outputActivation, hiddenActivation)

    def build(self, numInputs, numHidden, numOutputs, outputActivation, hiddenActivation):
        # Initialises all parameters of the model
        if outputActivation == 'SOFTMAX' and hiddenActivation is None:
            print("Hidden activation function must be specified when using softmax output.")
            sys.exit()

        # Activation functions at ouput and hidden layers
        self.outputActivation = outputActivation
        self.hiddenActivation = hiddenActivation

        # Saves model when new lowest test loss is reached
        self.bestLoss = 100

        # Architecture sizes
        self.numInputs = numInputs
        self.numHidden = numHidden
        self.numOutputs = numOutputs

        # Weights
        self.IH_weights = np.random.rand(self.numInputs, self.numHidden)    # Input -> Hidden
        self.HO_weights = np.random.rand(self.numHidden, self.numOutputs)   # Hidden -> Output

        # Biases
        self.IH_bias = np.zeros((1, self.numHidden))    # Input -> Hidden
        self.HO_bias = np.zeros((1, self.numOutputs))   # Hidden -> Output

        # Weights gradients computed during backprop
        self.IH_w_gradients = np.zeros_like(self.IH_weights)
        self.HO_w_gradients = np.zeros_like(self.HO_weights)

        # Biases gradients computed during backprop
        self.IH_b_gradients = np.zeros_like(self.IH_bias)
        self.HO_b_gradients = np.zeros_like(self.HO_bias)

    def activation(self, x, hiddenLayer=False, derivative=False):
        if (self.outputActivation == 'SIGMOID' or
                (hiddenLayer and self.hiddenActivation == 'SIGMOID')):
            if derivative:
                return x * (1 - x)
            return 1 / (1 + np.exp(-x))

        if (self.outputActivation == 'TANH' or
                (hiddenLayer and self.hiddenActivation == 'TANH')):
            if derivative:
                return 1 - np.tanh(x) ** 2
            return np.tanh(x)

        if (self.outputActivation == 'RELU' or
                (hiddenLayer and self.hiddenActivation == 'RELU')):
            if derivative:
                return (x > 0).astype(float)
            return np.maximum(0, x)

        if self.outputActivation == 'SOFTMAX':
            if derivative:
                return 1                            # Derivative not used, but needed for multiplication
            exps = np.exp(x - np.max(x))            # Allows for large values
            return (exps / np.sum(exps)) + 1e-10    # Allows for log(0) in loss function

        print("ERROR: \'" + self.outputActivation + "\' activation function not found.")
        sys.exit()

    def loss(self, prediction, label, derivative=False):
        # Softmax Cross Entropy
        if self.outputActivation == 'SOFTMAX':
            if derivative:
                return label - prediction
            return - np.sum([t * np.log(y) for y, t in zip(prediction, label)])

        # Squared Error
        else:
            if derivative:
                return 2 * (label - prediction)
            return (label - prediction) ** 2

    def forward(self, input):
        # Input neurons
        self.I = np.array(input).reshape(1, self.numInputs)
        # Hidden neurons
        self.H = self.activation(self.I.dot(self.IH_weights) + self.IH_bias, hiddenLayer=True)
        # Output neurons
        self.O = self.activation(self.H.dot(self.HO_weights) + self.HO_bias)
        return self.O

    def backwards(self, label):
        # Labels
        self.L = np.array(label).reshape(1, self.numOutputs)
        # Error between output and labels
        self.O_error = self.loss(self.O, self.L)
        # Partial derivative at output layer
        self.O_delta = self.loss(self.O, self.L, derivative=True) * self.activation(self.O, derivative=True)
        # Error at hidden layer
        self.H_error = self.O_delta.dot(self.HO_weights.T)
        # Partial derivative at hidden layer
        self.H_delta = self.H_error * self.activation(self.H, hiddenLayer=True, derivative=True)
        # Gradients of weights and biases in magnitude and direction
        # in order to reduce loss
        self.IH_w_gradients += self.I.T.dot(self.H_delta)
        self.HO_w_gradients += self.H.T.dot(self.O_delta)
        self.IH_b_gradients += self.H_delta
        self.HO_b_gradients += self.O_delta
        return self.O_error

    def updateWeights(self, learningRate):
        # Update weights and biases using gradients computed in backprop
        self.IH_weights += learningRate * self.IH_w_gradients
        self.HO_weights += learningRate * self.HO_w_gradients
        self.IH_bias += learningRate * self.IH_b_gradients
        self.HO_bias += learningRate * self.HO_b_gradients
        # Reset gradient values
        self.IH_w_gradients = np.zeros_like(self.IH_weights)
        self.HO_w_gradients = np.zeros_like(self.HO_weights)
        self.IH_b_gradients = np.zeros_like(self.IH_bias)
        self.HO_b_gradients = np.zeros_like(self.HO_bias)

    # Processes one entire epoch of train/test data
    def process(self, data, train=True, learningRate=0, updateFreq=0, save=False):
        # 'data' = train or test dict from DataSet class
        # 'updateFreq' = number of examples after which to update weights if training
        if train:
            if learningRate == 0:
                print("No learning rate provided for training.")
                sys.exit()
            if updateFreq == 0:     # Update only after entire epoch
                updateFreq = data['size'] - 1

        losses, accuracies = [], []
        for i in range(data['size']):
            prediction = self.forward(data['inputs'][i])
            label = data['labels'][i]
            loss = self.backwards(label)
            losses.append(loss)
            if self.outputActivation == 'SOFTMAX':
                accuracies.append(np.argmax(prediction) == np.argmax(label))
            else:
                accuracies.append(np.round(prediction) == label)
            if train and i % updateFreq == 0:
                self.updateWeights(learningRate)
        epochLoss = np.mean(losses)
        epochAccuracy = "%.2f" % (np.mean(accuracies) * 100)
        if not train and save and epochLoss < self.bestLoss:
            self.bestLoss = epochLoss
            self.save()
        # Accuracy only relevant for classification tasks, not regression
        return epochLoss, epochAccuracy

    def save(self):
        # Saves all the parameters of the model's current state
        if not os.path.exists(projectDir + '/SavedModel'):
            os.makedirs(projectDir + '/SavedModel')

        np.save(projectDir + '/SavedModel/bestLoss.npy', self.bestLoss)
        np.save(projectDir + '/SavedModel/numInputs.npy', self.numInputs)
        np.save(projectDir + '/SavedModel/numHidden.npy', self.numHidden)
        np.save(projectDir + '/SavedModel/numOutputs.npy', self.numOutputs)
        np.save(projectDir + '/SavedModel/outputActivation.npy', self.outputActivation)
        np.save(projectDir + '/SavedModel/hiddenActivation.npy', self.hiddenActivation)
        np.save(projectDir + '/SavedModel/IH_weights.npy', self.IH_weights)
        np.save(projectDir + '/SavedModel/HO_weights.npy', self.HO_weights)
        np.save(projectDir + '/SavedModel/IH_bias.npy', self.IH_bias)
        np.save(projectDir + '/SavedModel/HO_bias.npy', self.HO_bias)

    def load(self):
        # Loads a pre-trained model's parameters
        for file in ['IH_weights.npy', 'HO_weights.npy', 'IH_bias.npy', 'HO_bias.npy']:
            if not os.path.exists(projectDir + '/SavedModel/' + file):
                print('SavedModels/' + file + " does not exist.")
                sys.exit()

        self.bestLoss = np.load(projectDir + '/SavedModel/bestLoss.npy')
        self.numInputs = np.load(projectDir + '/SavedModel/numInputs.npy')
        self.numHidden = np.load(projectDir + '/SavedModel/numHidden.npy')
        self.numOutputs = np.load(projectDir + '/SavedModel/numOutputs.npy')
        self.outputActivation = np.load(projectDir + '/SavedModel/outputActivation.npy')
        self.hiddenActivation = np.load(projectDir + '/SavedModel/hiddenActivation.npy')
        self.IH_weights = np.load(projectDir + '/SavedModel/IH_weights.npy')
        self.HO_weights = np.load(projectDir + '/SavedModel/HO_weights.npy')
        self.IH_bias = np.load(projectDir + '/SavedModel/IH_bias.npy')
        self.HO_bias = np.load(projectDir + '/SavedModel/HO_bias.npy')

        self.IH_w_gradients = np.zeros_like(self.IH_weights)
        self.HO_w_gradients = np.zeros_like(self.HO_weights)
        self.IH_b_gradients = np.zeros_like(self.IH_bias)
        self.HO_b_gradients = np.zeros_like(self.HO_bias)
