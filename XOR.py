import sys

from Model import MLP, DataSet

if len(sys.argv) not in [2, 3]:
    print("Run program as \'python XOR.py <output activation> [<hidden activation>]\'.")
    sys.exit()

outputActivation = sys.argv[1].upper()

if len(sys.argv) == 3:
    hiddenActivation = sys.argv[2].upper()
else:
    hiddenActivation = None

numEpochs = 10000
printFreq = 1000

# SIGMOID & RELU
dataSet1 = DataSet([
    [[0, 0], 0],
    [[0, 1], 1],
    [[1, 0], 1],
    [[1, 1], 0]
])

# TANH
dataSet2 = DataSet([
    [[0, 0], -1],
    [[0, 1], 1],
    [[1, 0], 1],
    [[1, 1], -1]
])

# SOFTMAX
dataSet3 = DataSet([
    [[0, 0], [1, 0]],
    [[0, 1], [0, 1]],
    [[1, 0], [0, 1]],
    [[1, 1], [1, 0]]
])

# Best learning rates for each determined through experimenting
if outputActivation == 'SIGMOID':
    dataSet = dataSet1
    learningRate = 1
if outputActivation == 'TANH':
    dataSet = dataSet2
    learningRate = 1
if outputActivation == 'RELU':
    dataSet = dataSet1
    learningRate = 0.01
if outputActivation == 'SOFTMAX':
    dataSet = dataSet3
    learningRate = 0.1
if 'dataSet' not in list(locals()) + list(globals()):
    print("\'" + outputActivation + "\' is not a valid activation function.")
    sys.exit()

mlp = MLP(dataSet.numInputs, 2, dataSet.numOutputs, outputActivation, hiddenActivation)

for epoch in range(numEpochs):
    epochLoss, epochAccuracy = mlp.process(dataSet.trainData, train=True, learningRate=learningRate)
    if epoch % printFreq == 0:
        print("EPOCH:      ", epoch)
        print("TRAIN LOSS: ", epochLoss)
        print("TRAIN ACC:  ", epochAccuracy, "%\n")
