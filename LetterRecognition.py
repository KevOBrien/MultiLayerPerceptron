import sys

from Model import MLP, DataSet

# Adjustable parameters
hiddenActivation = 'TANH'
numHidden = 100
numEpochs = 3000
learningRate = 0.0005
updateFreq = 2000
printFreq = 10

alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'


def charToOneHot(char):
    oneHot = [0 for _ in range(len(alpha))]
    oneHot[alpha.index(char)] = 1
    return oneHot


# Read data set
with open('letter-recognition.data', 'r') as file:
    lines = file.readlines()

data = []
for line in lines:
    values = line.split(',')
    # Divide by 15 for Min Max Normalisation
    input = [(int(x.replace('\n', '')) / 15) for x in values[1:]]
    label = charToOneHot(values[0])
    data.append([input, label])

dataSet = DataSet(data, 0.8)

# Load or insantiate new MLP
if len(sys.argv) == 1:
    mlp = MLP(dataSet.numInputs, numHidden, dataSet.numOutputs, outputActivation='SOFTMAX', hiddenActivation=hiddenActivation)
    train = True
elif sys.argv[1] == 'load':
    mlp = MLP(loadModel=True)
    train = False
    numEpochs = 1

# Train and test
for epoch in range(numEpochs):
    trainLoss, trainAccuracy = mlp.process(dataSet.trainData, train=train, learningRate=learningRate, updateFreq=updateFreq)
    if epoch % printFreq == 0 or epoch == numEpochs - 1:
        testLoss, testAccuracy = mlp.process(dataSet.testData, train=False)
        print("EPOCH:      ", epoch)
        print("TRAIN LOSS: ", trainLoss)
        print("TRAIN ACC:  ", trainAccuracy, "%")
        print("TEST LOSS:  ", testLoss)
        print("TEST ACC:   ", testAccuracy, "%\n")
