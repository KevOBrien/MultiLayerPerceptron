import matplotlib.pyplot as plt
import numpy as np

from Model import MLP, DataSet

numEpochs = 10000
numHidden = 5
dataSetSize = 50
learningRate = 0.01
printFreq = 1000


def generateData(numDataPoints):
    data = []
    for i in range(numDataPoints):
        input = np.random.uniform(-1, 1, 4)
        label = np.sin(input[0] - input[1] + input[2] - input[3])
        data.append([input, label])
    return data


dataSet = DataSet(generateData(dataSetSize), 0.8)

mlp = MLP(4, numHidden, 1, 'TANH')

trainLosses, testLosses = [], []

# Train and test
for epoch in range(numEpochs):
    trainLoss, _ = mlp.process(dataSet.trainData, train=True, learningRate=learningRate)
    testLoss, _ = mlp.process(dataSet.testData, train=False)
    trainLosses.append(trainLoss)
    testLosses.append(testLoss)
    if epoch % printFreq == 0 or epoch == numEpochs - 1:
        print("EPOCH:      ", epoch)
        print("TRAIN LOSS: ", trainLoss)
        print("TEST LOSS:  ", testLoss, "\n")


# Generate new random data set of the same format
# ordered by the output values
orderedData = generateData(dataSetSize)
orderedData = sorted(orderedData, key=lambda x: x[1])

# Make a prediction for each input example
predictions, labels = [], []
for data in orderedData:
    predictions.append(mlp.forward(data[0]))
    labels.append(data[1])

# Plot the predictions and true values
predictions = np.array(predictions).flatten()
plt.plot(predictions, label='predictions')
plt.plot(labels, label='labels')
plt.legend()
plt.show()
