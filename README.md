# MultiLayerPerceptron

Implementation of the most simple MLP with a single hidden layer written from scratch in Python in an OOP programming style, using no Machine Learning related libraries.

The MLP is constructed in such a way that the user can specify to use Sigmoid, Hyperbolic Tangent, Rectified Linear Unit, or Softmax activation functions in the hidden and output layers.
In the case that Softmax is chosen for the output layer, the user specifies which of the other three to use in the hidden layer.

Includes a class named DataSet, which takes into its constructor a 2-dimensional list of inputs and corresponding labels, as well as a fraction to divide the data into training and test sets.
This makes handling the data easier, and allowed for helpful `MLP.train()` and `MLP.test()` methods to be created, which train and test for one epoch on the entire training and test sets, respectively.
Once the MLP is trained, new unseen data can easily be fed into it using the `MLP.forward()` function, and also retrained if required using `MLP.backwards()`.
