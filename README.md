# miniconv

This library is designed to be a minimal implementation of a convolutional
neural network (CNN) in Python with a little basic numpy for random numbers and matrix convenience.

### Usage

- You should not use this library in production code. It will be (much!) slower
  than PyTorch/a more verbose implementation. It is purely for educational
  purposes and helping to understand the structure of a CNN.
  
### Interpretation

- We define a GradFloat class that allows the creation of floats (and ints) with
  an attached wrapper that contains a numerical gradient data point.
- This gradient is calculated by backpropagation from an end node (also of the
  GradFloat class) when the backward() method is called on that object (in the
  CNN implementation this will be from the loss at the end of the neural
  network, but the user can backpropagate from any node manually).
- Standard addition, multiplication and tanh operations are defined on
  this class which extend the operation to calculate the cumulative gradient of
  the end node where backward() is called with respect to that instance of the
  GradFloat class, and to return a new GradFloat object with the correct
  properties.
- Finally we build up the ConvolutionNeuron, Layer, and CNN classes.
- In this implementation, input matrices are fully padded before the kernel is
  applied. There is no batching/pooling/dense network/set of linear classifier
  layers.
- The final reduction step is a linear method contained directly in the CNN
  class (to turn a set of features as p\*p matrices into a binary
  classification)
- We use a topological sort to arrage the backpropagation method, but a direct
  method is also relatively easy by attaching a layer property to each part of
  the network and searching for children in the next layer down

### Inspiration

- inspired by the micrograd MLP implementation by Andrej Karpathy - see the
  [micrograd github page](https://github.com/karpathy/micrograd).
