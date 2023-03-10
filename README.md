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
  applied. Features like batching/pooling are not defined for this minimal implementation but are easy to add on top of it.
- The final reduction step is a full size convolution with zero padding and 1 feature output, contained directly in the CNN
  class (to turn a set of features as p\*p matrices into a single float for classification purposes). It's easy to introduce more complex classifiers at this step, or adapt the miniconv Layer and ConvolutionNeuron classes to allow striding/less padding to reduce the activation output size gradually. One example would be adapting this convolution into a basic linear classifier by using np.matmul instead of the convolution operation.
- We use a topological sort to arrange the backpropagation method, but a direct
  method is also relatively easy by attaching a layer property to each part of
  the network and searching for children in the next layer down.

### Inspiration

- Inspired by the micrograd MLP implementation by Andrej Karpathy - see the
  [micrograd github page](https://github.com/karpathy/micrograd).
