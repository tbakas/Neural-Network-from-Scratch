# Neural-Network-from-Scratch
Using only Python and NumPy, I built a framework for building MLP neural networks with tensors, automatic back propagation and batch normalization. The file engine.py
contains the tensor and layer classes as well as functions for calculating classification loss. Instead of something like PyTorch's no_grad, I gave layers a predict method
that computes forward passes without any computations for setting up gradient updates.

The Jupyter notebook uses engine.py to create a MLP model and uses it to predict handwritten digits from the MNIST digit dataset. It achieves a 95.5% accuracy on a test set after 30 epochs of training with vanilla gradient descent.

It would be straightforward to add other tensor operations/loss functions to the engine.py file as well.
