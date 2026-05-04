import numpy


def get_sum_dimensions(input_shape, output_shape):
    # This determines what dimensions to sum over in backward pass when broadcasting occured in forward pass.
    if input_shape == output_shape:
        return ()

    n_input = len(input_shape)
    n_output = len(output_shape)
    padded_input_shape = (1,) * (n_output - n_input) + input_shape

    sum_dimensions = []
    for d in range(n_output):
        if output_shape[d] != padded_input_shape[d]:
            sum_dimensions.append(d)
    return tuple(sum_dimensions)


class Tensor:
    def __init__(self, data, children=(), requires_grad=False):
        assert isinstance(data, numpy.ndarray), 'the data is not of type ndarray'
        self.data = data
        self.shape = data.shape
        self.children = set(children)

        self.requires_grad = requires_grad
        if requires_grad:
            self.grad = numpy.zeros(data.shape)

        self._backward = lambda: None

    def __repr__(self):
        return self.data.__repr__()
    
    def __getitem__(self, item):
        # for getting batches from data tensors
        return Tensor(numpy.array(self.data[item]))

    def __add__(self, other):
        # tensor addition
        out = Tensor(self.data + other.data, (self, other), requires_grad=True)

        def backward():
            if self.requires_grad:
                sum_dimensions = get_sum_dimensions(self.shape, out.shape)
                self.grad += out.grad.sum(sum_dimensions)
            if other.requires_grad:
                sum_dimensions = get_sum_dimensions(other.shape, out.shape)
                other.grad += out.grad.sum(sum_dimensions)

        out._backward = backward

        return out

    def __matmul__(self, other):
        # matrix multiplication
        out = Tensor(self.data @ other.data, (self, other), requires_grad=True)

        def backward():
            if self.requires_grad:
                update_array = out.grad @ numpy.moveaxis(other.data, -1, -2)
                sum_dimensions = get_sum_dimensions(self.shape, update_array.shape)
                self.grad += update_array.sum(axis=sum_dimensions)

            if other.requires_grad:
                update_array = numpy.moveaxis(self.data, -1, -2) @ out.grad
                sum_dimensions = get_sum_dimensions(other.shape, update_array.shape)
                other.grad += update_array.sum(axis=sum_dimensions)

        out._backward = backward

        return out

    def relu(self):
        # non-linear activation function
        non_negative = numpy.where(self.data >= 0, 1, 0)
        out = Tensor(self.data * non_negative, (self,), requires_grad=True)

        def backward():
            if self.requires_grad:
                self.grad += out.grad * non_negative

        out._backward = backward

        return out
        
    def get_topological_sort(self):
        # This returns a list where child nodes are always before their parents for updating tensor gradients in the
        # correct order.
        topo_sort = []
        visited = set()

        def build(t):
            if t not in visited:
                visited.add(t)
                for child in t.children:
                    build(child)
                topo_sort.append(t)

        build(self)
        return topo_sort
        
    def backward(self):
        # This determines a valid order for updating the gradients,
        # zeros gradients and updates each tensor's gradient
        topo_sort = self.get_topological_sort()

        for tensor_object in topo_sort[:-1]:
            tensor_object.zero_grad()

        self.grad.fill(1)

        for tensor_object in reversed(topo_sort):
            tensor_object._backward()

    def zero_grad(self):
        if self.requires_grad:
            self.grad.fill(0)


def classification_loss(logits, targets):
    # For calculating the negative log likelihood of a classifier given its logits and the targets.
    # This assumes targets is an 1D array with values 0 to C - 1 where C is the number of classes.
    n = logits.shape[0]
    batch_index = numpy.arange(n)
    dummy = numpy.exp(logits.data - logits.data.max(axis=-1, keepdims=True))
    probabilities = dummy / dummy.sum(axis=-1, keepdims=True)

    neg_log_like = - numpy.log(probabilities[batch_index, targets.data])
    out = Tensor(numpy.array(neg_log_like.mean()), (logits,), requires_grad=True)
    
    # Calculates logits gradient directly.
    # out.grad assumed to be 1
    def backward():
        p_grad = numpy.zeros(probabilities.shape)
        p_grad[batch_index, targets.data] = - 1 / (n * probabilities[batch_index, targets.data])
        logits.grad = probabilities * (p_grad - (probabilities * p_grad).sum(axis=-1, keepdims=True))

    out._backward = backward

    return out

            
def relu_no_grad(x):
    # A relu activation without any gradient calculations for testing.
    return x.data * numpy.where(x.data >= 0, 1, 0)


def predict_class(logits):
    # Gives predictions of classifier without any gradient calculations for testing.
    dummy = numpy.exp(logits.data - logits.data.max(axis=-1, keepdims=True))
    probabilities = dummy / dummy.sum(axis=-1, keepdims=True)

    return probabilities.argmax(axis=-1)


def predict_probabilities(logits):
    # The same as previous function but it returns the probabilities instead of the predictions.
    dummy = numpy.exp(logits.data - logits.data.max(axis=-1, keepdims=True))
    probabilities = dummy / dummy.sum(axis=-1, keepdims=True)

    return probabilities
        

class Linear:
    # A basic linear layer with weights and a bias.
    def __init__(self, n_input, n_output, has_bias=True):
        # Initializes weights with a normal distribution.
        weights = numpy.random.normal(size=(n_input, n_output)) / n_input ** 0.5
        self.weights = Tensor(weights, requires_grad=True)
        self.n_parameters = n_input * n_output
        
        self.has_bias = has_bias
        if has_bias:
            self.bias = Tensor(numpy.zeros(n_output), requires_grad=True)
            self.n_parameters += n_output

    def forward(self, x):
        if self.has_bias:
            return x @ self.weights + self.bias
        else:
            return x @ self.weights

    def update(self, learning_rate):
        # This updates the weights/biases of the layer using regular gradient descent.
        # One could add more complicated optimizers if desired.
        self.weights.data -= learning_rate * self.weights.grad
        if self.has_bias:
        	self.bias.data -= learning_rate * self.bias.grad

    def predict(self, x):
        # Performs forward pass without any gradient computations. Used for testing.
        if self.has_bias:
            return Tensor(x.data @ self.weights.data + self.bias.data)
        else:
            return Tensor(x.data @ self.weights.data)

        
class Activation:
    # Activation layer.
    def __init__(self):
        self.n_parameters = 0
    
    def forward(self, x):
        return x.relu()
    
    def update(self, learning_rate):
        return None
    
    def predict(self, x):
        return Tensor(relu_no_grad(x))
    

class BatchNorm1d:
    # A batchnorm1d layer for standardizing data during training.
    # This uses an unbiased variance estimate.
    def __init__(self, n_features, epsilon=1e-5, momentum=0.1):
        self.gamma = Tensor(numpy.ones(n_features), requires_grad=True)
        self.beta = Tensor(numpy.zeros(n_features), requires_grad=True)
        
        self.n_parameters = 2 * n_features

        self.epsilon = epsilon
        self.momentum = momentum

        self.running_mean = numpy.zeros(n_features)
        self.running_variance = numpy.ones(n_features)

    def forward(self, x):
        # This should only can be used for training. Use predict for testing.
        
        mean = x.data.mean(axis=0)
        variance = x.data.var(axis=0, ddof=1)
        self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
        self.running_variance = (1 - self.momentum) * self.running_variance + self.momentum * variance

        x_norm = (x.data - mean) / numpy.sqrt(variance + self.epsilon)
        y = Tensor(self.gamma.data * x_norm + self.beta.data, children=(self.gamma, self.beta, x), requires_grad=True)
        n = x_norm.shape[0]

        def backward():
            # calculates gradient directly without the need of other tensor operations such as multiplication
            self.gamma.grad += (y.grad * x_norm).sum(axis=0)
            self.beta.grad += y.grad.sum(axis=0)

            if x.requires_grad:
                x.grad += (self.gamma.data / numpy.sqrt(variance + self.epsilon)) * (
                        y.grad - numpy.sum(y.grad, axis=0) / n - x_norm * numpy.sum(x_norm * y.grad, axis=0) / (n - 1))

        y._backward = backward

        return y

    def update(self, learning_rate):
        self.gamma.data -= learning_rate * self.gamma.grad
        self.beta.data -= learning_rate * self.beta.grad

    def predict(self, x):
        mean = self.running_mean
        variance = self.running_variance

        x_norm = (x.data - mean) / numpy.sqrt(variance + self.epsilon)
        y = self.gamma.data * x_norm + self.beta.data
        return Tensor(y)
