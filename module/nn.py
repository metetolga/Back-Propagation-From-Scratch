import math
import numpy as np
import random

class Value:
    def __init__(self, data, _children = (), _op = '', label = '') -> None:
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = _children 
        self._op = _op
        self.label = label

    def __repr__(self) -> str:
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out
    
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), 'only support int and floats powers'
        out = Value(self.data ** other, (self, ), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * other**-1

    def log(self):
        x = self.data
        t = np.log(x)
        out = Value(t, (self, ), 'log')
        def _backward():
            self.grad += (1/x) * out.grad  
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad 
        out._backward = _backward
        return out

    def relu(self):
        x = self.data
        t = max(0, x)
        out = Value(t, (self, ), 'relu')
        def _backward():
            self.grad = 1.0 * out.grad if x >= 0 else out.grad 
        out._backward = _backward 
        return out
    
    def softmax(self):
        pass

    def sigmoid(self):
        x = self.data
        t = 1 / (1 + math.exp(-x))
        out = Value(t, (self,), 'sigmoid')
        def _backward():
            self.grad += t * (1 - t) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()    

def softmax_score(scores):
    exps = np.exp(scores)
    exp_sums = np.sum(exps, axis=1, keepdims=True)
    return np.array(exps / exp_sums)

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x, activation=None):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b) 
        match activation:
            case 'relu':
                out = act.relu()
            case 'tanh':
                out = act.tanh()
            case 'softmax':
                out = act.softmax()
            case 'sigmoid':
                out = act.sigmoid()
            case _:
                out = act
        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, in_shape, out_shape, activation=None):
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.activation = activation
        self.neurons = [Neuron(in_shape) for _ in range(out_shape)]

    def __call__(self, x):
        outs = [n(x, self.activation) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    
class MLP:
    def __init__(self, in_shape, out_shape):
        self.in_shape = in_shape; self.out_shape = out_shape
        self.layers = []
    
    def add_layer(self, x:Layer):
        self.layers.append(x)

    def __call__(self, xs):
        out = []
        for x in xs:
            for layer in self.layers:
                x = layer(x)
            out.append(x)
        return out
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    