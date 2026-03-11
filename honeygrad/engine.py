import math

class Value:
    def __init__(self, data, _prev=(), _op=''):
        self.data = data
        self._prev = _prev
        self._backward = lambda: None
        self.grad = 0.0
        self._op = _op

    def __repr__(self):
        return f"Value({self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(other.data * self.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, exp):
        assert isinstance(exp, (int, float))
        out = Value(self.data ** exp, (self, ), '**')

        def _backward():
            self.grad += exp * self.data ** (exp - 1) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self, ), 'ReLU')

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)
    
    def exp(self):
        out = Value(math.exp(self.data), (self, ))

        def _backward():
            self.grad += out.data * out.grad
        
        out._backward = _backward

        return out

    def tanh(self):
        out = Value(math.tanh(self.data), (self, ))

        def _backward():
            self.grad += (1 - out.data ** 2) * out.grad

        out._backward = _backward

        return out

    def sigmoid(self):
        out = Value(1/(1 + math.exp(-1 * self.data)), (self, ))

        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(x):
            if x not in visited:
                visited.add(x)
                for p in x._prev:
                    build_topo(p)
                topo.append(x)
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()