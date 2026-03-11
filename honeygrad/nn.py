import random
from engine import Value

class Neuron:
    def __init__(self, nin):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(random.uniform(-1, 1))
    def __call__(self, xs):
        assert len(xs) == len(self.weights)
        act = self.bias
        for w, x in zip(self.weights, xs):
            act += w * x
        out = act.tanh()
        return out

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self, xs):
        outs = [n(xs) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

class MLP:
    def __init__(self, nin, nouts):
        _size = [nin] + nouts
        self.layers = [Layer(_size[i], _size[i + 1]) for i in range(len(_size) - 1)]
    def __call__(self, xs):
        for layer in self.layers:
            xs = layer(xs)
        return xs

nn = MLP(3, [4,4,1])
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

ypred = [nn(x) for x in xs]
loss = sum((ygt - yout)**2 for ygt, yout in zip(ys, ypred))

print("target:", ys)
print("output:", ypred)
print("loss:  ", loss)