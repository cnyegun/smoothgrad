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
    def parameters(self):
        return self.weights + [self.bias]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]
    def __call__(self, xs):
        outs = [n(xs) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
    def parameters(self):
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params

class MLP:
    def __init__(self, nin, nouts):
        _size = [nin] + nouts
        self.layers = [Layer(_size[i], _size[i + 1]) for i in range(len(_size) - 1)]
    def __call__(self, xs):
        for layer in self.layers:
            xs = layer(xs)
        return xs
    def parameters(self):
        params = []
        for l in self.layers:
            params.extend(l.parameters())
        return params


random.seed(1337)
nn = MLP(3, [4,4,1])
params = nn.parameters()

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

step_size = 0.05
training_set = 100 
for _ in range(training_set):
    ypred = [nn(x) for x in xs]
    loss = sum((ygt - yout)**2 for ygt, yout in zip(ys, ypred))

    # always zero out all the grad before do backward pass
    for p in params:
        p.grad = 0.0
    loss.backward()

    print("target:", ys)
    print("output:", ypred)
    print(f"loss:   {loss.data:.5f}", )

    for p in params:
        p.data += -step_size * p.grad

print("final params:", nn.parameters())