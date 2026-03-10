import math
import random

class Value:
    def __init__(self, data, _prev=()):
        self.data = data
        self._prev = _prev
        self._backward = lambda: None
        self.grad = 0.0

    def __str__(self):
        return f"Value({self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, (self, other))
        return out

    def __rsub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(other.data - self.data, (self, other))
        return out
        
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(other.data * self.data, (self, other))
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, exp):
        assert isinstance(exp, (int, float))
        out = Value(self.data ** exp, (self, ))
        return out

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)
    
    def exp(self):
        out = Value(math.exp(self.data), (self, ))
        return out
