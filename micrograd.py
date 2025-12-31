## Import required libraries
import math as m
import random as r
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## Make class to store number with it's parents and operations and define all functions and backward function (backpropagation)
class Value:
    
  def __init__(self, data, _children=(), _op = ""):
    self.data = data
    self.grad = 0.00
    self._prev = set(_children)
    self._op = _op
    self._backward = lambda : None

  def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data + other.data, (self, other), "+")
    def _backward():
      self.grad += 1.00 * out.grad
      other.grad += 1.00 * out.grad
    out._backward = _backward
    return out

  def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    out = Value(self.data * other.data, (self, other), "*")
    def _backward():
      self.grad += other.data * out.grad
      other.grad +=self.data * out.grad
    out._backward = _backward
    return out

  def __radd__(self, other):
    return self + other

  def __rmul__(self, other):
    return self * other

  def __repr__(self):
    return f"Value(data = {self.data}, grad = {self.grad})"

  def tanh(self):
    t = (m.exp(2 * self.data) - 1) / (m.exp(2 * self.data) + 1)
    out = Value(t, (self, ), "tanh" )
    def _backward():
      self.grad += (1 - t ** 2) * out.grad
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

    self.grad = 1.00

    for node in reversed(topo):
      node._backward()

  def __pow__(self, other):
    assert isinstance(other, (int, float))
    out = Value(self.data**other, (self,), f"**{other}")
    def _backward():
        self.grad += (other * self.data**(other - 1)) * out.grad
    out._backward = _backward
    return out

  def __neg__(self):
    return self * -1

  def __sub__(self, other):
    return self + (-other)

  def __rsub__(self, other):
    return other + (-self)

  def __truediv__(self, other):
    return self * other**-1

  def __rtruediv__(self, other):
    return other * self**-1


a = Value(-2.0); b = Value(3.0)
d = a * b; e = d + b; f = e.tanh()
f.backward()

# Defining Neuro class
class Neuron:
  def __init__(self, nin):
    self.w = [Value(r.uniform(-1,1)) for _ in range(nin)]
    self.b = Value(r.uniform(-1, 1))
  def __call__(self, x):
    act = sum((wi*xi for wi, xi in zip(self.w, x)),self.b)
    out = act.tanh()
    return out
  def parameters(self):
    return self.w + [self.b]

# Defining Layer class
class Layer:
  def __init__(self, nin, nout):
    self.neurons = [Neuron(nin) for i in range(nout)]
  def __call__(self, x):
    out = [n(x) for n in self.neurons]
    return out[0] if len(out) == 1 else out
  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]

# Defining MLP class
class MLP:
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]

## Running code for inputs x and making MLP objects
x = [2.0, 3.0, -1.0]
n = MLP(3, [4, 4, 1])
output = n(x)
output.backward()
print(output)
print(f"Total Parameters: {len(n.parameters())}")   ## total parameters (weights and biases of MLP)


## Running code for input xs and y_actual = ys
xs = [
  [2.0, 3.0, -1.0],
  [3.0, -1.0, 0.5],
  [0.5, 1.0, 1.0],
  [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

## Running same code for multiple epochs to obtain accurate weights and biases
for k in range(100):
  y_pred = [n(x) for x in xs]
  loss = sum([(yout - ygt)**2 for ygt, yout in zip(ys, y_pred)])
  for p in n.parameters():
    p.grad = 0.00
  loss.backward()
  for p in n.parameters():
    p.data -= 0.05 * p.grad
  print(f"Step {k}, Loss: {loss.data}")
print([n(x).data for x in xs])