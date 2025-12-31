# MicrogradFromScratch : A tiny autograd system from scratch
Project Overview
<br>
This project is a deep-dive into the fundamentals of Modern AI. Instead of using high-level libraries like PyTorch or TensorFlow, I built a Scalar-valued Autograd Engine and a Neural Network Library using pure Python.The goal was to understand exactly how "Learning" happens in a machine by implementing the backpropagation algorithm from first principles.
<br>
Core FeaturesCustom Autograd Engine: A Value class that tracks mathematical operations in a Directed Acyclic Graph (DAG).Automatic Differentiation: Built-in Chain Rule implementation to calculate gradients for any complex expression.Neural Network Module: Implementation of Neuron, Layer, and MLP classes.Optimization: A manual training loop using Stochastic Gradient Descent (SGD) to minimize Squared Error Loss.
<br>
ArchitectureThe project follows a hierarchical structure:Value Class: The atomic unit. It stores data and the _backward logic for operations like +, *, pow, and tanh.Neuron: Performs $tanh(\sum w_i x_i + b)$.Layer: A collection of independent Neurons.MLP: Multiple layers stacked together to form a deep network.
<br>
How it WorksForward Pass: Data flows through the MLP, generating a prediction.Loss Calculation: Measures the gap between prediction and target.Backpropagation: Triggers a Topological Sort of all operations and calls the chain rule to distribute the "blame" (gradient) to every weight.Update: Adjusts weights in the opposite direction of the gradient to reduce error.
<br>ResultsThe model successfully classifies a binary dataset, reducing loss from ~3.8 down to 0.003 in just 100 iterations, achieving high-precision predictions (e.g., target 1.0 $\rightarrow$ output 0.98).
