# This is a sample Python script.
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class NeuralNetwork:
    def __init__(self):
        self.W1 = np.random.randn(2, 3)*0.1
        self.W2 = np.random.randn(4,1)*0.1

        



def print_hi(name):
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


if __name__ == '__main__':
    print_hi('PyCharm')

