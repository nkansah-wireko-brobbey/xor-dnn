# This is a sample Python script.
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class NeuralNetwork:
    def __init__(self):
        self.W1 = np.random.randn(2, 4)*0.1
        self.W2 = np.random.randn(4,1)*0.1

        self.b1 = np.zeros((1, 4))
        self.b2 = np.zeros((1, 1))

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)

        return self.a2

    def backward(self, X, y, learning_rate=0.5):
        m = X.shape[0]
        output_error = self.a2 - y

        d_z2 = output_error * sigmoid_derivative(self.z2)
        d_W2 = self.a1.T @ d_z2 / m
        d_b2 = np.sum(d_z2, axis=0, keepdims=True) / m

        hidden_error = d_z2 @ self.W2.T
        d_z1 = hidden_error * sigmoid_derivative(self.z1)

        d_W1 = X.T @ d_z1 / m
        d_b1 = np.sum(d_z1, axis=0, keepdims=True) / m

        self.W2 -= learning_rate * d_W2
        self.b2 -= learning_rate * d_b2
        self.W1 -= learning_rate * d_W1
        self.b1 -= learning_rate * d_b1

    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2)

def main():
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])

    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    np.random.seed(0)
    nn = NeuralNetwork()

    print("Training network...")

    for epoch in range(500000):

        prediction = nn.forward(X)

        loss = nn.compute_loss(prediction, y)

        nn.backward(X, y, learning_rate=0.1)

        if epoch % 1000 == 0:
            print("Epoch {}: Loss: {}".format(epoch, loss))

    print("Finished training")
    print(f"{'Input':<15} {'Predicted':<12} {'Rounded':<10} {'True'}")
    print("-"*45)

    final_preds = nn.forward(X)
    for i in range(len(X)):
        inputs = str(X[i].tolist())
        pred = float(final_preds[i][0])
        true = int(y[i][0])
        print(f"{inputs:<15} {pred:<12.4f} {round(pred):<10} {true}")

    print("\n✓ Network successfully learned XOR!" if all(
        round(float(final_preds[i][0])) == int(y[i][0]) for i in range(len(X))
    ) else "✗ Training needs more epochs or tuning.")

if __name__ == '__main__':
    main()

