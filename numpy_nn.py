"""Neural Network from scratch using only NumPy (No PyTorch).

Architecture:
    Input (16) -> Hidden 1 (16) -> Hidden 2 (16) -> Hidden 3 (8) -> Output (2)

Activations:
    - Hidden Layers: ReLU
    - Output Layer: Softmax
Loss:
    - Categorical Cross-Entropy
Backpropagation:
    - Full analytical gradients derived from scratch.
"""

import numpy as np


class NeuralNetworkNumPy:
    """
    Fully connected neural network implemented entirely in NumPy.
    Architecture: [16 -> 16 -> 16 -> 8 -> 2]
    """

    def __init__(self, layer_dims=[16, 16, 16, 8, 2], seed=42):
        np.random.seed(seed)
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1  # 4 weight layers

        # Parameters storage
        self.weights = {}
        self.biases = {}

        # He (Kaiming) Normal Initialization for ReLU networks
        for l in range(1, len(layer_dims)):
            in_dim = layer_dims[l - 1]
            out_dim = layer_dims[l]
            # Standard deviation: sqrt(2 / in_dim)
            std = np.sqrt(2.0 / in_dim)
            self.weights[f"W{l}"] = np.random.randn(in_dim, out_dim) * std
            self.biases[f"b{l}"] = np.zeros((1, out_dim))

    # ==========================================
    # Activation Functions & Derivatives
    # ==========================================
    @staticmethod
    def relu(z):
        """ReLU activation: max(0, z)"""
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        """Gradient of ReLU: 1 if z > 0 else 0"""
        return (z > 0).astype(float)

    @staticmethod
    def softmax(z):
        """Numerically stable Softmax activation along class axis."""
        shifted_z = z - np.max(z, axis=-1, keepdims=True)
        exp_z = np.exp(shifted_z)
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)

    # ==========================================
    # Forward Pass
    # ==========================================
    def forward(self, X):
        """
        Forward propagation.
        Args:
            X: Input array of shape (batch_size, 16)
        Returns:
            A_last: Output probabilities of shape (batch_size, 2)
            cache: Dictionary of intermediate values needed for backprop
        """
        cache = {"A0": X}
        A = X

        # Hidden layers 1 to 3 with ReLU
        for l in range(1, self.num_layers):
            Z = np.dot(A, self.weights[f"W{l}"]) + self.biases[f"b{l}"]
            A = self.relu(Z)
            cache[f"Z{l}"] = Z
            cache[f"A{l}"] = A

        # Output layer with Softmax
        Z_last = (
            np.dot(A, self.weights[f"W{self.num_layers}"])
            + self.biases[f"b{self.num_layers}"]
        )
        A_last = self.softmax(Z_last)
        cache[f"Z{self.num_layers}"] = Z_last
        cache[f"A{self.num_layers}"] = A_last

        return A_last, cache

    # ==========================================
    # Loss Function
    # ==========================================
    @staticmethod
    def compute_loss(Y_pred, Y_true, eps=1e-12):
        """
        Categorical Cross-Entropy Loss:
        L = - (1 / N) * sum(Y_true * log(Y_pred))
        """
        N = Y_true.shape[0]
        # Clip to prevent log(0)
        Y_pred = np.clip(Y_pred, eps, 1.0 - eps)
        loss = -np.sum(Y_true * np.log(Y_pred)) / N
        return loss

    # ==========================================
    # Backward Pass (Backpropagation)
    # ==========================================
    def backward(self, Y_true, cache):
        """
        Computes gradients for all weights and biases via backpropagation.
        Args:
            Y_true: Ground truth one-hot targets of shape (batch_size, 2)
            cache: Saved activations and pre-activations from forward pass
        Returns:
            grads: Dictionary containing dW1..dW4 and db1..db4
        """
        grads = {}
        N = Y_true.shape[0]
        L = self.num_layers

        # 1. Output Layer: Derivative of Cross-Entropy with Softmax simplifies cleanly to:
        # dZ_L = (A_L - Y_true) / N
        A_last = cache[f"A{L}"]
        dZ = (A_last - Y_true) / N

        # Gradients for layer 4
        A_prev = cache[f"A{L-1}"]
        grads[f"dW{L}"] = np.dot(A_prev.T, dZ)
        grads[f"db{L}"] = np.sum(dZ, axis=0, keepdims=True)

        # 2. Backpropagate through hidden layers (Layer 3 -> 2 -> 1)
        for l in range(L - 1, 0, -1):
            # Gradient flowing backwards from layer (l + 1)
            dA = np.dot(dZ, self.weights[f"W{l+1}"].T)

            # Gradient through ReLU: dZ = dA * relu'(Z)
            dZ = dA * self.relu_derivative(cache[f"Z{l}"])

            # Parameter gradients for layer l
            A_prev = cache[f"A{l-1}"]
            grads[f"dW{l}"] = np.dot(A_prev.T, dZ)
            grads[f"db{l}"] = np.sum(dZ, axis=0, keepdims=True)

        return grads

    # ==========================================
    # Parameter Update (Gradient Descent)
    # ==========================================
    def update_params(self, grads, lr=0.01):
        """Gradient Descent step."""
        for l in range(1, self.num_layers + 1):
            self.weights[f"W{l}"] -= lr * grads[f"dW{l}"]
            self.biases[f"b{l}"] -= lr * grads[f"db{l}"]

    def predict(self, X):
        """Returns class predictions (0 or 1)."""
        probs, _ = self.forward(X)
        return np.argmax(probs, axis=-1)


# ==========================================
# Demonstration & Training
# ==========================================
if __name__ == "__main__":
    print("=" * 65)
    print(" NUMPY NEURAL NETWORK (NO TORCH) ")
    print(" Architecture: 16 -> 16 -> 16 -> 8 -> 2 ")
    print("=" * 65)

    # 1. Create Synthetic Dataset: 500 samples, 16 features, 2 classes
    np.random.seed(42)
    num_samples = 600
    num_features = 16
    num_classes = 2

    # Generate synthetic input data
    X = np.random.randn(num_samples, num_features)

    # Define a non-linear ground truth function to classify
    # Class 1 if sum of first 8 features squared > sum of next 8, else Class 0
    scores = np.sum(X[:, :8] ** 2, axis=1) - np.sum(X[:, 8:] ** 2, axis=1)
    labels = (scores > 0).astype(int)

    # One-hot encode targets for 2 classes
    Y = np.zeros((num_samples, num_classes))
    Y[np.arange(num_samples), labels] = 1.0

    # Train / Test split (80% train, 20% test)
    split_idx = int(0.8 * num_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]
    labels_test = labels[split_idx:]

    print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"Features: {num_features}, Output classes: {num_classes}\n")

    # 2. Instantiate Network
    model = NeuralNetworkNumPy(layer_dims=[16, 16, 16, 8, 2], seed=42)

    # 3. Training Loop
    epochs = 400
    learning_rate = 0.05
    batch_size = 32

    print("Training Progress:")
    print("-" * 65)
    for epoch in range(1, epochs + 1):
        # Shuffle batches
        indices = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[indices]
        Y_shuffled = Y_train[indices]

        # Mini-batch gradient descent
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_shuffled[i : i + batch_size]
            Y_batch = Y_shuffled[i : i + batch_size]

            # Forward pass
            Y_pred, cache = model.forward(X_batch)

            # Backward pass
            grads = model.backward(Y_batch, cache)

            # Update weights & biases
            model.update_params(grads, lr=learning_rate)

        # Log metrics every 50 epochs
        if epoch % 50 == 0 or epoch == 1:
            train_preds, _ = model.forward(X_train)
            train_loss = model.compute_loss(train_preds, Y_train)
            train_acc = np.mean(np.argmax(train_preds, axis=1) == np.argmax(Y_train, axis=1)) * 100
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")

    print("-" * 65)

    # 4. Evaluation on Unseen Test Set
    test_preds_class = model.predict(X_test)
    test_acc = np.mean(test_preds_class == labels_test) * 100
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")

    # 5. Inspect a single sample prediction
    sample_x = X_test[:1]
    probs, _ = model.forward(sample_x)
    pred_class = np.argmax(probs, axis=1)[0]
    true_class = labels_test[0]
    print(f"Sample test prediction:")
    print(f"  Input shape   : {sample_x.shape}")
    print(f"  Probabilities : Class 0: {probs[0, 0]:.4f}, Class 1: {probs[0, 1]:.4f}")
    print(f"  Predicted     : Class {pred_class} | Actual: Class {true_class}")
    print("=" * 65)
