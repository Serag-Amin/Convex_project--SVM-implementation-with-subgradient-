import numpy as np
from sklearn.metrics import accuracy_score
from utils import  calculate_total_loss, compute_gradient, compute_subgradient

class SVM:
    def __init__(self, c=1, kernel = None): # Add kernels ?
        self.c = c 
        self.w = None
        self.b = 0
        self.loss_history = []
        self.accuracy_history = []
    
    def initialize_weights(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0
        
    def fit(self, X, y, X_val=None, y_val=None, epochs=1500, lr=0.01, subgradient=False):
        if self.w is None:
            self.initialize_weights(X.shape[1])

        for epoch in range(epochs):
            loss = calculate_total_loss(X, y, self.w, self.b, self.c)
            self.loss_history.append(loss)

            if X_val is not None and y_val is not None:
                y_pred = self.predict(X_val)
                acc = accuracy_score(y_val, y_pred)
                self.accuracy_history.append(acc)

            print(f'Epoch {epoch+1}, loss: {loss}')
            self.step(X, y, lr, subgradient=subgradient)
    
    def step(self, X, y, lr, subgradient):
        if subgradient:
            dw, db = compute_subgradient(X, y, self.w, self.b, self.c)
        else:
            dw, db = compute_gradient(X, y, self.w, self.b, self.c)
        self.w = self.w - lr * dw 
        self.b = self.b - lr * db

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)