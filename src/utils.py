import numpy as np    

def calculate_hinge_loss(x, y, w, b): #Hinge Loss for a single data point
    return max(0, 1 - y * (np.dot(w.T, x) + b))

def calculate_total_loss(X, y, w, b, c): #Average Loss over all the points
    n = X.shape[0]
    total_loss = 0
    for i in range(n):
        total_loss += calculate_hinge_loss(X[i], y[i], w, b)
    avg_loss = total_loss / n
    reg_term = (c / 2) * np.linalg.norm(w) ** 2
    return avg_loss + reg_term

def compute_gradient(X, y, w, b, c):
    n = X.shape[0]
    dw = np.zeros_like(w)
    db = 0

    for i in range(n):
        if y[i] * (np.dot(w.T, X[i]) + b) < 1:
                dw += -y[i] * X[i]
                db += -y[i]
    dw /= n
    db /= n
    dw += c * w

    return dw, db

def compute_subgradient(X, y, w, b, c):
    n = X.shape[0]
    dw = np.zeros_like(w)
    db = 0

    for i in range(n):
        if y[i] * (np.dot(w.T, X[i]) + b) < 1:
                dw += -y[i] * X[i]
                db += -y[i]
        elif y[i] * (np.dot(w.T, X[i]) + b) == 1:
                alpha = np.random.uniform(0, 1)
                dw += alpha * (-y[i] * X[i])
                db += alpha * (-y[i])
    dw /= n
    db /= n
    dw += c * w

    return dw, db