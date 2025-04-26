import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def plot_loss_curve(loss_history_gd, loss_history_subgd):
    plt.figure(figsize=(8,5))
    plt.plot(loss_history_gd, label="Gradient Descent (GD)")
    plt.plot(loss_history_subgd, label="Subgradient Descent (Sub-GD)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve Comparison")
    plt.legend()
    plt.grid()
    plt.show()

def plot_accuracy_curve(acc_history_gd, acc_history_subgd):
    plt.figure(figsize=(8,5))
    plt.plot(acc_history_gd, label="Gradient Descent (GD)")
    plt.plot(acc_history_subgd, label="Subgradient Descent (Sub-GD)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve Comparison")
    plt.legend()
    plt.grid()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=[-1,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[-1,1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

def evaluate_stability(loss_history):
    loss_history = np.array(loss_history)
    return np.std(loss_history)

def get_convergence_speed(loss_history, threshold=1e-3):
    initial_loss = loss_history[0]
    for epoch, loss in enumerate(loss_history):
        if abs(loss - loss_history[-1]) <= threshold * abs(initial_loss):
            return epoch
    return len(loss_history)
