import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from data_processing import load_and_preprocess_data
from svm import SVM  
from evaluation import plot_loss_curve, plot_accuracy_curve, plot_confusion_matrix, evaluate_stability, get_convergence_speed

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data_banknote_authentication.txt")

    # Custom SVM (GD)
    svm_gd = SVM(c=0.1)
    svm_gd.fit(X_train, y_train, X_val=X_test, y_val=y_test, epochs=1500, lr=0.01, subgradient=False)
    y_pred_gd = svm_gd.predict(X_test)
    acc_gd = accuracy_score(y_test, y_pred_gd)
    print(f"Custom SVM (GD) Accuracy: {acc_gd:.4f}")

    # Custom SVM (Sub-GD)
    svm_subgd = SVM(c=0.1)
    svm_subgd.fit(X_train, y_train, X_val=X_test, y_val=y_test, epochs=1500, lr=0.01, subgradient=True)
    y_pred_subgd = svm_subgd.predict(X_test)
    acc_subgd = accuracy_score(y_test, y_pred_subgd)
    print(f"Custom SVM (Sub-GD) Accuracy: {acc_subgd:.4f}")

    # Sklearn SVM
    clf = SVC(kernel='linear', C=0.1)
    clf.fit(X_train, y_train)
    y_pred_sklearn = clf.predict(X_test)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print(f"Sklearn SVM Accuracy: {acc_sklearn:.4f}")

    # --- Visualization and Evaluation ---
    plot_loss_curve(svm_gd.loss_history, svm_subgd.loss_history)
    plot_accuracy_curve(svm_gd.accuracy_history, svm_subgd.accuracy_history)

    plot_confusion_matrix(y_test, y_pred_gd, title="Confusion Matrix (GD)")
    plot_confusion_matrix(y_test, y_pred_subgd, title="Confusion Matrix (Sub-GD)")
    plot_confusion_matrix(y_test, y_pred_sklearn, title="Confusion Matrix (Sklearn SVM)")

    # Stability (standard deviation of loss)
    stability_gd = evaluate_stability(svm_gd.loss_history)
    stability_subgd = evaluate_stability(svm_subgd.loss_history)
    print(f"Stability of GD: {stability_gd:.6f}")
    print(f"Stability of Sub-GD: {stability_subgd:.6f}")

    # Convergence speed
    convergence_gd = get_convergence_speed(svm_gd.loss_history)
    convergence_subgd = get_convergence_speed(svm_subgd.loss_history)
    print(f"Convergence Speed (epochs) GD: {convergence_gd}")
    print(f"Convergence Speed (epochs) Sub-GD: {convergence_subgd}")
