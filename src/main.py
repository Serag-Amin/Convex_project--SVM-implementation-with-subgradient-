import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from data_processing import load_and_preprocess_data
from svm import SVM  

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data("data_banknote_authentication.txt")
    #Custom SVM
    my_svm = SVM(c=0.1)
    my_svm.fit(X_train, y_train, epochs=1500, lr=0.01, subgradient=True)
    y_pred_custom = my_svm.predict(X_test)
    acc_custom = accuracy_score(y_test, y_pred_custom)
    print(f"Custom SVM Accuracy: {acc_custom:.4f}")

    # Sklearn SVM
    clf = SVC(kernel='linear', C=0.1)
    clf.fit(X_train, y_train)
    y_pred_sklearn = clf.predict(X_test)
    acc_sklearn = accuracy_score(y_test, y_pred_sklearn)
    print(f"Sklearn SVM Accuracy: {acc_sklearn:.4f}")
