import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath):
    data = np.loadtxt(filepath, delimiter=",")
    
    X = data[:, :-1]
    y = data[:, -1]

    y = np.where(y == 0, -1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
