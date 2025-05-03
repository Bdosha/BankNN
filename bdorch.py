import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def data_split(data: pd.DataFrame, train_ratio=0.8) -> (pd.DataFrame, pd.DataFrame):
    train = data.sample(frac=train_ratio)
    test = data.drop(train.index)
    return train, test


def get_xy(data: pd.DataFrame):
    return data[[i for i in data.keys() if i != 'Дефолт']], data['Дефолт']


def scale_data(data):
    return (data - data.mean()) / data.std()


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def cross_entropy(y, y_pred) -> np.ndarray:
    y_pred = np.clip(y_pred, 1e-5, 1 - 1e-5)
    return - 1 / y.size * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))


def w_grad(x, y, y_pred):
    x = np.array(x, dtype=np.float64)
    y = np.array(y.values, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    grad = np.array([(1 / len(y)) * np.sum((y_pred - y) * x[:, j]) for j in range(30)]).T

    return grad


def b_grad(y, y_pred):
    y = np.array(y.values, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    grad = (1 / len(y)) * np.sum((y_pred - y))

    return grad


class Perceptron:
    def __init__(self):
        self.w = np.random.randn(30, 1)
        self.b = np.zeros((1,))

    def __call__(self, x) -> np.ndarray:
        X = np.array(x, dtype=np.float64)
        z = X @ self.w + self.b
        return sigmoid(z).squeeze()


def get_acc(y_test, y_pred):
    predicted = (y_pred > 0.5).astype(np.float32)
    correct = np.sum(predicted == y_test)
    total = y_test.size
    return 100 * correct / total

def cor(df1, df2, default):
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(
        df1,
        df2,
        c=default,
        cmap='coolwarm',
        alpha=0.7
    )
    plt.grid(True)
    plt.show()