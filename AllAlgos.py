import numpy as np


class MultiLinearRegression:
    def __init__(self, X_train, Y_train, lr, iter):
        self.X_train = X_train
        self.Y_train = Y_train
        self.lr = lr
        self.iter = iter
        self.n = X_train.shape[1]
        self.m = X_train.shape[0]
        self.weights = np.zeros(self.n)
        self.bias = 0.0

    def fit(self):
        # get the dw and db
        for num in range(self.iter):
            pred = self.get_train_pred()
            total_err_b = 0
            total_err_w = np.zeros(self.n)
            for i in range(self.m):
                err = pred[i] - self.Y_train[i]
                total_err_b += err
                total_err_w += err * self.X_train[i]

            db = total_err_b / self.m
            dw = total_err_w / self.m

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            if num % 1000 == 0:
                print(self.get_cost())
        return (self.weights, self.bias)

    def get_train_pred(self):
        temp = self.weights.reshape(1, -1)
        return (np.dot(self.X_train, temp.T).T + self.bias).reshape(-1)

    def get_cost(self):
        err = np.sum(self.get_train_pred() - self.Y_train) ** 2
        err = err / (2 * self.m)
        return err

    def predict(self, input):
        return np.dot(input, self.weights) + self.bias