import numpy as np


class SquareLoss:

    # it devides on 2 instead of len(y) in original implementation. Why?
    def loss(self, y, y_pred):
        return np.power(y - y_pred, 2) / len(y)

    def gradient(self, y, y_pred):
        return -(y - y_pred)


class CrossEntropyLoss:

    def loss(self, y, y_pred):
        #avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)

        return y * np.log(p) - (1 - y) * np.log(1 - p)

    #find derivative by y_pred
    def gradient(self, y, y_pred):
        #avoid division by zero
        p = np.clip(y_pred, 1e-15, 1 - 1e-15)

        return y / p + (y - 1) / (1 - p)