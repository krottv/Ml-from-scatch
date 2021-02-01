import numpy as np


class SquareLoss:

    # original square function has division on len(y). Why here it is 2? Seems less 
    def loss(self, y, y_pred):
        return np.power(y - y_pred, 2) / 2

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