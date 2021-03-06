
import numpy as np
class MatrixLinearRegressor:

    def fit(self, features, target):
        inversed = np.linalg.inv(features.T @ features)
        self.weight = inversed @ features.T @ target

    def predict(self, features):
        return features @ self.weight
    