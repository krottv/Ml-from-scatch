
import numpy as np

class CovarLinearRegressor:

    def variance(self, array):
        mean = np.mean(array)
        return sum([(mean - x) ** 2 for x in array])

    def covariance(self, arr1, arr2):
        arrs = [arr1, arr2]
        """
        The covariance of two groups of numbers describes how those numbers change together.

        Covariance is a generalization of correlation. Correlation describes the relationship between two groups of numbers, whereas covariance can describe the relationship between two or more groups of numbers.
        	
        covariance = sum((x(i) - mean(x)) * (y(i) - mean(y)))
        """

        covars = []

        min_len = min([len(x) for x in arrs])
        means = [np.mean(x) for x in arrs]

        for i in range(0, min_len):
            covars.append(sum([means[j] - arrs[j][i] for j in range(0, len(arrs))]))
        
        return sum(covars)

    # Calculate coefficients
    def coefficients(self, features, target):

        shape = features.shape

        #covariance / variance
        b1 = [self.covariance(features[:, i], target) / self.variance(features[:,i]) for i in range(0, shape[1])]

        b0 = np.mean(target) - sum(b1) * sum([np.mean(features[:, i]) for i in range(0, shape[1])])
        
        return b0, b1


    def fit(self, features, target):
        self.b0, self.b1 = self.coefficients(features, target)
        print(f'features.shape {features.shape}, fit finished {self.b0}, {self.b1}')


    def predict(self, features):
        
        predictions = []

        shape = features.shape
        for i in range(0, shape[0]):
            row = features[i, :]
            yhat = self.b0 + sum(self.b1 * row)
            predictions.append(yhat)


        return predictions