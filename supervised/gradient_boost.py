import numpy as np
from numpy.lib import utils
from decision_tree import DecisionTree
from utils import loss
from abc import ABC, abstractmethod

class GradientBoosting(ABC):
    """
    Uses ensemble of decision trees which train on predicting the gradient of a loss function
    
    Parameters:
    ----------

    n_estimators
        number of decision trees inside

    learning_rate
        gradient of the loss function is multiplied by learning_rate

    All other parameters are the same as in decision_tree except criterion (always gini). splitter (always random)

    """

    def __init__(self, regression, n_estimators=10, learning_rate=0.01, max_depth=32, min_samples_split=2, min_samples_leaf=1, debug=False, min_impurity_decrease = 0, random_state=None):

        self.regression = regression
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.debug = debug
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = np.random.RandomState(seed=random_state)

        self.trees = []

        self.loss_func = loss.SquareLoss if regression else loss.CrossEntropyLoss

        for _ in range(0, n_estimators):
            self.trees.append(DecisionTree(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
             criterion='gini', debug=debug, splitter='random', min_impurity_decrease=min_impurity_decrease, random_state=random_state))

    def fit(self, X, y):
        pass


    def predict(self, X):
        pass