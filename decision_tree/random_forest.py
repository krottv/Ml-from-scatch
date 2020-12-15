import numpy as np
import pandas as pd


class RandomForest: 
    """
    Uses ensemble of decision trees on subset of random data using random subset of features
    
    Parameters:
    ----------

    n_estimators
        number of decision trees inside

    max_features
        maximum number of features that decision tree is allowed to use

    All other parameters are the same as in decision_tree

    """

    def __init__(self, n_estimators=10, max_features=None, max_depth=32, min_samples_split=2, min_samples_leaf=1, criterion='gini', debug=False, 
    splitter='best', min_impurity_decrease = 0, random_state=None):

        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.debug = debug
        self.splitter = splitter
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = np.random.RandomState(seed=random_state)
