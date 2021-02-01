import numpy as np
from mlscratchnk.supervised.decision_tree import DecisionTreeRegressor
from mlscratchnk.utils.loss import SquareLoss, CrossEntropyLoss
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

    def __init__(self, n_estimators=10, learning_rate=0.01, max_depth=32, min_samples_split=2, min_samples_leaf=1, debug=False, min_impurity_decrease = 0, random_state=None):

        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.debug = debug
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = np.random.RandomState(seed=random_state)

        self.trees = []

        self.loss_obj = self.get_loss_function()

        for _ in range(0, n_estimators):
            self.trees.append(DecisionTreeRegressor(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, debug=debug, splitter='random', min_impurity_decrease=min_impurity_decrease, random_state=random_state))

    @abstractmethod
    def get_loss_function(self):
        pass

    def fit(self, X, y):
        y_pred = np.full(np.shape(y), np.mean(y))

        for i in range(self.n_estimators):
            gradient = self.loss_obj.gradient(y = y, y_pred = y_pred)
            self.trees[i].fit(X, gradient)
            update = self.trees[i].predict(X)
            # Update y prediction
            y_pred -= np.multiply(self.learning_rate, update)


    def predict(self, X):
        y_pred = np.array([])
        
        for tree in self.trees:
            update = tree.predict(X)
            update = np.multiply(self.learning_rate, update)
            y_pred = -update if not y_pred.any() else y_pred - update

        return y_pred


class GradientBoostingClassifier(GradientBoosting):

    def get_loss_function(self):
        return CrossEntropyLoss()
    
    
    def to_categorical(self, x, n_col=None):
        """ One-hot encoding of nominal values """
        if not n_col:
            n_col = np.amax(x) + 1
        one_hot = np.zeros((x.shape[0], n_col))
        one_hot[np.arange(x.shape[0]), x] = 1
        return one_hot


    def fit(self, X, y):
        super().fit(X, self.to_categorical(y))

    def predict(self, X):
        y_pred = super().predict(X)
        
        # Turn into probability distribution
        y_pred = np.exp(y_pred) / np.expand_dims(np.sum(np.exp(y_pred), axis=1), axis=1)
        # Set label to the value that maximizes probability
        y_pred = np.argmax(y_pred, axis=1)

        return y_pred



class GradientBoostingRegressor(GradientBoosting):

    def get_loss_function(self):
        return SquareLoss()