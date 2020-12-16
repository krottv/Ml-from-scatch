import numpy as np
import pandas as pd
from decision_tree_np import DecisionTreeClassifier


class RandomForestClassifier: 
    """
    Uses ensemble of decision trees on subset of random data using random subset of features
    
    Parameters:
    ----------

    n_estimators
        number of decision trees inside

    max_features
        maximum number of features that decision tree is allowed to use. If none, use sqrt(features)

    All other parameters are the same as in decision_tree

    """

    def __init__(self, n_estimators=10, max_features=None, max_depth=32, min_samples_split=2, min_samples_leaf=1, criterion='gini', debug=False, splitter='best', min_impurity_decrease = 0, random_state=None):

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

        self.trees = []

        for _ in range(0, n_estimators):
            self.trees.append(DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
             criterion=criterion, debug=debug, splitter=splitter, min_impurity_decrease=min_impurity_decrease, random_state=random_state))


    def fit(self, features, target):

        if isinstance(features, pd.DataFrame):
            features = features.values
        if isinstance(target, pd.Series):
            target = target.values

        n_features = np.shape(features)[1]
        max_features = self.max_features
        if max_features is None:
            max_features = np.sqrt(n_features).astype('int')

        subsets = self.get_random_subsets(features, target, self.n_estimators)

        for i in range(self.n_estimators):
            X_subset, y_subset = subsets[i]

            # Feature bagging (select random subsets of the features)
            ids = self.random_state.choice(range(n_features), max_features, replace=True)
        
            # Save the indices of the features for prediction
            tree = self.trees[i]
            tree.feature_ids = ids

            # Choose the features corresponding to the indices
            features_to_fit = X_subset[:, ids]

            # Fit the tree to the data
            tree.fit(features_to_fit, y_subset)

        
    def predict(self, features):
        if isinstance(features, pd.DataFrame):
            features = features.values
        
        predictions = np.full((features.shape[0], self.n_estimators), 0)
        # Let each tree make a prediction on the data
        
        for i in range(self.n_estimators):
            
            tree = self.trees[i]
            # Indices of the features that the tree has trained on
            feature_ids = tree.feature_ids
            # Make a prediction based on those features

            predicted = tree.predict(features[:, feature_ids])

            predictions[:, i] = predicted
            
        result = np.empty(features.shape[0])

        # For each sample
        for i in range(predictions.shape[0]):

            # Select the most common class prediction
            most_frequent_class = np.bincount(predictions[i, :]).argmax()
            result[i] = most_frequent_class

        return result

    def get_depth(self):
        return max([x.get_depth() for x in self.trees])

    def get_random_subsets(self, features, target, n_subsets):
        """
        1. concatenate features and target
        2. choice
        """
        select_indexes = range(len(target))

        result = []
        for _ in range(n_subsets):
            ids = self.random_state.choice(select_indexes, size=len(target), replace=True)
            x = features[ids]
            y = target[ids]
            result.append([x, y])

        return result

            
