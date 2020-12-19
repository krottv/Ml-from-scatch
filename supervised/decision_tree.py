import numpy as np
from abc import ABC, abstractmethod

class TreeNode():

    left = None
    right = None

    def __init__(self, feature_index = None, feature_threshold = None, impurity = 0,
     depth = 0, nsamples=0, hindex=0, value=None, clazz_percentage=0, impurity_decrease = 0):

        self.feature_index = feature_index
        self.feature_threshold = feature_threshold
        self.impurity = impurity
        self.depth = depth
        self.nsamples = nsamples
        self.hindex = hindex
        self.value = value
        self.clazz_percentage = clazz_percentage
        self.impurity_decrease = impurity_decrease

    def is_leaf(self):
        return self.left is None and self.right is None

    def graph_id(self):
        return f'{self.depth}d{self.hindex}h'

    def displayName(self, label_classes = None, label_features = None):
        displayClass = self.value if label_classes is None else label_classes[self.value]
        displayFeature = self.feature_index if label_features is None or self.feature_index is None else label_features[self.feature_index]

        if self.depth == 0:
            return 'root node {:} - {:.2f}, impurity {:.2f}, nsamples {:}'.format(displayFeature, self.feature_threshold, self.impurity, self.nsamples)
        elif self.is_leaf():
            return 'leaf, impurity {:.2f}\ndepth {:}, hindex {:}, nsamples {:}\nclass {:} {:0.2%}'.format(self.impurity, self.depth, self.hindex, self.nsamples, displayClass, self.clazz_percentage)
        else:
            return '{:} - {:.2f}, impurity {:.2f}\ndepth {:}, hindex {:}, nsamples {:}\nclass {:} {:0.2%} impr decr {:0.2%}'.format(displayFeature, self.feature_threshold, self.impurity, self.depth, self.hindex, self.nsamples, displayClass, self.clazz_percentage, self.impurity_decrease)


    def __str__(self):
        return self.displayName()



class DecisionTree(ABC):

    """
    Abstract class for DecisionTreeClassifier and DecisionTreeRegressor

    leaf (also called external node) and an internal node. 
    An internal node will have further splits (also called children), 
    while a leaf is by definition a node without any children (without any further splits).

    Parameters:
    -----------

    min_samples_split
        will evaluate the number of
        samples in the node, and if the number is less than the minimum 
        the split will be avoided and the node will be a leaf.

    min_samples_leaf 
        checks before the node is generated, that is,
        if the possible split results in a child with fewer samples, the split will be avoided 
        (since the minimum number of samples for the child to be a leaf has not been reached) and the node will be replaced by a leaf.

    splitter
        RandomSplitter initiates a **random split on each chosen feature**, 
        whereas BestSplitter goes through **all possible splits on each chosen feature**.


    min_impurity_decrease
        A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
        The weighted impurity decrease equation is the following:

        N_t / N * (impurity - N_t_R / N_t * right_impurity
                    - N_t_L / N_t * left_impurity)
        where N is the total number of samples, N_t is the number of samples at the current node, N_t_L is the number of samples in the left child, and N_t_R is the number of samples in the right child.

        N, N_t, N_t_R and N_t_L all refer to the weighted sum, if sample_weight is passed.
    
    """

    def __init__(self, max_depth=32, min_samples_split=2, min_samples_leaf=1, criterion='gini', debug=False, 
    splitter='best', min_impurity_decrease = 0, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.debug = debug
        self.splitter = splitter
        self.min_impurity_decrease = min_impurity_decrease

        self.random_state = random_state if isinstance(random_state, np.random.RandomState) else np.random.RandomState(seed=0)

    def print(self, message):
        if self.debug:
            print(message)


    def calc_impurity_decrease(self, total_samples, left_impurity, right_impurity, left_nsamples, right_nsamples):
 
        decrease = total_samples / self.total_samples * ((left_impurity + right_impurity) - right_nsamples / total_samples * right_impurity - left_nsamples / total_samples * left_impurity)

        return decrease

    def get_depth(self):
        return self.current_depth

    @abstractmethod
    def calculate_value_and_its_percentage_for_node(self, y):
        pass

    @abstractmethod
    def calculate_impurity(self, y, y_left, y_right):
        pass

    def fit(self, X, y):
        self.total_samples =len(y)
        self.root_node = TreeNode(nsamples=self.total_samples)
        self.current_depth = 0
        self.split_inner_node(self.root_node, X, y)


    def split_inner_node(self, node, features, targets):

        if node.clazz_percentage == 1:
            self.print(f'stop because we dont have more classes to separate {node}')
            return False

        if len(targets) < self.min_samples_split:
            self.print(f'stop because of min_samples_split {node}')
            return False

        if node.depth >= self.max_depth:
            self.print(f'stop because of max_depth {node}')
            return False


        min_impurity_left, min_impurity_right, found_threshold, found_feature_index, features_left, features_right, targets_left, targets_right = self.find_impurity(features, targets)

        impurity_gain = node.impurity - min_impurity_left - min_impurity_right

        self.print(f'found best impurity gain = {impurity_gain}, found_threshold = {found_threshold}, feature {found_feature_index}, new depth = {node.depth + 1}')


        if targets_left is None or targets_right is None or self.min_samples_leaf > len(targets_left) or self.min_samples_leaf > len(targets_right):
            self.print(f'stop because of min_samples_leaf {node}')
            return False

        new_depth = node.depth + 1

        if self.current_depth < new_depth:
            self.current_depth = new_depth

        impurity_decrease = self.calc_impurity_decrease(len(targets), min_impurity_left, min_impurity_right, len(targets_left), len(targets_right))
        
        if self.min_impurity_decrease != 0 and impurity_decrease < self.min_impurity_decrease:
            self.print('stop because min_impurity_decrease {:.3%} > impurity_decrease {:.3%}'.format(self.min_impurity_decrease, impurity_decrease))
            return False
        
        node.impurity_decrease = impurity_decrease
        if node.feature_index is None:
            node.feature_index = found_feature_index
            node.feature_threshold = found_threshold
        
        left_clazz, left_clazz_percentage = self.calculate_value_and_its_percentage_for_node(targets_left)

        node.left = TreeNode(impurity = min_impurity_left, depth = new_depth, nsamples=len(targets_left), hindex=node.hindex * 2, value=left_clazz, clazz_percentage=left_clazz_percentage)
        self.split_inner_node(node.left, features_left, targets_left)

        right_clazz, right_clazz_percentage = self.calculate_value_and_its_percentage_for_node(targets_right)
        node.right = TreeNode(impurity = min_impurity_right, depth = new_depth, nsamples=len(targets_right), hindex=node.hindex*2 + 1, value=right_clazz, clazz_percentage=right_clazz_percentage)
        self.split_inner_node(node.right, features_right, targets_right)
        
        return True


    def separate_and_get_impurity(self, threshold, features, feature_index, targets):

        condition = features[:, feature_index] >= threshold
        indexes_more = np.where(condition)
        indexes_less = np.where(~condition)

        features_more = features[indexes_more]
        features_less = features[indexes_less]
        y_right = targets[indexes_more]
        y_left = targets[indexes_less]

        left_impurity, right_impurity = self.calculate_impurity(targets, y_left, y_right)

        return features_more, features_less, y_right, y_left, left_impurity, right_impurity


    def find_best_impurity(self, features, targets):

        min_impurity_left, min_impurity_right = np.inf, np.inf
        features_left, features_right, targets_right, targets_left, found_feature = None, None, None, None, None
        found_threshold = 0

        
        for feature in features:
            unique =  np.sort(np.unique(features[:, feature]))
            
            for index in range(1, len(unique)):

                threshold = unique[index]
                features_more, features_less, y_right, y_left, left_impurity, right_impurity = self.separate_and_get_impurity(threshold, features, feature, targets)
                
                if left_impurity + right_impurity < (min_impurity_left + min_impurity_right):

                    min_impurity_left, min_impurity_right = left_impurity, right_impurity
                    features_left, features_right = features_less, features_more
                    targets_left, targets_right = y_left, y_right
                    found_threshold = threshold
                    found_feature = feature

        return min_impurity_left, min_impurity_right, found_threshold, found_feature, features_left, features_right, targets_left, targets_right



    def find_random_impurity(self, features, targets):

        feature_numbers = features.shape[1]
        random_numbers = self.random_state.rand(feature_numbers)

        min_impurity_left, min_impurity_right = np.inf, np.inf
        features_left, features_right, targets_right, targets_left, found_feature = None, None, None, None, None
        found_threshold = 0

        for feature_index in range(0, feature_numbers):
            
            unique =  np.sort(np.unique(features[:, feature_index]))
            
            threshold = 0

            if len(unique) == 1:
                threshold = unique[0]
            else:
                mx = unique[-2]
                mn = unique[1]
                threshold = (mx - mn) * random_numbers[feature_index] + mn

            features_more, features_less, y_right, y_left, left_impurity, right_impurity = self.separate_and_get_impurity(threshold, features, feature_index, targets)
            
            self.print(f'threshold = {threshold} len more {features_more.shape}, len less {features_less.shape}')

            if left_impurity + right_impurity < (min_impurity_left + min_impurity_right):

                min_impurity_left, min_impurity_right = left_impurity, right_impurity
                features_left, features_right = features_less, features_more
                targets_left, targets_right = y_left, y_right
                found_threshold = threshold
                found_feature = feature_index

        
        return min_impurity_left, min_impurity_right, found_threshold, found_feature, features_left, features_right, targets_left, targets_right
            

    def find_impurity(self, features, targets):

        return self.find_random_impurity(features, targets) if self.splitter == 'random' else self.find_best_impurity(features, targets)
            

    def predict(self, features):
        if self.root_node.left is None:
            raise Exception('we dont have any nodes')

        targets = np.empty(features.shape[0])

        # we predict features one by one because view of a view of a view can lose references to original array numpy

        for i in range(len(targets)):
            targets[i] = self.predict_node(self.root_node, features[i, :])

        
        return targets.astype('int')



    def predict_node(self, node: TreeNode, features):

        if node.is_leaf():
            return node.value
            
        else:
            feature_value = features[node.feature_index]
            if feature_value >= node.feature_threshold:
                return self.predict_node(node.right, features)
            else:
                return self.predict_node(node.left, features)


class DecisionTreeRegressor(DecisionTree):

    def calculate_value_and_its_percentage_for_node(self, y):
        return (y[0] if len(y) == 1 else np.mean(y), 0)

    def squared_residual_sum(self, y):
        return np.sum((y - np.mean(y)) ** 2)

    def calculate_impurity(self, y, y_left, y_right):

        #return (np.var(y_left), np.var(y_right))

        return (self.squared_residual_sum(y_left), self.squared_residual_sum(y_right))

    def calculate_variance(self, X):
        """ Return the variance of the features in dataset X """
        mean = np.ones(np.shape(X)) * X.mean(0)
        n_samples = np.shape(X)[0]
        variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))
        
        return variance
        
class DecisionTreeClassifier(DecisionTree):

    def fit(self, X, y):
        self.classes = np.unique(y)

        super().fit(X, y)

    def calculate_value_and_its_percentage_for_node(self, y):

        left_value_counts = np.unique(y, return_counts=True)
        left_clazz_index = left_value_counts[1].argmax()
        left_clazz_percentage = left_value_counts[1][left_clazz_index] / np.sum(left_value_counts[1])
        return (left_value_counts[0][left_clazz_index], left_clazz_percentage)


    def calculate_impurity(self, y, y_left, y_right):
        left_impurity = 0 
        right_impurity = 0

        for c in self.classes:
            left_impurity += self.calc_impurity_for_class(y_left[y_left == c].shape[0], y_left.shape[0])
            right_impurity += self.calc_impurity_for_class(y_right[y_right == c].shape[0], y_right.shape[0])

        total_samples = len(y)
        left_impurity = (left_impurity * len(y_left)) / total_samples
        right_impurity = (right_impurity * len(y_right)) / total_samples

        return (left_impurity, right_impurity)


    def calc_entropy_for_class(self, class_counts, total_samples):
        if total_samples == 0:
            return 0

        p = class_counts / total_samples

        return -p * np.log2(p)

    def calc_entropy(self, classes_counts, total_samples):
        return sum([self.calc_entropy_for_class(x, total_samples) for x in classes_counts])

    def calc_gini(self, samples, total_samples):

        gini = 0
        for sample in samples:
            gini += self.calc_gini_for_class(sample, total_samples)

        return gini

    def calc_gini_for_class(self, sample, total_samples):
        if total_samples == 0:
            return 0
        p = sample / total_samples

        return p * (1 - p)

    def calc_impurity_for_class(self, sample, total_samples):
        return self.calc_entropy_for_class(sample, total_samples) if self.criterion == 'entropy' else self.calc_gini_for_class(sample, total_samples)
