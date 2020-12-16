import numpy as np
import pandas as pd
from sklearn.utils import validation

class TreeNode():

    left = None
    right = None

    def __init__(self, feature_index = None, feature_threshold = None, impurity = 0,
     depth = 0, nsamples=0, hindex=0, clazz=None, clazz_percentage=0, impurity_decrease = 0):

        self.feature_index = feature_index
        self.feature_threshold = feature_threshold
        self.impurity = impurity
        self.depth = depth
        self.nsamples = nsamples
        self.hindex = hindex
        self.clazz = clazz
        self.clazz_percentage = clazz_percentage
        self.impurity_decrease = impurity_decrease

    def is_leaf(self):
        return self.left is None and self.right is None

    def graph_id(self):
        return f'{self.depth}d{self.hindex}h'

    def displayName(self, label_classes = None, label_features = None):
        displayClass = self.clazz if label_classes is None else label_classes[self.clazz]
        displayFeature = self.feature_index if label_features is None or self.feature_index is None else label_features[self.feature_index]

        if self.depth == 0:
            return 'root node {:} - {:.2f}, impurity {:.2f}, nsamples {:}'.format(displayFeature, self.feature_threshold, self.impurity, self.nsamples)
        elif self.is_leaf():
            return 'leaf, impurity {:.2f}\ndepth {:}, hindex {:}, nsamples {:}\nclass {:} {:0.2%}'.format(self.impurity, self.depth, self.hindex, self.nsamples, displayClass, self.clazz_percentage)
        else:
            return '{:} - {:.2f}, impurity {:.2f}\ndepth {:}, hindex {:}, nsamples {:}\nclass {:} {:0.2%} impr decr {:0.2%}'.format(displayFeature, self.feature_threshold, self.impurity, self.depth, self.hindex, self.nsamples, displayClass, self.clazz_percentage, self.impurity_decrease)


    def __str__(self):
        return self.displayName()



class DecisionTreeClassifier():

    """
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
        self.random_state = np.random.RandomState(seed=random_state)

    def print(self, message):
        if self.debug:
            print(message)

    def calc_entropy_for_class(self, sample, total_samples):
        if total_samples == 0:
            return 0

        p = sample / total_samples

        return -p * np.log2(p)

    def calc_entropy(self, samples, total_samples):
        return sum([self.calc_entropy_for_class(x, total_samples) for x in samples])

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


    def calc_impurity_decrease(self, total_samples, left_impurity, right_impurity, left_nsamples, right_nsamples):
 
        decrease = total_samples / self.total_samples * ((left_impurity + right_impurity) - right_nsamples / total_samples * right_impurity - left_nsamples / total_samples * left_impurity)

        return decrease

    def get_depth(self):
        return self.current_depth

    def fit(self, features, target):

        if isinstance(features, pd.DataFrame):
            features = features.values
        if isinstance(target, pd.Series):
            target = target.values

        value_counts = np.unique(target, return_counts=True)

        #sort_value_counts = np.argsort(-value_counts[1])
        #value_counts = (value_counts[0][sort_value_counts], value_counts[1][sort_value_counts])

        self.classes = value_counts[0]

        root_samples = value_counts[1]
        self.total_samples = len(target)

        root_impurity = self.calc_entropy(root_samples, self.total_samples) if self.criterion == 'entropy' else self.calc_gini(root_samples, self.total_samples)

        self.print(f'root_samles {root_samples}, root impurity {root_impurity}, root_samples {root_samples}, classes {self.classes}')

        self.root_node = TreeNode(impurity = root_impurity, nsamples=self.total_samples)
        self.current_depth = 0
        self.split_inner_node(self.root_node, features, target)


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

            left_counts = np.unique(targets_left, return_counts=True) if targets_left is not None else None
            right_counts = np.unique(targets_right, return_counts=True) if targets_right is not None else None

            self.print(f'stop because of min_samples_leaf {node} left:\n{left_counts}\nright:\n{right_counts}')
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
        
        left_clazz, left_clazz_percentage = self.get_clazz_and_its_percentage(targets_left)
        node.left = TreeNode(impurity = min_impurity_left, depth = new_depth, nsamples=len(targets_left), hindex=node.hindex * 2, clazz=left_clazz, clazz_percentage=left_clazz_percentage)
        self.split_inner_node(node.left, features_left, targets_left)

        right_clazz, right_clazz_percentage = self.get_clazz_and_its_percentage(targets_right)
        node.right = TreeNode(impurity = min_impurity_right, depth = new_depth, nsamples=len(targets_right), hindex=node.hindex*2 + 1, clazz=right_clazz, clazz_percentage=right_clazz_percentage)
        self.split_inner_node(node.right, features_right, targets_right)
        
        return True

    def get_clazz_and_its_percentage(self, targets_left):
        left_value_counts = np.unique(targets_left, return_counts=True)
        left_clazz_index = left_value_counts[1].argmax()
        left_clazz_percentage = left_value_counts[1][left_clazz_index] / np.sum(left_value_counts[1])
        return (left_value_counts[0][left_clazz_index], left_clazz_percentage)


    def separate_and_get_impurity(self, threshold, features, feature_index, targets):

        condition = features[:, feature_index] >= threshold
        indexes_more = np.where(condition)
        indexes_less = np.where(~condition)

        features_more = features[indexes_more]
        features_less = features[indexes_less]
        targets_more = targets[indexes_more]
        targets_less = targets[indexes_less]

        left_impurity = 0
        right_impurity = 0
        for c in self.classes:
            left_impurity += self.calc_impurity_for_class(targets_less[targets_less == c].shape[0], targets_less.shape[0])
            right_impurity += self.calc_impurity_for_class(targets_more[targets_more == c].shape[0], targets_more.shape[0])

        total_samples = len(targets_more) + len(targets_less)
        left_impurity = (left_impurity * len(targets_less)) / total_samples
        right_impurity = (right_impurity * len(targets_more)) / total_samples

        return features_more, features_less, targets_more, targets_less, left_impurity, right_impurity


    def find_best_impurity(self, features, targets):

        min_impurity_left, min_impurity_right = np.inf, np.inf
        features_left, features_right, targets_right, targets_left, found_feature = None, None, None, None, None
        found_threshold = 0

        
        for feature in features:
            unique =  np.sort(np.unique(features[:, feature]))
            
            for index in range(1, len(unique)):

                threshold = unique[index]
                features_more, features_less, targets_more, targets_less, left_impurity, right_impurity = self.separate_and_get_impurity(threshold, features, feature, targets)
                
                if left_impurity + right_impurity < (min_impurity_left + min_impurity_right):

                    min_impurity_left, min_impurity_right = left_impurity, right_impurity
                    features_left, features_right = features_less, features_more
                    targets_left, targets_right = targets_less, targets_more
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
            
            mx = unique[-2]
            mn = unique[1]

            threshold = (mx - mn) * random_numbers[feature_index] + mn


            features_more, features_less, targets_more, targets_less, left_impurity, right_impurity = self.separate_and_get_impurity(threshold, features, feature_index, targets)
            
            self.print(f'threshold = {threshold} len more {features_more.shape}, len less {features_less.shape}')

            if left_impurity + right_impurity < (min_impurity_left + min_impurity_right):

                min_impurity_left, min_impurity_right = left_impurity, right_impurity
                features_left, features_right = features_less, features_more
                targets_left, targets_right = targets_less, targets_more
                found_threshold = threshold
                found_feature = feature_index

        
        return min_impurity_left, min_impurity_right, found_threshold, found_feature, features_left, features_right, targets_left, targets_right
            

    def find_impurity(self, features, targets):

        return self.find_random_impurity(features, targets) if self.splitter == 'random' else self.find_best_impurity(features, targets)
            

    def predict(self, features):
        if self.root_node.left is None:
            raise Exception('we dont have any nodes')

        if isinstance(features, pd.DataFrame):
            features = features.values

        targets = np.empty(features.shape[0])

        # we predict features one by one because view of a view of a view can lose references to original array numpy

        for i in range(len(targets)):
            targets[i] = self.predict_node(self.root_node, features[i, :])

        
        return targets.astype('int')



    def predict_node(self, node: TreeNode, features):

        if node.is_leaf():
            return node.clazz
            
        else:
            feature_value = features[node.feature_index]
            if feature_value >= node.feature_threshold:
                return self.predict_node(node.right, features)
            else:
                return self.predict_node(node.left, features)
