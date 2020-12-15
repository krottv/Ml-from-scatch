import numpy as np
import pandas as pd

class TreeNode():

    left = None
    right = None

    def __init__(self, feature_index = None, feature_threshold = 0, impurity = 0, is_left = None,
     depth = 0, nsamples=0, hindex=0, clazz=None, clazz_percentage=0, impurity_decrease = 0):

        self.feature_index = feature_index
        self.feature_threshold = feature_threshold
        self.is_left = is_left
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

    def displayName(self, label_classes = None):
        displayClass = self.clazz if label_classes is None else label_classes[self.clazz]

        if self.depth == 0:
            return 'root node impurity = {:.2f}, nsamples {:}'.format(self.impurity, self.nsamples)

        impurity_decrease_str = "" if self.is_leaf() else "impr decr {:0.2%}".format(self.impurity_decrease)

        return '{:} {:} {:.2f}, impurity {:.2f}\ndepth {:}, hindex {:}, nsamples {:}\nclass {:} {:0.2%} {}'.format(self.feature_index, "<" if self.is_left else ">=", self.feature_threshold, self.impurity, self.depth, self.hindex, self.nsamples, displayClass, self.clazz_percentage, impurity_decrease_str)


    def __str__(self):
        return self.displayName()



class CustomDecisionTree():

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

        self.classes = target.unique()
        value_counts = target.value_counts()

        # it makes value_count in a good order
        root_samples = [value_counts[self.classes[x]] for x in range(0, len(value_counts))]
        self.total_samples = len(target)
        root_impurity = self.calc_entropy(root_samples, self.total_samples) if self.criterion == 'entropy' else self.calc_gini(root_samples, self.total_samples)

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

        self.print(f'found best impurity gain = {impurity_gain}, found_threshold = {found_threshold}, feature {found_feature_index}, new depth = {node.depth + 1} current node is left {node.is_left}')


        if targets_left is None or targets_right is None or self.min_samples_leaf > len(targets_left) or self.min_samples_leaf > len(targets_right):

            left_counts = targets_left.value_counts() if targets_left is not None else None
            right_counts = targets_right.value_counts() if targets_right is not None else None

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
        
        left_clazz, left_clazz_percentage = self.get_clazz_and_its_percentage(targets_left)
        node.left = TreeNode(feature_index = found_feature_index, feature_threshold = found_threshold, impurity = min_impurity_left, is_left = True, depth = new_depth, nsamples=len(targets_left), hindex=node.hindex * 2, clazz=left_clazz, clazz_percentage=left_clazz_percentage)
        self.split_inner_node(node.left, features_left, targets_left)

        right_clazz, right_clazz_percentage = self.get_clazz_and_its_percentage(targets_right)
        node.right = TreeNode(feature_index = found_feature_index, feature_threshold = found_threshold, impurity = min_impurity_right, is_left = False, depth = new_depth, nsamples=len(targets_right), hindex=node.hindex*2 + 1, clazz=right_clazz, clazz_percentage=right_clazz_percentage)
        self.split_inner_node(node.right, features_right, targets_right)
        
        return True

    def get_clazz_and_its_percentage(self, targets_left):
        left_value_counts = targets_left.value_counts()
        left_clazz = left_value_counts.index[left_value_counts.argmax()]
        left_clazz_percentage = left_value_counts[left_clazz] / left_value_counts.sum()
        return (left_clazz, left_clazz_percentage)


    def separate_and_get_impurity(self, threshold, features, feature_index, targets):
        features_more = features[features[feature_index] >= threshold]
        features_less = features[features[feature_index] < threshold]
        targets_more = targets[features_more.index]
        targets_less = targets[features_less.index]

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
            unique =  np.sort(features[feature].unique())
            
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
            
            feature = features.columns[feature_index]
            unique =  np.sort(features[feature].unique())
            
            mx = unique[-2]
            mn = unique[1]

            threshold = (mx - mn) * random_numbers[feature_index] + mn

            features_more, features_less, targets_more, targets_less, left_impurity, right_impurity = self.separate_and_get_impurity(threshold, features, feature, targets)

            if left_impurity + right_impurity < (min_impurity_left + min_impurity_right):

                min_impurity_left, min_impurity_right = left_impurity, right_impurity
                features_left, features_right = features_less, features_more
                targets_left, targets_right = targets_less, targets_more
                found_threshold = threshold
                found_feature = feature

        
        return min_impurity_left, min_impurity_right, found_threshold, found_feature, features_left, features_right, targets_left, targets_right
            

    def find_impurity(self, features, targets):

        return self.find_random_impurity(features, targets) if self.splitter == 'random' else self.find_best_impurity(features, targets)
            

    def predict(self, features):
        targets = pd.Series([None] * len(features.index), index=features.index)

        if self.root_node.left is not None:
            self.predict_node(self.root_node.left, True, features, targets)
            self.predict_node(self.root_node.right, False, features, targets)
        else:
            raise Exception('we dont have any nodes')
        
        return targets.astype(int)



    def predict_node(self, node: TreeNode, is_left: bool, features, targets):

        condition = features[node.feature_index] < node.feature_threshold if is_left else features[node.feature_index] >= node.feature_threshold
        selected_features = features[condition]

        if node.is_leaf():
            targets[selected_features.index] = node.clazz
        else:
            self.predict_node(node.left, True, selected_features, targets)
            self.predict_node(node.right, False, selected_features, targets)

