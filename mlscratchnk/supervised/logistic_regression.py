import numpy as np
from sklearn.metrics import f1_score

class LogisticRegression:
    """
    Parameters:
    -----------
    learning_rate: float
        The step length that will be taken when following the negative gradient during
        training. 

    max_iters
        maximum allowed iterations

    tol
        tolerance for stopping criteria

    Attributes:
    ----------
    classes_
        list of known classes
    n_iter_ 
        actual number of iterations made 

    """

    def __init__(self, learning_rate=0.1, random_state=None, max_iters=1000, verbose=False, tol=1e-4):
        self.learning_rate = learning_rate
        self.random_state = np.random.RandomState(random_state)
        self.max_iters = max_iters
        self.verbose = verbose
        self.tol = tol

    def sigmoid(self, z):
        return 1 / (1 + np.e ** (-z))


    def get_current_iterations(self):
        return self.n_iter_
    
    def print_verbose(self, message):
        if self.verbose:
            print(message)

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        n_features = np.shape(X)[1]
        # Initialize parameters between [-1/sqrt(N), 1/sqrt(N)]
        limit = 1 / np.sqrt(n_features)
        self.weights_ = self.random_state.uniform(-limit, limit, (n_features,))

        self.print_verbose(f'initialization classes {self.classes_}, weights {self.weights_}')

        # Move against the gradient of the loss function with
        # respect to the parameters to minimize the loss

        for i in range(self.max_iters):
            self.n_iter_ = i

            y_pred = self.sigmoid(X @ self.weights_)

            self.weights_ -= self.learning_rate * -(y - y_pred).dot(X)

            score = f1_score(y, np.round(y_pred).astype(int))

            self.print_verbose(f'iteration {i}, f1 {score}, weights {self.weights_}')

            
    

    def predict(self, X):
        return np.round(self.sigmoid(X @ self.weights_)).astype(int)