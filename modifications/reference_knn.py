import numpy as np
from scipy.spatial.distance import euclidean
from collections import Counter
import pandas as pd


class KNNModifiedClassifier:
    def __init__(self, k=5, distance_func=euclidean, weight='uniform'):
        self.k = None if k == 0 else k  
        self.distance_func = distance_func
        self.weight = weight
        self.X = None
        self.y = None



    def _setup_input(self, X, y=None):
        if not isinstance(X, np.ndarray): X = np.array(X)
        if X.size == 0: raise ValueError("Got an empty matrix.")
        if X.ndim == 1: self.n_samples, self.n_features = 1, X.shape
        else: self.n_samples, self.n_features = X.shape[0], np.prod(X.shape[1:])
        self.X = X

        if y is None: raise ValueError("Missed required argument y")
        if not isinstance(y, np.ndarray): y = np.array(y)
        if y.size == 0: raise ValueError("The targets array must be non-empty.")
        self.y = y



    def fit(self, X, y=None):
        self._setup_input(X, y)



    def predict(self, X=None):
        if not isinstance(X, np.ndarray): X = np.array(X)
        if self.X is not None: return self._predict(X)
        else: raise ValueError("You must call `fit` before `predict`")



    def aggregate(self, neighbors_targets, distances):
        if self.weight == 'uniform':
            most_common_label = Counter(neighbors_targets).most_common(1)[0][0]
            return most_common_label
        
        elif self.weight == 'distance':
            weighted_labels = Counter()
            for dist, target in zip(distances, neighbors_targets):
                if dist == 0: weight = 1.0
                else: weight = 1 / dist
                weighted_labels[target] += weight
            most_common_label = weighted_labels.most_common(1)[0][0]
            return most_common_label



    def _predict(self, X=None):
        predictions = [self._predict_x(x) for x in X]
        return np.array(predictions)



    def _predict_x(self, x):
        distances = [self.distance_func(x, example) for example in self.X]
    
        if self.weight == 'distance':
            zero_distances = [i for i, dist in enumerate(distances) if dist == 0]
            if zero_distances:
                weights = np.zeros(len(distances))
                weights[zero_distances] = 1.0
            else: weights = 1.0 / np.array(distances)
        else: weights = np.ones(len(distances))

        neighbors = sorted(zip(distances, self.y, weights), key=lambda x: x[0]) # Create tuples sorted by distance
        neighbors_targets = [target for (_, target, _) in neighbors[: self.k]] # Get the target label of k nearest neighbours
        neighbor_weights = [weight for (_, _, weight) in neighbors[: self.k]] # Get the weight of k nearest neighbours
        return self.aggregate(neighbors_targets, neighbor_weights)

    

