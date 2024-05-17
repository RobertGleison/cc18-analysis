import numpy as np
from scipy.spatial.distance import euclidean
# from scipy.spatial.distance import cityblock
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import chi2
import scipy.linalg as sp
from scipy.stats import chi2_contingency
from collections import Counter
import pandas as pd




class KNNModifiedClassifier:
    def __init__(self, k=5, distance_func=euclidean, weight='uniform'):
        self.k = None if k == 0 else k  
        self.distance_func = distance_func
        self.weight = weight
        self.chi_correlations = []
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

        if self.weight == 'correlations':
            self.get_chi_correlations()



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
        
        weighted_labels = Counter() # Work for correlations and for distance
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
        if self.weight == 'correlations':
            distances = [self.corr_manhattan(x, example) for example in self.X]
        else: distances = [self.distance_func(x, example) for example in self.X]
    
        if self.weight == 'distance' or self.weight == 'correlations':
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


    def get_chi_correlations(self, strategy='uniform'):
        bins = int(np.sqrt(self.X.shape[0]))
        chi2_results = []
        features = pd.DataFrame(self.X)
        target = self.y
        
        for col in features.columns:
            if features[col].dtype.kind in 'bifc':  # if the column is continuous
                kbins = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy=strategy)
                discretized_col = kbins.fit_transform(features[[col]]).astype(int).flatten()
            else:
                discretized_col = features[col]
            contingency_table = pd.crosstab(discretized_col, target)
            chi2_stat, *_= chi2_contingency(contingency_table)  # Include p-value if needed
            chi2_results.append(chi2_stat)
        
        self.chi_correlations=chi2_results
    


    def _validate_vector(self, u, dtype=None):
        u = np.asarray(u, dtype=dtype, order='c')
        if u.ndim == 1:
            return u
        raise ValueError("Input vector should be 1-D.")



    def _validate_weights(self, w, dtype=np.float64):
        w = self._validate_vector(w, dtype=dtype)
        if np.any(w < 0):
            raise ValueError("Input weights should be all non-negative")
        return w
    

    def corr_manhattan(self, u, v):
        u = self._validate_vector(u)
        v = self._validate_vector(v)
        l1_diff = abs(u - v)
        correlationed_diff = l1_diff * self.chi_correlations
        return correlationed_diff.sum()


    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self


