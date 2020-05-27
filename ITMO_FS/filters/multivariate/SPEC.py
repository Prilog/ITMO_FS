import numpy as np
from .measures import GLOB_MEASURE
from scipy.sparse import csgraph


class SPEC(object):
    """
        Implements SPEC feature selection
        Zhao Zheng and Liu Huan. "Spectral Feature Selection for Supervised and Unsupervised Learning." ICML 2007.
        http://www.public.asu.edu/~huanliu/papers/icml07.pdf

        Parameters
        ----------
        measure : string or callable
                A metric name defined in GLOB_MEASURE or a callable with
                signature measure(selected_features, free_features, dataset, labels)
                which should return a list of metric values for each feature in the dataset.
        k : integer number of scoring type. Also influences ranking.
            k = -1  - score = f'Lf. Measures the value of the normalized cut by using f as
                        the soft cluster indicator to partition the graph G. Ranked in descending order.
            k = 0   - using all eigenvalues except the first. Small score indicates that feature aligns closely
                        to those nontrivial eigenvalues, hence provides good separability.
                        Ranked in descending order.
            k > 0   - using first k except the first. Achieves an effect of reducing noise.
                        Ranked in ascending order.

        Examples
        ----------
        from sklearn.datasets import load_iris
        from ITMO_FS.filters import pearson_corr
        from ITMO_FS.filters.multivariate import SPEC

        data, target = load_iris(True)
        spec = SPEC()
        print(spec.run(data, pearson_corr, -1))
    """

    def __init__(self, measure, k):
        if type(measure) is str:
            try:
                self.__measure = GLOB_MEASURE[measure]
            except KeyError:
                raise KeyError("No %r measure yet" % measure)

        if k < -1:
            raise Exception('k must me >= -1')
        self.__measure = measure
        self.__k = k
        self.__desc = None
        self.__ranked_features = None

    def fit(self, X):
        """
            Computes score of each feature.

            Parameters
            ----------
            X : numpy array, shape (n_samples, n_features)
                Input samples.

            Returns
            ----------
            None
        """
        n_samples, n_features = X.shape

        # building adjacency matrix
        W = np.array([[self.__measure(i, j) for i in X] for j in X])
        W = W - np.diag(np.diag(W))

        # building degree matrix
        D = np.diag(W.sum(axis=1))

        # building normalized laplacian matrix and calculating spectral information
        L_norm = csgraph.laplacian(W, normed=True)
        s, U = np.linalg.eigh(L_norm)

        # selecting features
        __desc = np.ones(n_features) * 1000
        D2 = np.power(D, 0.5)

        for i in range(n_features):
            f = X[:, i]
            D1 = D2.dot(f)
            D1_norm = np.linalg.norm(D1)
            if D1_norm < 100 * np.spacing(1):
                __desc[i] = 1000
                continue
            else:
                D1 = D1 / D1_norm

            a = np.array(np.dot(np.transpose(D1), U))
            a = np.multiply(a, a)
            a = np.transpose(a)

            if self.__k == -1:
                __desc[i] = np.transpose(D1).dot(L_norm).dot(D1)
            elif self.__k == 0:
                __desc[i] = (np.transpose(D1).dot(L_norm).dot(D1))/(np.transpose(D1).dot(U[0]))
            else:
                a1 = a[n_samples-self.__k:n_samples-1]
                __desc[i] = np.sum(a1 * (2 - s[n_samples-self.__k: n_samples - 1]))

        if self.__k > 0:
            __desc[__desc == 1000] = -1000
        self.__desc = __desc

    def transform(self, score):
        """
            Ranks features depending on k.

            Parameters
            ----------
            score : numpy array, shape (n_features)
                Scores of features, computed by SPEC.

            Returns
            ----------
            ranked_features: numpy array, shape (n_features)
                Sorted list of features. Depends on k.
        """
        sorted_score = np.argsort(score)
        if self.__k > 0:
            self.__ranked_features = sorted_score
        else:
            self.__ranked_features = sorted_score[::-1]
        return self.__ranked_features

    def fit_transform(self, X):
        """
            Computes scores of features and ranks them.

            Parameters
            ----------
            X : numpy array, shape (n_samples, n_features)
                Input samples.

            Returns
            ----------
            ranked_features: numpy array, shape (n_features)
                Sorted list of features. Depends on k.
        """
        self.fit(X)
        return self.transform(self.__desc)
