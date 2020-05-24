import numpy as np
from scipy.sparse import csgraph


class SPEC(object):
    """
        Implements SPEC feature selection
        Zhao Zheng and Liu Huan. "Spectral Feature Selection for Supervised and Unsupervised Learning." ICML 2007.
        http://www.public.asu.edu/~huanliu/papers/icml07.pdf

        Parameters
        ----------
        None

        Examples
        ----------
        from sklearn.datasets import load_iris
        from ITMO_FS.filters import pearson_corr

        data, target = load_iris(True)
        spec = SPEC()
        print(spec.run(data, pearson_corr, -1))
    """

    def __init__(self):
        self.__scores = None
        self.__ranked_features = None

    def __score_features(self, X, measure, style):
        """
            Computes score of each feature.

            Parameters
            ----------
            X : numpy array, shape (n_samples, n_features)
                Input samples.
            measure : string or callable
                A metric name defined in GLOB_MEASURE or a callable with
                signature measure(selected_features, free_features, dataset, labels)
                which should return a list of metric values for each feature in the dataset.
            style : integer number of scoring type
                -1      - score = f'Lf. Measures the value of the normalized cut by using f as
                            the soft cluster indicator to partition the graph G.
                0       - using all eigenvalues except the first. Small score indicates that feature aligns closely
                            to those nontrivial eigenvalues, hence provides good separability.
                k > 0   - using first k except the first. Achieves an effect of reducing noise.

            Returns
            ----------
            None
        """

        if style < -1:
            return

        n_samples, n_features = X.shape

        # building adjacency matrix
        W = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                if i != j:
                    W[i][j] = measure(X[i], X[j])

        # building degree matrix
        d = np.array(W.sum(axis=1))
        D = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            D[i, i] = d[i]

        # building normalized laplacian matrix and calculating spectral information
        L_norm = csgraph.laplacian(W, normed=True)
        s, U = np.linalg.eigh(L_norm)

        # selecting features
        __scores = np.ones(n_features) * 1000
        D2 = np.power(D, 0.5)

        for i in range(n_features):
            f = X[:, i]
            D1 = D2.dot(f)
            D1_norm = np.linalg.norm(D1)
            if D1_norm < 100 * np.spacing(1):
                __scores[i] = 1000
                continue
            else:
                D1 = D1 / D1_norm

            a = np.array(np.dot(np.transpose(D1), U))
            a = np.multiply(a, a)
            a = np.transpose(a)

            if style == -1:
                __scores[i] = np.transpose(D1).dot(L_norm).dot(D1)
            elif style == 0:
                __scores[i] = (np.transpose(D1).dot(L_norm).dot(D1))/(np.transpose(D1).dot(U[0]))
            else:
                a1 = a[n_samples-style:n_samples-1]
                __scores[i] = np.sum(a1 * (2 - s[n_samples-style: n_samples - 1]))

        if style > 0:
            __scores[__scores == 1000] = -1000
        self.__scores = __scores

    def __rank_features(self, score, style):
        """
            Ranks features depending on style.

            Parameters
            ----------
            score : numpy array, shape (n_features)
                Scores of features, computed by SPEC.
            style : integer number of scoring type. Defines how features should be ranked.
                -1      - ranking in descending order, the higher the score, the more relevant feature is.
                0       - ranking in descending order, the higher the score, the more relevant feature is.
                k > 0   - ranking in ascending order, the lower the score, the more relevant feature is.

            Returns
            ----------
            None
        """
        if style < -1:
            return
        sorted_score = np.argsort(score)
        if style > 0:
            self.__ranked_features = sorted_score
        else:
            self.__ranked_features = sorted_score[::-1]

    def run(self, X, measure, style):
        """
            Implements SPEC feature selection.

            Parameters
            ----------
            X : numpy array, shape (n_samples, n_features)
                Input samples.
            measure : string or callable
                A metric name defined in GLOB_MEASURE or a callable with
                signature measure(selected_features, free_features, dataset, labels)
                which should return a list of metric values for each feature in the dataset.
            style : integer number of scoring type. Also influences ranking.
                -1      - score = f'Lf. Measures the value of the normalized cut by using f as
                            the soft cluster indicator to partition the graph G. Ranked in descending order.
                0       - using all eigenvalues except the first. Small score indicates that feature aligns closely
                            to those nontrivial eigenvalues, hence provides good separability.
                            Ranked in descending order.
                k > 0   - using first k except the first. Achieves an effect of reducing noise.
                            Ranked in ascending order.

            Returns
            ----------
            scores: numpy array, shape (n_features)
                Scores of features, depending on style.
            ranked_features: numpy array, shape (n_features_
                Sorted list of features. Depends on style.
        """
        self.__score_features(X, measure, style)
        self.__rank_features(self.__scores, style)
        return self.__scores, self.__ranked_features
