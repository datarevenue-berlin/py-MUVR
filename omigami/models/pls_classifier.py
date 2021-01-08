import logging
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder


def get_vip(model):
    # Calculate VIP
    # https://github.com/scikit-learn/scikit-learn/pull/13492
    # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3883573/#S6title
    T = model.x_scores_
    W = model.x_weights_
    Q = model.y_loadings_
    w0, w1 = W.shape
    s = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
    s_sum = np.sum(s, axis=0)
    w_norm = np.array([(W[:, i] / np.linalg.norm(W[:, i])) for i in range(w1)])
    vip = np.sqrt(w0 * np.sum(s * w_norm.T ** 2, axis=1) / s_sum)
    return vip


class PLSClassifier(PLSRegression):
    """This class extends the scikit-learn PLS regression to
    multi-class classification. To do so, it converts the classification problem to
    a multi dimensional regression. Output classes are one-hot encoded and used
    to train a PLS regressor. Predictions are given by casting the output
    of the regression to a binary matrix where the 1s correspond to the maximum
    of each row. Finally the matrix is converted to classes using the inverse one hot
    transformation.

    The signaure is analogous to sklearn.cross_decomposition.PLSRegression:

    Args:
        n_components (int, optional): Number of components to keep. Defaults to 2.
        scale (bool, optional): whether to scale the data. Defaults to True.
        max_iter (int, optional): the maximum number of iterations of the NIPALS
            inner loop. Defaults to 500.
        tol (float, optional): tolerance used in the iterative algorithm.
            Defaults to 1e-06.
        copy (bool, optional):  Whether the deflation should be done on a copy.
            Let the default value to True unless you don’t care about side effects.
            Defaults to True.
        regressor (PLSRegression, optional): the PLS regressor to be used.
            Specifying this parameter will ignore all the others.
            Leave None unless you have good reason not to. Defaults to None.

    """

    encoder = None
    feature_importances_ = None

    def fit(self, X, Y):
        if self.n_components > X.shape[1]:
            # there might be occasions in which the loops try to fit on a metrix
            # with n_features, a PLS with n_components > n_features. The recursive
            # feature elimination where we stop at n_features = 1 is a clear case
            # of this
            logging.info("Lowering PLSC n_components to %d during fit", X.shape[1])
            self.set_params(n_components=X.shape[1])
        self.encoder = OneHotEncoder().fit(Y.reshape(-1, 1))
        encoded_y = self.encoder.transform(Y.reshape(-1, 1)).toarray()
        super().fit(X, encoded_y)
        self.feature_importances_ = get_vip(self)
        return self

    def predict(self, X, copy=True):
        y_pred = super().predict(X, copy=copy)
        y_pred = np.equal(y_pred.T, y_pred.max(axis=1)).T.astype(float)
        return self.encoder.inverse_transform(y_pred).ravel()
