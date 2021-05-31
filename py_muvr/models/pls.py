import logging
from typing import Union
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder
from py_muvr.data_structures.data_types import NumpyArray


log = logging.getLogger(__name__)


def get_vip(model: PLSRegression) -> NumpyArray:
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

    It also provides `feature_importances_` as VIP and prevent fit failures
    if the number of components is bigger than the number of features

    The signature is analogous to sklearn.cross_decomposition.PLSRegression:

    Parameters
    ----------
    n_components: int, optional
        Number of components to keep, by default 2.
    scale: bool, optional
        whether to scale the data, by default True.
    max_iter: int, optional
        the maximum number of iterations of the NIPALS inner loop, by default 500.
    tol: float, optional
        tolerance used in the iterative algorithm, by default 1e-06.
    copy: bool, optional
        Whether the deflation should be done on a copy. Let the default value to True
        unless you don’t care about side effects, by default True.

    """

    encoder = None
    feature_importances_ = None
    _estimator_type = "classifier"

    def fit(self, X: NumpyArray, Y: NumpyArray):
        if self.n_components > X.shape[1]:
            reduce_pls_components(self, X.shape[1])
        self.encoder = OneHotEncoder().fit(Y.reshape(-1, 1))
        encoded_y = self.encoder.transform(Y.reshape(-1, 1)).toarray()
        super().fit(X, encoded_y)
        self.feature_importances_ = get_vip(self)
        return self

    def predict(self, X: NumpyArray, copy: bool = True):
        y_pred = super().predict(X, copy=copy)
        y_pred = np.equal(y_pred.T, y_pred.max(axis=1)).T.astype(float)
        return self.encoder.inverse_transform(y_pred).ravel()


class PLSRegressor(PLSRegression):
    """This class extends the scikit-learn PLS regression.
    It provides `feature_importances_` as VIP and prevent fit failures
    if the number of components is bigger than the number of features

    The signature is analogous to sklearn.cross_decomposition.PLSRegression:

    Parameters
    ----------
    n_components: int, optional
        Number of components to keep, by default 2.
    scale: bool, optional
        whether to scale the data, by default True.
    max_iter: int, optional
        the maximum number of iterations of the NIPALS inner loop, by default 500.
    tol: float, optional
        tolerance used in the iterative algorithm, by default 1e-06.
    copy: bool, optional
        Whether the deflation should be done on a copy. Let the default value to True
        unless you don’t care about side effects, by default True.

    """

    feature_importances_ = None

    def fit(self, X: NumpyArray, Y: NumpyArray):
        if self.n_components > X.shape[1]:
            reduce_pls_components(self, X.shape[1])
        super().fit(X, Y)
        self.feature_importances_ = get_vip(self)
        return self

    def predict(self, X: NumpyArray, copy: bool = True):
        return super().predict(X, copy=copy).ravel()


def reduce_pls_components(pls: Union[PLSRegressor, PLSClassifier], n_components: int):
    # there might be occasions in which the loops try to fit on a matrix
    # with n_features, a PLS with n_components > n_features. The recursive
    # feature elimination where we stop at n_features = 1 is a clear case
    # of this
    log.debug("Lowering PLS n_components to %s during fit", n_components)
    pls.set_params(n_components=n_components)
