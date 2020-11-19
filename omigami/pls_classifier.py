import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import OneHotEncoder


class PLSClassifier(BaseEstimator):
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

    def __init__(
        self,
        n_components: int = 2,
        scale: bool = True,
        max_iter: int = 500,
        tol: float = 1e-06,
        copy: bool = True,
        regressor: PLSRegression = None,
    ):
        self.regressor = regressor
        if not isinstance(regressor, PLSRegression):
            self.regressor = PLSRegression(
                n_components=n_components,
                scale=scale,
                max_iter=max_iter,
                tol=tol,
                copy=copy,
            )
        self.coef_ = None
        self.encoder = None
        self.n_components = n_components
        self.scale = scale
        self.max_iter = max_iter
        self.tol = tol
        self.copy = copy

    def fit(self, X, y):
        self.encoder = OneHotEncoder().fit(y.reshape(-1, 1))
        encoded_y = self.encoder.transform(y.reshape(-1, 1)).toarray()
        self.regressor.fit(X, encoded_y)
        self.coef_ = [np.abs(self.regressor.coef_).sum(axis=1)]
        return self

    def predict(self, X):
        y_pred = self.regressor.predict(X)
        y_pred = np.equal(y_pred.T, y_pred.max(axis=1)).T.astype(float)
        return self.encoder.inverse_transform(y_pred).ravel()
