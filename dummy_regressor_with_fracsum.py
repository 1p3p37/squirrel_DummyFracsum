import numpy as np
from sklearn.dummy import DummyRegressor

class DummyRegressorWithFracSum(DummyRegressor):
    """
    DummyRegressorWithFracSum is an extension of DummyRegressor from scikit-learn
    that adds a new strategy "fracsum" which returns the sum of fractional parts
    of the training target values similar to the "mean" and "median" strategies.

    Parameters
    ----------
    strategy : {"mean", "median", "quantile", "constant", "fracsum"}, default="mean"
        Strategy to use to generate predictions.

        * "mean": always predicts the mean of the training set
        * "median": always predicts the median of the training set
        * "quantile": always predicts a specified quantile of the training set,
          provided with the quantile parameter.
        * "constant": always predicts a constant value that is provided by
          the user.
        * "fracsum": predicts the sum of fractional parts of the training target values.

    constant : int or float or array-like of shape (n_outputs,), default=None
        The explicit constant as predicted by the "constant" strategy.
        This parameter is useful only for the "constant" strategy.

    quantile : float in [0.0, 1.0], default=None
        The quantile to predict using the "quantile" strategy. A quantile of
        0.5 corresponds to the median, while 0.0 to the minimum and 1.0 to the
        maximum.

    Attributes
    ----------
    constant_ : ndarray of shape (1, n_outputs)
        Mean or median or quantile of the training targets or constant value
        given by the user.

    n_outputs_ : int
        Number of outputs.

    See Also
    --------
    DummyRegressor: Regressor that makes predictions using simple rules.
    """

    def __init__(self, *, strategy="fracsum", constant=None, quantile=None):
        super().__init__(strategy=strategy, constant=constant, quantile=quantile)

    def fit(self, X, y, sample_weight=None):
        """Fit the random regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # super()._validate_params()

        # Calculate the sum of fractional parts
        # fractional_sum = np.sum(np.mod(y, 1))
        fractional_sum = np.sum(y - np.floor(y))


        # Store the result as the constant_
        self.constant_ = np.array([fractional_sum])

        # n_outputs_ количество выходных переменных (число регрессионных целей)
        """
        y.ndim возвращает количество измерений (dimensions) в массиве y. 
        Если y - одномерный массив, то y.ndim вернет 1, а если y - двумерный массив, то вернет 2.
        """
        if y.ndim == 1:
            self.n_outputs_ = 1
        else:
            self.n_outputs_ = y.shape[1] # количества столбцов 

        return self
