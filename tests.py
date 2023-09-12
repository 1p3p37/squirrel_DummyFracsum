import unittest
import numpy as np
from dummy_regressor_with_fracsum import DummyRegressorWithFracSum

class TestDummyRegressorWithFracSum(unittest.TestCase):

    def test_fit_with_fracsum_strategy(self):
        # Test the fit method with the "fracsum" strategy
        X = np.array([[1, 2], [2, 3], [1, 4]])
        y = np.array([0.5, 1.3, -0.8])

        regressor = DummyRegressorWithFracSum(strategy='fracsum')
        regressor.fit(X, y)

        # Check if constant_ is correctly calculated
        self.assertEqual(regressor.constant_, 1)

        # Check if n_outputs_ is correctly set
        self.assertEqual(regressor.n_outputs_, 1)

    def test_fit_with_other_strategies(self):
        # Test the fit method with other strategies (mean, median)
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([0.5, 1.3, -0.8])

        # Test with mean strategy
        regressor_mean = DummyRegressorWithFracSum(strategy='mean')
        regressor_mean.fit(X, y)
        self.assertEqual(regressor_mean.n_outputs_, 1)

        # Test with median strategy
        regressor_median = DummyRegressorWithFracSum(strategy='median')
        regressor_median.fit(X, y)
        self.assertEqual(regressor_median.n_outputs_, 1)

    def test_fit_with_multioutput_y(self):
        # Test the fit method with multi-output y
        X = np.array([[1, 2], [2, 3], [3, 4]])
        y = np.array([[0.5, 1.0], [1.3, 2.0], [-0.8, -1.0]])

        regressor = DummyRegressorWithFracSum(strategy='fracsum')
        regressor.fit(X, y)

        # Check if n_outputs_ is correctly set for multi-output y
        self.assertEqual(regressor.n_outputs_, 2)

if __name__ == '__main__':
    unittest.main()
