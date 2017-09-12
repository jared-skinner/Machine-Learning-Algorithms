import numpy as np
import unittest
from training_model import TrainingModel

class TestHappy(unittest.TestCase):

    def test_sigmoid(self):
        self.assertAlmostEqual(TrainingModel.sigmoid([1],[2]), 0.880797077978, places=7)

        a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        b = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

        np.testing.assert_allclose(np.array([[0.9999833, 0.99999917, 0.99999996, 1.], [ 1., 1., 1., 1.],[ 1., 1., 1., 1.],[ 1., 1., 1., 1.]]), TrainingModel.sigmoid(a, b))


if __name__ == '__main__':
    unittest.main()
