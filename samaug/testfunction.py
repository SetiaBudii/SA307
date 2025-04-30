import unittest
import numpy as np
from randomsampling import get_random_point

class TestRandomSampling(unittest.TestCase):
    def setUp(self):
        self.mask = np.array([
            [False, True, False],
            [True, False, True],
            [False, True, False]
        ])

    def test_get_random_point(self):
        result = get_random_point(self.mask)
        valid_points = np.argwhere(self.mask == True)
        print(f"Valid points: {valid_points.tolist()}")
        print(f"Random point: {result}")
        self.assertIn(result, valid_points.tolist())

if __name__ == '__main__':
    unittest.main()