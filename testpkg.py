import unittest
import setuptools_sandbox
import numpy as np
import numpy.testing

from setuptools_sandbox import addfloats, makefloats


class TestSandbox(unittest.TestCase):
    def test_addfloats(self):
        """
        Test sum of two floats
        """
        result = addfloats(3,4)
        self.assertEqual(result, 7.0)

    def test_makefloats(self):
        """
        Test returning an array of floats
        """
        data = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0, 81.0])
        result = makefloats(10)
        numpy.testing.assert_array_equal(result, data)

if __name__ == '__main__':
    unittest.main()

