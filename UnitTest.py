import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
from lbcnn import *

class UnitTestLBC(unittest.TestCase):
    def test_LBC(self):
        test_sub_layer1 = TestSubLayer()
        test_sub_layer2 = TestSubLayer()
        anchor_weights = np.array(
            [
                [[1,2],[3,4]],
                [[5,6],[7,8]]
            ]
        )
        lbc = LBC(2, anchor_weights, test_sub_layer1, test_sub_layer2)
        x = np.array([ [1,2,3,4], [5,-6,7,8], [9,8,7,6], [5,4,3,-2] ])
        y = np.array([ [1,2,3,4], [5,0,7,8], [9,8,7,6], [5,4,3,0] ])
        expected_output = tf.constant(np.array([ [1,2,3,4], [5,0,7,8], [9,8,7,6], [5,4,3,0] ]))
        real_output = lbc(x)

        tf.debugging.assert_equal(expected_output, real_output)

