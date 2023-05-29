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
        expected_output = tf.constant(y)
        real_output = lbc(x)

        tf.debugging.assert_equal(expected_output, real_output)

    def test_SubLayer1(self):
        sub_layer = FirstSubLayerLBC2D()
        x = np.array([
            [ 
                [[1],[2],[3],[4]], 
                [[5],[6],[7],[8]], 
                [[9],[8],[7],[6]], 
                [[5],[4],[3],[2]]
            ]
            ])
        anchor_weights = np.array(
            [
                [[[1,0]],[[0,-1]]],
                [[[0,0]],[[1,0]]]
                
            ]
        )
        expected_y = np.array([
            [
                [[7, -2],   [9, -3],    [11, -4],   [4, 0]], 
                [[13, -6],  [13, -7],   [13, -8],   [8, 0]], 
                [[13, -8],  [11, -7],   [9, -6],    [6, 0]], 
                [[5, -4],   [4, -3],    [3, -2],    [2, 0]]
             ]
            ])
        real_y = sub_layer.calculate(x, anchor_weights).numpy()
        self.assertTrue(np.array_equal(expected_y, real_y))

    def test_SubLayer1_2channels(self):
        sub_layer = FirstSubLayerLBC2D()
        x = np.array([
            [ 
                [[1,1],[2,2],[3,3],[4,4]], 
                [[5,5],[6,6],[7,7],[8,8]], 
                [[9,9],[8,8],[7,7],[6,6]], 
                [[5,5],[4,4],[3,3],[2,2]]
            ]
            ])
        anchor_weights = np.array(
            [
                [[[1,0],[1,0]],[[0,-1], [0,-1]]],
                [[[0,0],[0,0]],[[1,0],  [1,0]]]
                
            ]
        )
        expected_y = np.array([
            [
                [[14, -4],   [18, -6],    [22, -8],   [8, 0]], 
                [[26, -12],  [26, -14],   [26, -16],   [16, 0]], 
                [[26, -16],  [22, -14],   [18, -12],    [12, 0]], 
                [[10, -8],   [8, -6],    [6, -4],    [4, 0]]
             ]
            ])
        real_y = sub_layer.calculate(x, anchor_weights).numpy()
        self.assertTrue(np.array_equal(expected_y, real_y))

    def test_SubLayer2(self):
        sub_layer = SecondSubLayerLBC2D()
        intermediary_feature_map = np.array([
            [
                [[14, -4],   [18, -6],    [22, -8],   [8, 0]], 
                [[26, -12],  [26, -14],   [26, -16],   [16, 0]], 
                [[26, -16],  [22, -14],   [18, -12],    [12, 0]], 
                [[10, -8],   [8, -6],    [6, -4],    [4, 0]]
             ]
            ])
        x = tf.constant(intermediary_feature_map)
        filters = np.array([[[[1],[2]]]])
        real_y = sub_layer.calculate(x, filters).numpy()
        expected_y = np.array([[
                [[6],   [6],    [6],    [8]],
                [[2],   [-2],   [-6],   [16]],
                [[-6],  [-6],   [-6],  [12]],
                [[-6],  [-4],   [-2],   [4]]
            ]])
        
        self.assertTrue(np.array_equal(expected_y, real_y))