# -*- encoding: utf-8 -*-
# @Time: 2021/9/28 11:40
# @Author: mahanghang
# @File: ac_utils.py

import os
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from scipy.special import softmax
from scipy.spatial.distance import cdist, squareform

def cal_distances(vec1, vec2):
    return cdist(vec1, vec2)

if __name__ == "__main__":
    labeled_score = np.random.random((20000, 261))
    unlabeled_score = np.random.random((1000, 261))
    cal_distances(labeled_score, unlabeled_score)









