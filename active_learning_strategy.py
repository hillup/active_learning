# -*- encoding: utf-8 -*-
# @Time: 2021/9/28 11:40
# @Author: mahanghang
# @File: active_learning_strategy.py

import os
import numpy as np

from sklearn.cluster import KMeans
from scipy.special import softmax


class active_learning_selection:
    def __init__(self, unlabeled_scores, num_select, labeled_scores=None):
        """
        :param unlabeled_scores: unlabeled pool's scores,the shape is [batch_size, num_classes]
        :param labeled_scores: training samples pool's scores,the shape is [batch_size, num_classes]
        :param num_select: the number need to label
        """
        self.unlabeled_scores = unlabeled_scores
        self.labeled_scores = labeled_scores
        self.num_select = num_select

    def random_sample(self):
        select_idx = np.random.choice(a=self.unlabeled_scores.shape[0], size=self.num_select, replace=False)
        return select_idx

    def least_confident_sample(self):
        # shape: [batch_size, num_classes] -> [batch_size]
        unlabeled_scores = self.unlabeled_scores.max(axis=1)
        select_idx = np.argsort(unlabeled_scores)
        # select data that max score is least
        return select_idx[:self.num_select]

    def margin_sample(self):
        sorted_scores = np.sort(self.unlabeled_scores)
        margin = sorted_scores[:, -1] - sorted_scores[:, -2]
        select_idx = np.argsort(margin)
        # select data that margin is least
        return select_idx[:self.num_select]

    def entropy_sample(self):
        log_scores = np.log(self.unlabeled_scores)
        entropy = np.sum(- log_scores * self.unlabeled_scores, axis=1)
        select_idx = np.argsort(entropy)
        # select data that entropy is maximum
        return select_idx[-self.num_select:]

    def k_means_sample(self):
        kmeans = KMeans(n_clusters=self.num_select)
        kmeans.fit(self.unlabeled_scores)
        # get cluster id for each sample
        cluster_ids = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_
        sample_centers = cluster_centers[cluster_ids]
        distances = ((self.unlabeled_scores - sample_centers) ** 2).sum(axis=1)
        # each cluster id should have the nearest sample
        select_idx = []
        for idx in range(self.num_select):
            cluster_samples = np.arange(self.unlabeled_scores.shape[0])[cluster_ids==idx]
            if cluster_samples[0] < 1:
                continue
            select_idx.append(cluster_samples[distances[cluster_samples].argmin()])
        return np.array(select_idx)

    def k_center_greedy(self):
        if not self.labeled_scores:
            raise ValueError("Input labeled_scores is not valid")



if __name__ == "__main__":
    predict_scores = softmax(np.random.random((1000, 261)), axis=1)
    ac_lr = active_learning_selection(predict_scores, 100)
    # use least confident sample
    least_confident_sample = ac_lr.least_confident_sample()
    from IPython import embed
    embed()









