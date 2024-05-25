
from multiprocessing import Pool
import hashlib
from collections import Counter
from skimage.feature import local_binary_pattern
import cv2
import numpy as np
import torch

class PIHA:
    def __init__(self, arguments):
        self.input_shape = arguments["input_shape"]
        self.block_size = arguments["block_size"]
        self.input_idx = 0
        self.cache = self.getDigest(np.zeros(arguments["input_shape"]))

    def _piha_hash(self, x):
        N = x.shape[2]
        # Image preprocessing
        x = x.transpose(1, 2, 0)
        x_filtered = cv2.GaussianBlur(x, (3, 3), 1)

        # Color space transformation
        x_filtered = np.float32(x_filtered)
        x_hsv = cv2.cvtColor(x_filtered, cv2.COLOR_RGB2HSV)

        # Use only H channel for HSV color space
        x_h = x_hsv[:, :, 0].reshape((N, N, 1))
        x_h = np.pad(x_h,
                     ((0, self.block_size - N % self.block_size), (0, self.block_size - N % self.block_size), (0, 0)),
                     'constant')
        N = x_h.shape[0]

        # Block division and feature matrix calculation
        blocks_h = [x_h[i:i + self.block_size, j:j + self.block_size] for i in range(0, N, self.block_size) for j in
                    range(0, N, self.block_size)]
        features_h = np.array([np.sum(block) for block in blocks_h]).reshape(
            (N // self.block_size, N // self.block_size))

        # Local binary pattern feature extraction
        features_lbp = local_binary_pattern(features_h, 8, 1)

        # Hash generation
        # hash_array = ''.join([f'{(int(_)):x}' for _ in features_lbp.flatten().tolist()])
        # hash_array = ''.join([format(int(_), '02x') for _ in features_lbp.flatten().tolist()])
        hash_array = features_lbp.flatten()
        hash_array = np.expand_dims(hash_array, axis=0)
        return hash_array

    def getDigest(self, img):
        h = self._piha_hash(img)
        return h

    def resetCache(self):
        self.cache = self.getDigest(torch.zeros(tuple(self.input_shape)))

    def add(self, img):
        self.input_idx += 1
        hash = self.getDigest(img)
        self.cache = np.concatenate((self.cache, hash))

    def resultsTopk(self, img, k):
        hash = self.getDigest(img)
        hamming_dists = np.count_nonzero(hash != self.cache, axis=1) / self.cache.shape[1]
        closest = np.argsort(hamming_dists)[:k]
        # remove dummy element if present
        closest = closest[closest != 0]
        if len(closest) == 0:
            return []
        result = [hamming_dists[i] for i in closest]
        return result

