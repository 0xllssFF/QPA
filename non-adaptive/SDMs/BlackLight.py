
import hashlib
from collections import Counter
import sys
import numpy as np
from multiprocessing import Pool
import pickle
import imp

class BlackLight:
    def __init__(self, arguments):
        self.window_size = arguments["window_size"]
        self.num_hashes_keep = arguments["num_hashes_keep"]
        self.round = arguments["round"]
        self.step_size = arguments["step_size"]
        self.input_shape = arguments["input_shape"]
        self.salt_type = arguments["salt"]
        if self.salt_type:
            self.salt = (np.random.rand(*tuple(self.input_shape)) * 255).astype(np.int16)
        else:
            self.salt = (np.zeros(tuple(self.input_shape)) * 255).astype(np.int16)

        self.inverse_cache = {}
        self.input_idx = 0
        self.pool = Pool(processes=arguments["num_processes"])


    @staticmethod
    def hash_helper(arguments):
        img = arguments['img']
        idx = arguments['idx']
        window_size = arguments['window_size']
        return hashlib.sha256(img[idx:idx + window_size]).hexdigest()

    def preprocess(self, array, salt, round=1, normalized=True):
        if len(array.shape) != 3:
            raise Exception("expected 3d image")
        if (normalized):
            # input image normalized to [0,1]
            array = np.array(array) * 255.
        array = (array + salt) % 255.
        array = array.reshape(-1)
        array = np.around(array / round, decimals=0) * round
        array = array.astype(np.int16)
        return array

    def getDigest(self, img):
        img = self.preprocess(img, self.salt, self.round)
        total_len = int(len(img))
        idx_ls = []

        for el in range(int((total_len - self.window_size + 1) / self.step_size)):
            idx_ls.append({"idx": el * self.step_size, "img": img, "window_size": self.window_size})
        hash_list = self.pool.map(BlackLight.hash_helper, idx_ls)
        hash_list = list(set(hash_list))
        hash_list = [r[::-1] for r in hash_list]
        hash_list.sort(reverse=True)
        return hash_list[:self.num_hashes_keep]

    def resetCache(self):
        self.inverse_cache = {}
        self.input_idx = 0
        if self.salt_type:
            self.salt = (np.random.rand(*tuple(self.input_shape)) * 255).astype(np.int16)
        else:
            self.salt = (np.zeros(tuple(self.input_shape)) * 255).astype(np.int16)

    def add(self, img):
        self.input_idx += 1
        hashes = self.getDigest(img)
        for el in hashes:
            if el not in self.inverse_cache:
                self.inverse_cache[el] = [self.input_idx]
            else:
                self.inverse_cache[el].append(self.input_idx)
    def resultsTopk(self, img, k):
        hashes = self.getDigest(img)
        sets = list(map(self.inverse_cache.get, hashes))
        sets = [i for i in sets if i is not None]
        sets = [item for sublist in sets for item in sublist]
        if not sets:
            return []
        sets = Counter(sets)
        result = [x[1] for x in sets.most_common(k)]
        return result
