import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
np.random.seed(666)
import threading
import PIL
from PIL import Image
from multiprocessing import Pool
import hashlib
from collections import Counter
import json
import networkx as nx
from tqdm import tqdm
from argparse import ArgumentParser
import cv2
from skimage.feature import local_binary_pattern


def hash_helper(arguments):
    img = arguments['img']
    idx = arguments['idx']
    window_size = arguments['window_size']
    return hashlib.sha256(img[idx:idx + window_size]).hexdigest()

class CD_BlackLight:
    def __init__(self, query, window_size, num_hashes_keep, round=50, step_size=1, workers=5, salt=None):
        self.window_size = window_size
        self.num_hashes_keep = num_hashes_keep
        self.round = round
        self.step_size = step_size
        if(salt != None):
            self.salt = salt
        else:
            self.salt = np.random.rand(*query.shape) * 255.

        self.hash_dict = {}
        self.pool = Pool(processes=workers)
        self.input_idx = 0
        self.output = {}

        self.g = nx.DiGraph()
        self.alerted_nodes = set()
        self.threshold = 2


    def preprocess(self, array, round=1, normalized=True):
        if (normalized):
            # input image normalized to [0,1]
            array = np.array(array) * 255.
        array = (array + self.salt) % 255.
        array = array.reshape(-1)
        array = np.around(array / round, decimals=0) * round
        array = array.astype(np.int16)
        return array


    def hash_img(self, img, window_size, round, step_size, preprocess=True):
        if preprocess:
            img = self.preprocess(img, round)
        total_len = int(len(img))
        idx_ls = []
        for el in range(int((total_len - window_size + 1) / step_size)):
            idx_ls.append({"idx": el * step_size, "img": img, "window_size": window_size})
        hash_list = self.pool.map(hash_helper, idx_ls)
        hash_list = list(set(hash_list))
        hash_list = [r[::-1] for r in hash_list]
        hash_list.sort(reverse=True)
        return hash_list
        
    def check_img(self, hashes):
        sets = list(map(self.hash_dict.get, hashes))
        sets = [i for i in sets if i is not None]
        sets = [item for sublist in sets for item in sublist]
        if not sets or len(sets) == 0:
            return 0, 0
        sets = Counter(sets)
        cnt = sets.most_common(1)[0][1]
        idx = sets.most_common(1)[0][0]
        return idx, cnt
        
    def add_img(self, img):
        hashes = self.hash_img(img, self.window_size, self.round, self.step_size)[:self.num_hashes_keep]
        idx, cnt = self.check_img(hashes)
        for el in hashes:
            if el not in self.hash_dict:
                self.hash_dict[el] = [self.input_idx]
            else:
                self.hash_dict[el].append(self.input_idx)
        self.g.add_node(self.input_idx)
        if cnt > self.threshold:
            self.g.add_edge(idx, self.input_idx, label=cnt)
        self.input_idx += 1
        return cnt, idx


class CD_PIHA:
    def __init__(self, input_shape, block_size, threshold):
        self.input_shape = input_shape
        self.block_size = block_size
        self.input_idx = 0
        self.cache = self.getDigest(torch.zeros(input_shape))

        self.hash_dict = {}
        self.input_idx = 0
        self.output = {}
        self.g = nx.DiGraph()
        # self.graph_model = torch.load(model_path)
        # self.graph_model.eval()

        self.threshold = threshold
        print("Threshold: {}".format(self.threshold))

    def _piha_hash(self, x):
        N = x.shape[2]
        # Image preprocessing
        x = x.numpy().transpose(1, 2, 0)
        x_filtered = cv2.GaussianBlur(x, (3, 3), 1)

        # Color space transformation
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
        cnt, idx = self.resultsTopk(img, 1)
        
        hashes = self.getDigest(img)
        self.cache = np.concatenate((self.cache, hashes))
        self.g.add_node(self.input_idx)

        if cnt > self.threshold:
            self.g.add_edge(idx, self.input_idx, label=cnt)
        return cnt, idx


    def resultsTopk(self, img, k):
        hashes = self.getDigest(img)
        hamming_dists = np.count_nonzero(hashes == self.cache, axis=1)
        closest = np.argsort(-hamming_dists)[:k]
        # remove dummy element if present
        closest = closest[closest != 0]
        if len(closest) == 0:
            return 0, 0
        cnt = hamming_dists[closest[0]]
        idx = closest[0]
        return cnt, idx


def load_image(path):
    transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor()])
    img = transform(PIL.Image.open(path).convert("RGB"))
    img = np.asarray(img).astype(np.float32)
    return img

def get_image_list(image_paths):
    all_x = []
    for i in image_paths:
        x = load_image(i)
        all_x.append(x)
    return all_x



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, default='a')
    parser.add_argument('--threshold', type=int, default=174)
    args = parser.parse_args()
    threshold = args.threshold
    # For BlackLight detection
    # if args.data == 'a':
    # ## For generation of anomaly graph
    #     window_size = 20
    #     hash_kept = 50
    #     roundto = 50
    #     step_size = 1
    #     workers = 5
    #     threshold = 25
    #     Image_path = "/home/lsf/BlackBox/ccs_23_oars_stateful_attacks/data/imagenet/anomaly_query/"
    #     anomaly_img = sorted(os.listdir(Image_path))[:1]
    #     cnt = max([int(i) for i in os.listdir('../graph_data/raw/anomaly_graph')[-1].split('.')[0]]) + 1
    #     for i in tqdm(anomaly_img):
    #         path = os.path.join(Image_path, i)
    #         image_paths = [os.path.join(path, i) for i in os.listdir(path) if i.endswith('.png')]
    #         image_paths = sorted(image_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))[:500]
    #         img_list = get_image_list(image_paths)
    #         print(img_list[0].shape)
    #         tracker = CD(img_list[0], window_size, hash_kept, round=roundto, step_size=step_size, workers=workers)
    #         print("Blacklight detector created.")

    #         for k, img in enumerate(img_list):
    #             _, idx = tracker.add_img(img)
    #             if (k+1) % 100 == 0:
    #                 nx.drawing.nx_pydot.write_dot(tracker.g, 'graph_data/raw/anomaly_graph/{}.dot'.format(cnt))
    #                 cnt += 1

    # elif args.data == 'b':
    #     ## For generation of normal graph
    #     window_size = 20
    #     hash_kept = 50
    #     roundto = 50
    #     step_size = 1
    #     workers = 5
    #     threshold = 25
    #     imagenet_path = "/home/lsf/BlackBox/ccs_23_oars_stateful_attacks/data/imagenet/"
    #     data_path = os.path.join(imagenet_path, 'val')
    #     data_path_img = [os.path.join(data_path, i) for i in os.listdir(data_path)]
    #     image_paths = []
    #     for path in data_path_img:
    #         image_paths.extend([os.path.join(path, i) for i in os.listdir(path)])
    #     assert len(image_paths) == 50000
    #     cnt = max([int(i.split('.')[0]) for i in os.listdir('../graph_data/raw/normal_graph')]) + 1
    #     print(cnt)
    #     for _ in tqdm(range(500)):
    #         image_paths = random.sample(image_paths, 500)
    #         img_list = get_image_list(image_paths)
    #         print(img_list[0].shape)
    #         tracker = CD(img_list[0], window_size, hash_kept, round=roundto, step_size=step_size, workers=workers)
    #         print("Blacklight detector created.")
    #         for img in img_list:
    #             _, idx = tracker.add_img(img)

    #         nx.drawing.nx_pydot.write_dot(tracker.g, 'graph_data/raw/normal_graph/{}.dot'.format(cnt))
    #         cnt += 1


    # For PIHA detection
    if args.data == 'a':
    ## For generation of anomaly graph
        input_shape = [3,224,224]
        block_size = 7
        Image_path = "/home/lsf/BlackBox/ccs_23_oars_stateful_attacks/data/imagenet/anomaly_query/"
        anomaly_img = sorted(os.listdir(Image_path))

        dir_list = os.listdir('../graph_data/raw/anomaly_graph')
        cnt = max([int(i.split('.')[0]) for i in dir_list]) + 1 if len(dir_list) != 0 else 0
        for i in tqdm(anomaly_img):
            path = os.path.join(Image_path, i)
            image_paths = [os.path.join(path, i) for i in os.listdir(path) if i.endswith('.png')]
            image_paths = sorted(image_paths, key=lambda x: int(x.split('/')[-1].split('.')[0]))[:2000]
            img_list = get_image_list(image_paths)
            tracker = CD_PIHA(input_shape, block_size, threshold)
            print("PIHA created.")

            for k, img in enumerate(img_list):
                _, idx = tracker.add(torch.from_numpy(img))
                largest = max(nx.weakly_connected_components(tracker.g),key=len)
                largest_connected_subgraph = tracker.g.subgraph(largest)
                if (k+1) % 50 == 0 and largest_connected_subgraph.number_of_nodes() > 20:
                    nx.drawing.nx_pydot.write_dot(tracker.g, '../graph_data/raw/anomaly_graph/{}.dot'.format(cnt))
                    cnt += 1

    elif args.data == 'b':
        ## For generation of normal graph
        input_shape = [3,224,224]
        block_size = 7        
        imagenet_path = "/home/lsf/BlackBox/ccs_23_oars_stateful_attacks/data/imagenet/"
        data_path = os.path.join(imagenet_path, 'val')
        data_path_img = [os.path.join(data_path, i) for i in os.listdir(data_path)]
        image_paths = []
        for path in data_path_img:
            image_paths.extend([os.path.join(path, i) for i in os.listdir(path)])
        assert len(image_paths) == 50000
        dir_list = os.listdir('../graph_data/raw/normal_graph')
        cnt = max([int(i.split('.')[0]) for i in dir_list]) + 1 if len(dir_list) != 0 else 0
        print(cnt)
        for _ in tqdm(range(10)):
            image_paths = random.sample(image_paths, 5000)
            img_list = get_image_list(image_paths)
            tracker = CD_PIHA(input_shape, block_size, threshold)
            print("PIHA detector created.")
            for k, img in enumerate(img_list):
                _, idx = tracker.add(torch.from_numpy(img))

                largest = max(nx.weakly_connected_components(tracker.g),key=len)
                largest_connected_subgraph = tracker.g.subgraph(largest)
                if (k + 1) % 200 == 0 and largest_connected_subgraph.number_of_nodes() > 20:
                    nx.drawing.nx_pydot.write_dot(tracker.g, '../graph_data/raw/normal_graph/{}.dot'.format(cnt))
                    cnt += 1


