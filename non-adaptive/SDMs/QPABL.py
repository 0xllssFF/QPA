import networkx as nx
import hashlib
from collections import Counter
from collections import defaultdict
import sys
import numpy as np
from multiprocessing import Pool
import pickle
import imp
import smirnov_grubbs as grubbs
from GraphCache import *
from scipy.stats import kstest
from torch_geometric.data import Dataset, download_url
from torch_geometric.data import Data
from torch_geometric.utils.convert import from_networkx
from skimage.feature import local_binary_pattern
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, TransformerConv
from torch_geometric.nn import global_mean_pool
from torch.nn import Linear
import torch.nn.functional as F
import torch.nn as nn

import torch
import cv2

class LeNet5(nn.Module):

    def __init__(self, num_classes, grayscale=False):
        super(LeNet5, self).__init__()
        
        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(
            
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        # probas = F.softmax(logis, dim=1)
        return logits

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings 
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x


class QPABL:
    # def __init__(self, query, args):
    #     self.input_shape = args["input_shape"]
    #     self.block_size = args["block_size"]
    #     self.init_hash = self.getDigest(np.zeros(self.input_shape))
    #     self.cache = self.init_hash
    #     self.hash_dict = {0: self.init_hash}
    #     self.cache_idx_map = [0]
    #     self.g = nx.Graph()
    #     self.alerted_nodes = set()
    #     self.input_idx = 0
    #     self.output = {}
    #     self.train_set = len(query)
    #     self.graph_model = torch.load(args["model_path"])
    #     self.graph_model.eval()

    #     for img in query:
    #         self.input_idx += 1
    #         hashes = self.getDigest(img)
    #         self.cache = np.concatenate((self.cache, hashes))
    #         self.hash_dict[self.input_idx] = hashes
    #         self.cache_idx_map.append(self.input_idx)
    #         self.g.add_node(self.input_idx)
        
    #     print(self.cache.shape)
    #     dist_list = []
    #     for i in range(1, self.train_set + 1):
    #         hashes = self.hash_dict[i]
    #         hamming_dists = np.count_nonzero(hashes == self.cache, axis=1)
    #         closest = np.argsort(-hamming_dists)[:2]
    #         # remove dummy element if present
    #         # closest = closest[closest != 0]
    #         if len(closest) <= 1:
    #             dist_list.append(0)
    #             continue
    #         cnt = hamming_dists[closest[1]]
    #         dist_list.append(cnt)
    #     self.threshold = np.percentile(dist_list, 90)
    #     print("Threshold: {}".format(self.threshold))
    #     dist_list.clear()

    # def _piha_hash(self, x):
    #     N = x.shape[2]
    #     # Image preprocessing
    #     x = x.transpose(1, 2, 0)
    #     x_filtered = cv2.GaussianBlur(x, (3, 3), 1)

    #     # Color space transformation
    #     x_filtered = np.float32(x_filtered)
    #     x_hsv = cv2.cvtColor(x_filtered, cv2.COLOR_RGB2HSV)

    #     # Use only H channel for HSV color space
    #     x_h = x_hsv[:, :, 0].reshape((N, N, 1))
    #     x_h = np.pad(x_h,
    #                  ((0, self.block_size - N % self.block_size), (0, self.block_size - N % self.block_size), (0, 0)),
    #                  'constant')
    #     N = x_h.shape[0]

    #     # Block division and feature matrix calculation
    #     blocks_h = [x_h[i:i + self.block_size, j:j + self.block_size] for i in range(0, N, self.block_size) for j in
    #                 range(0, N, self.block_size)]
    #     features_h = np.array([np.sum(block) for block in blocks_h]).reshape(
    #         (N // self.block_size, N // self.block_size))

    #     # Local binary pattern feature extraction
    #     features_lbp = local_binary_pattern(features_h, 8, 1)

    #     # Hash generation
    #     # hash_array = ''.join([f'{(int(_)):x}' for _ in features_lbp.flatten().tolist()])
    #     # hash_array = ''.join([format(int(_), '02x') for _ in features_lbp.flatten().tolist()])
    #     hash_array = features_lbp.flatten()
    #     hash_array = np.expand_dims(hash_array, axis=0)
    #     return hash_array

    # def getDigest(self, img):
    #     h = self._piha_hash(img)
    #     return h

    # def resetCache(self):
    #     self.cache = self.getDigest(torch.zeros(tuple(self.input_shape)))

    # def add(self, img):
    #     self.input_idx += 1
    #     cnt, idx = self.resultsTopk(img, 1)
        
    #     hashes = self.getDigest(img)
    #     self.cache = np.concatenate((self.cache, hashes))
    #     self.hash_dict[self.input_idx] = hashes
    #     self.cache_idx_map.append(self.input_idx)
    #     self.g.add_node(self.input_idx)

    #     res = False
    #     if cnt > self.threshold:
    #         self.g.add_edge(idx, self.input_idx, label=cnt)
    #         if idx in self.alerted_nodes:
    #             res = True
    #             self.alerted_nodes.add(self.input_idx)
    #     return cnt, idx ,res


    # def resultsTopk(self, img, k):
    #     hashes = self.getDigest(img)
    #     hamming_dists = np.count_nonzero(hashes == self.cache, axis=1)
    #     closest = np.argsort(-hamming_dists)[:k]
    #     # remove dummy element if present
    #     closest = closest[np.array(self.cache_idx_map)[closest] != 0]
    #     if len(closest) == 0:
    #         return 0, 0
    #     cnt = hamming_dists[closest[0]]
    #     idx = self.cache_idx_map[closest[0]]
    #     return cnt, idx

    def __init__(self, query, args, ttd):
        self.window_size = args["window_size"]
        self.num_hashes_keep = args["num_hashes_keep"]
        self.round = args["round"]
        self.step_size = args["step_size"]
        if(args["salt"] != None):
            self.salt = args["salt"]
        else:
            self.salt = np.random.rand(args["input_shape"]) * 255.

        self.hash_dict = {}
        self.pool = Pool(processes=5)
        self.input_idx = 0

        self.g = nx.Graph()
        self.alerted_nodes = set()
        self.threshold = 0
        self.graph_model = torch.load(args["model_path"])
        self.graph_model.eval()
        self.ttd = ttd
        self.train_set = len(query)
        hash_cache = defaultdict(list)

        for img in query:
            hashes = self.getDigest(img)
            hash_cache[self.input_idx] = hashes
            for el in hashes:
                if el not in self.hash_dict:
                    self.hash_dict[el] = [self.input_idx]
                else:
                    self.hash_dict[el].append(self.input_idx)
            self.g.add_node(self.input_idx)
            self.input_idx += 1
        dist_list = []
        def tmp_check_img(hashes):
            sets = list(map(self.hash_dict.get, hashes))
            sets = [i for i in sets if i is not None]
            sets = [item for sublist in sets for item in sublist]
            if not sets or len(sets) == 0:
                return 0, 0
            sets = Counter(sets)
            most_common = sets.most_common(2)
            if len(most_common) == 1:
                return 0, 0
            cnt = sets.most_common(2)[1][1]
            idx = sets.most_common(2)[1][0]
            return idx, cnt
        
        for i in range(self.train_set):
            hashes = hash_cache[i]
            idx, cnt = tmp_check_img(hashes)
            dist_list.append(cnt)
        self.threshold = np.percentile(dist_list, 90)
        print("Threshold: {}".format(self.threshold))
        hash_cache.clear()
        dist_list.clear()

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
        hash_list = self.pool.map(QPABL.hash_helper, idx_ls)
        hash_list = list(set(hash_list))
        hash_list = [r[::-1] for r in hash_list]
        hash_list.sort(reverse=True)
        return hash_list[:self.num_hashes_keep]

    def add(self, img):
        cnt, idx, hashes = self.resultsTopk(img, 1)

        for el in hashes:
            if el not in self.hash_dict:
                self.hash_dict[el] = [self.input_idx]
            else:
                self.hash_dict[el].append(self.input_idx)

        self.g.add_node(self.input_idx)
        res = False
        if cnt > self.threshold:
            self.g.add_edge(idx, self.input_idx, label=cnt)
            if idx in self.alerted_nodes:
                res = True
                self.alerted_nodes.add(self.input_idx)
        self.input_idx += 1
        return cnt, idx ,res


    def resultsTopk(self, img, k):
        hashes = self.getDigest(img)
        sets = list(map(self.hash_dict.get, hashes))
        sets = [i for i in sets if i is not None]
        sets = [item for sublist in sets for item in sublist if item in self.g.nodes()]
        if not sets or len(sets) == 0:
            return 0, 0, hashes
        sets = Counter(sets)
        x = sets.most_common(k)[0]
        return x[1], x[0], hashes


    def graph_checker(self, graph):
        for edge in graph.edges:
            graph.edges[edge]['label'] = float(graph.edges[edge]['label'])
        converted_largest_connected_subgraph = nx.line_graph(graph)
        edge_attr = nx.get_edge_attributes(graph, 'label')
        for node in converted_largest_connected_subgraph.nodes:
            converted_largest_connected_subgraph.nodes[node]['label'] = edge_attr[node]
        pyg_graph = from_networkx(converted_largest_connected_subgraph, group_node_attrs = all)
        batch = torch.zeros(pyg_graph.num_nodes, dtype=torch.long)
        out = self.graph_model(pyg_graph.x, pyg_graph.edge_index, batch)
        pred = out.argmax(dim=1)
        # print(pred)
        return pred == 1


    def detector(self, topK = 100):
        self.alerted_nodes = set()
        subgraph_list = [CacheGraph(self.g.subgraph(s).copy()) for s in nx.connected_components(self.g)]
        score = [s.GetGraphScore() for s in subgraph_list]
        cov = grubbs.max_test_outliers(score, alpha=0.01)
        if len(cov) != 0:
            flag = True
            while flag:
                outliers = grubbs.max_test_outliers(cov, alpha=0.01)
                if len(outliers) == 0:
                    break
                cov = outliers
            indices = [x for x in subgraph_list if x.GetGraphScore() in cov]
            for i in indices:
                if i.node_nums > self.ttd:
                    if self.graph_checker(i.graph) :
                        # print('[Alert]: ', i.GetGraphScore(), i.node_nums, i.edge_nums)
                        # print('[Graph]: ', i.graph.nodes(), i.graph.edges())
                        self.alerted_nodes |= set(i.graph.nodes())
                    else:
                        subgraph_list.remove(i)

        # subgraph_list.sort(key = lambda x: x.GetGraphScore(), reverse = True)
        # self.g = nx.compose_all([x.graph for x in subgraph_list[:topK]])
        # self.g.add_nodes_from([i for i in range(0, self.train_set)])
        