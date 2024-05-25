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

import torch
import cv2

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

class QPA:
    def __init__(self, query, args, ttd):
        self.input_shape = args["input_shape"]
        self.block_size = args["block_size"]
        self.init_hash = self.getDigest(np.zeros(self.input_shape))
        self.cache = self.init_hash
        self.hash_dict = {0: self.init_hash}
        self.cache_idx_map = [0]
        self.g = nx.Graph()
        self.alerted_nodes = set()
        self.input_idx = 0
        self.output = {}
        self.train_set = len(query)
        self.graph_model = torch.load(args["model_path"])
        self.graph_model.eval()
        self.ttd = ttd

        for img in query:
            self.input_idx += 1
            hashes = self.getDigest(img)
            self.cache = np.concatenate((self.cache, hashes))
            self.hash_dict[self.input_idx] = hashes
            self.cache_idx_map.append(self.input_idx)
            self.g.add_node(self.input_idx)
        
        print(self.cache.shape)
        dist_list = []
        for i in range(1, self.train_set + 1):
            hashes = self.hash_dict[i]
            hamming_dists = np.count_nonzero(hashes == self.cache, axis=1)
            closest = np.argsort(-hamming_dists)[:2]
            # remove dummy element if present
            # closest = closest[closest != 0]
            if len(closest) <= 1:
                dist_list.append(0)
                continue
            cnt = hamming_dists[closest[1]]
            dist_list.append(cnt)
        self.threshold = np.percentile(dist_list, 90)
        print("Threshold: {}".format(self.threshold))
        dist_list.clear()

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
        cnt, idx = self.resultsTopk(img, 1)
        
        hashes = self.getDigest(img)
        self.cache = np.concatenate((self.cache, hashes))
        self.hash_dict[self.input_idx] = hashes
        self.cache_idx_map.append(self.input_idx)
        self.g.add_node(self.input_idx)

        res = False
        if cnt > self.threshold:
            self.g.add_edge(idx, self.input_idx, label=cnt)
            if idx in self.alerted_nodes:
                res = True
                self.alerted_nodes.add(self.input_idx)
        return cnt, idx ,res


    def resultsTopk(self, img, k):
        hashes = self.getDigest(img)
        hamming_dists = np.count_nonzero(hashes == self.cache, axis=1)
        closest = np.argsort(-hamming_dists)[:k]
        # remove dummy element if present
        closest = closest[np.array(self.cache_idx_map)[closest] != 0]
        if len(closest) == 0:
            return 0, 0
        cnt = hamming_dists[closest[0]]
        idx = self.cache_idx_map[closest[0]]
        return cnt, idx

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
        return pred == 1

    def detector(self, topK = 100):
        print("==============")
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
                        self.alerted_nodes |= set(i.graph.nodes())
                    else:
                        subgraph_list.remove(i)

        ### performance opt
        # subgraph_list.sort(key = lambda x: x.GetGraphScore(),reverse = True)
        # self.g = nx.compose_all([x.graph for x in subgraph_list[:topK]])
        # self.g.add_nodes_from([i for i in range(1, self.train_set + 1)])
        # # if self.cache.shape[0] > 10000:
        # self.cache = self.init_hash
        # self.cache_idx_map = [0]
        # for i, idx in enumerate(self.g.nodes):
        #     self.cache = np.concatenate((self.cache, self.hash_dict[idx]))
        #     self.cache_idx_map.append(idx)
            

        # in_cache_nodes = list(self.g.nodes)
        # if self.cache.shape[0] > 10000:
            # self.cache = self.cache[in_cache_nodes]
        # sorted_edges = sorted(self.g.edges(), key=lambda x: x[2]['label'], reverse=True)

        