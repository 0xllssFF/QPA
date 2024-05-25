import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision
from models.resnet import resnet152, resnet20
from models.iot_sqa import IOTSQAClassifier, IOTSQAEncoder
from abc import abstractmethod
from multiprocessing import Pool
import hashlib
from collections import Counter
from skimage.feature import local_binary_pattern
import cv2
from models.GraphCache import *
import models.smirnov_grubbs as grubbs
import json
from PIL import Image
import os
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import global_mean_pool
from collections import defaultdict



class StateModule:
    @abstractmethod
    def getDigest(self, img):
        """
        Returns a digest of the image
        :param img:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def resultsTopk(self, img, k):
        """
        Return a list of top k tuples (distance, prediction) - smallest distance first
        :param img:
        :param k:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def resetCache(self):
        """
        Reset the cache
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def add(self, img, prediction):
        """
        Add an image to the cache
        :param img:
        :return:
        """
        raise NotImplementedError


class BlackLight(StateModule):
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

        self.cache = {}
        self.inverse_cache = {}
        self.input_idx = 0
        self.pool = Pool(processes=arguments["num_processes"])

        # self.dataset = 'imagenet'
        # img_num = [int(i) for i in os.listdir('/home/lsf/BlackBox/ccs_23_oars_stateful_attacks/data/imagenet/anomaly_query')]
        # self.img_id = max(img_num) + 1 if len(img_num) != 0 else 0
        # print(self.img_id)
        # img_dir = '/home/lsf/BlackBox/ccs_23_oars_stateful_attacks/data/imagenet/anomaly_query/{}'\
        #     .format(self.img_id)
        # if not os.path.exists(img_dir):
        #     os.makedirs(img_dir)



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
            array = np.array(array.cpu()) * 255.
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
        self.cache = {}
        self.inverse_cache = {}
        self.input_idx = 0
        if self.salt_type:
            self.salt = (np.random.rand(*tuple(self.input_shape)) * 255).astype(np.int16)
        else:
            self.salt = (np.zeros(tuple(self.input_shape)) * 255).astype(np.int16)

        
        # self.img_id += 1
        # img_dir = '/home/lsf/BlackBox/ccs_23_oars_stateful_attacks/data/imagenet/anomaly_query/{}'\
        #     .format(self.img_id)
        # if not os.path.exists(img_dir):
        #     os.makedirs(img_dir)

    def add(self, img, prediction):
        self.input_idx += 1
        hashes = self.getDigest(img)
        for el in hashes:
            if el not in self.inverse_cache:
                self.inverse_cache[el] = [self.input_idx]
            else:
                self.inverse_cache[el].append(self.input_idx)
        self.cache[self.input_idx] = prediction

    def resultsTopk(self, img, k):
        hashes = self.getDigest(img)
        sets = list(map(self.inverse_cache.get, hashes))
        sets = [i for i in sets if i is not None]
        sets = [item for sublist in sets for item in sublist]
        if not sets:
            return []
        sets = Counter(sets)
        result = [((self.num_hashes_keep - x[1]) / self.num_hashes_keep, self.cache[x[0]]) for x in sets.most_common(k)]
        return result


class OSDEncoder(torch.nn.Module):
    def __init__(self):
        super(OSDEncoder, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding='same')
        self.conv2 = torch.nn.Conv2d(32, 32, 3)
        self.drop1 = torch.nn.Dropout2d(0.25)

        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding='same')
        self.conv4 = torch.nn.Conv2d(64, 64, 3)
        self.drop2 = torch.nn.Dropout2d(0.25)

        self.fc1 = torch.nn.Linear(64 * 6 * 6, 512)
        self.drop3 = torch.nn.Dropout2d(0.5)
        self.fc2 = torch.nn.Linear(512, 256)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv2(x)), 2)
        x = self.drop1(x)
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv4(x)), 2)
        x = self.drop2(x)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, 64 * 6 * 6)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)
        return x


class OriginalStatefulDetector(StateModule):
    def __init__(self, arguments):
        self.encoder = OSDEncoder().cuda()
        checkpoint = torch.load(arguments["encoder_path"])
        self.encoder.load_state_dict(checkpoint)
        self.encoder.eval()
        self.input_shape = arguments["input_shape"]
        if arguments["salt"] is not None:
            self.salt = arguments["salt"]
        else:
            self.salt = np.zeros(self.input_shape).astype(np.int16)

        self.cache = {}

    def getDigest(self, img):
        if len(img.shape) != 3:
            raise Exception("expected 3d image")
        return self.encoder(img.to(next(self.encoder.parameters()).device).detach().unsqueeze(0)).squeeze(0)

    def resetCache(self):
        self.cache = {}

    def add(self, img, prediction):
        if len(img.shape) != 3:
            raise Exception("expected 3d image")
        encoding = self.getDigest(img)
        self.cache[encoding] = prediction

    def resultsTopk(self, img, k):
        # print(torch.min(img), torch.max(img))
        img = torch.clamp(img, 0, 1)
        embed = self.getDigest(img)
        dists = []
        preds = []
        for query_embed, pred in self.cache.items():
            dist = torch.linalg.norm(embed - query_embed).item()
            dists.append(dist)
            preds.append(pred)
        top_dists = np.argsort(dists)[:k]
        # top_dists = np.argpartition(dists, k - 1)
        result = [(dists[i], preds[i]) for i in top_dists]
        return result


class IOTSQA(StateModule):
    def __init__(self, arguments):
        self.encoder = IOTSQAEncoder()
        # self.encoder.load_weights("models/pretrained/iot_sqa_encoder.h5")
        self.encoder.eval()
        self.encoder = self.encoder.cuda()

        self.cache = {}

    def getDigest(self, img):
        if len(img.shape) != 3:
            raise Exception("expected 3d image")
        return self.encoder(img.detach().unsqueeze(0)).squeeze(0)

    def resetCache(self):
        self.cache = {}

    def add(self, img, prediction):
        if len(img.shape) != 3:
            raise Exception("expected 3d image")
        encoding = self.getDigest(img)
        self.cache[encoding] = prediction

    def resultsTopk(self, img, k):
        # print(torch.min(img), torch.max(img))
        img = torch.clamp(img, 0, 1)
        embed = self.getDigest(img)
        dists = []
        preds = []
        for query_embed, pred in self.cache.items():
            dist = torch.linalg.norm(embed - query_embed).item()
            dists.append(dist)
            preds.append(pred)
        top_dists = np.argsort(dists)[:k]
        # top_dists = np.argpartition(dists, k - 1)
        result = [(dists[i], preds[i]) for i in top_dists]
        return result


class PIHA(StateModule):
    def __init__(self, arguments):
        super(PIHA, self).__init__()
        self.input_shape = arguments["input_shape"]
        self.block_size = arguments["block_size"]
        self.cache_predictions = {}
        self.input_idx = 0
        self.cache = self.getDigest(torch.zeros(arguments["input_shape"]))

    def _piha_hash(self, x):
        N = x.shape[2]
        # Image preprocessing
        x = x.cpu().numpy().transpose(1, 2, 0)
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

    def add(self, img, prediction):
        self.input_idx += 1
        hash = self.getDigest(img)
        self.cache = np.concatenate((self.cache, hash))
        self.cache_predictions[self.input_idx] = prediction

    def resultsTopk(self, img, k):
        hash = self.getDigest(img)
        hamming_dists = np.count_nonzero(hash != self.cache, axis=1) / self.cache.shape[1]
        closest = np.argsort(hamming_dists)[:k]
        # remove dummy element if present
        closest = closest[closest != 0]
        if len(closest) == 0:
            return []
        result = [(hamming_dists[i], self.cache_predictions[i]) for i in closest]
        return result


class NoOpState(StateModule):
    def __init__(self, arguments, attack, targeted):
        # self.attack_method = attack
        # self.targeted = 'targeted' if targeted else 'untargeted'
        # img_num = [int(i) for i in os.listdir('/home/lsf/BlackBox/ccs_23_oars_stateful_attacks/data/celebahq/attack_query/{}/{}'.format(self.attack_method, self.targeted))]
        # self.img_id = max(img_num) if len(img_num) != 0 else 0
        # print(self.img_id)
        # img_dir = '/home/lsf/BlackBox/ccs_23_oars_stateful_attacks/data/celebahq/attack_query/{}/{}/{}'\
        #     .format(self.attack_method, self.targeted, self.img_id)
        # if not os.path.exists(img_dir):
        #     os.makedirs(img_dir)
        pass

    def getDigest(self, img):
        return img

    def resetCache(self):
        # self.img_id += 1
        # img_dir = '/home/lsf/BlackBox/ccs_23_oars_stateful_attacks/data/celebahq/attack_query/{}/{}/{}'\
        #     .format(self.attack_method, self.targeted, self.img_id)
        # if not os.path.exists(img_dir):
        #     os.makedirs(img_dir)
        pass

    def add(self, img, prediction):
        pass

    def resultsTopk(self, img, k):
        return []

class ToolName(StateModule):
    ### PIHA version
    def __init__(self, arguments):
        self.input_shape = arguments["input_shape"]
        self.block_size = arguments["block_size"]
        self.model_path = arguments["model_path"]
        self.cache_predictions = {}
        self.init_hash = self.getDigest(torch.zeros(self.input_shape))
        self.cache = self.init_hash
        self.hash_dict = {0: self.init_hash}
        self.cache_idx_map = [0]
        self.g = nx.Graph()
        self.alerted_nodes = set()
        self.input_idx = 0
        self.threshold = None
        self.train_set = None
        self.graph_model = None
        self.len_train_set = None
        self.f_model = None
        self.store_graph = []

    def init_graph(self, model, query):
        self.train_set = query
        self.len_train_set = len(query)
        self.f_model = model
        self.graph_model = torch.load(self.model_path)
        self.graph_model.eval()

        for img in query:
            self.input_idx += 1
            hashes = self.getDigest(img)
            self.cache = np.concatenate((self.cache, hashes))
            self.hash_dict[self.input_idx] = hashes
            self.cache_idx_map.append(self.input_idx)
            self.g.add_node(self.input_idx)
            self.cache_predictions[self.input_idx] = model.model(img.unsqueeze(0)).detach()

        
        print(self.cache.shape)
        dist_list = []
        for i in range(1, self.len_train_set + 1):
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
        # x = x.numpy()
        N = x.shape[2]
        # Image preprocessing
        x = x.cpu().numpy().transpose(1, 2, 0)
        x_filtered = cv2.GaussianBlur(x, (3, 3), 1)

        # Color space transformation
        # x_filtered = np.float32(x_filtered)
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
        self.cache_predictions = {}
        self.input_idx = 0
        self.init_hash = self.getDigest(torch.zeros(self.input_shape))
        self.cache = self.init_hash
        self.hash_dict = {0: self.init_hash}
        self.cache_idx_map = [0]
        self.g = nx.Graph()
        self.alerted_nodes = set()
        self.init_graph(self.f_model, self.train_set)

    def add(self, img, prediction, cnt, idx):
        self.input_idx += 1
        # cnt, idx = self.resultsTopk(img, 1)
        
        hashes = self.getDigest(img)
        self.cache = np.concatenate((self.cache, hashes))
        self.hash_dict[self.input_idx] = hashes
        self.cache_idx_map.append(self.input_idx)
        self.g.add_node(self.input_idx)
        self.cache_predictions[self.input_idx] = prediction
        res = False
        if cnt is not None and cnt > self.threshold:
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
        closest = closest[closest != 0]
        if len(closest) == 0:
            return []
        # cnt = hamming_dists[closest[0]]
        # idx = self.cache_idx_map[closest[0]]
        result = [(hamming_dists[i], self.cache_predictions[self.cache_idx_map[i]], self.cache_idx_map[i]) for i in closest]
        # print(result)
        return result

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
        print(pred)
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
                if i.node_nums > 10:
                    if self.graph_checker(i.graph):
                        self.alerted_nodes |= set(i.graph.nodes())
                    # else:
                        # flag_store = True
                        # for s_g in self.store_graph:
                        #     if nx.utils.graphs_equal(s_g, i.graph):
                        #         flag_store = False
                        # if flag_store:
                        #     dir_list = os.listdir('/home/lsf/BlackBox/ccs_23_oars_stateful_attacks/graph_data/raw/anomaly_graph')
                        #     cnt = max([int(i.split('.')[0]) for i in dir_list]) + 1 if len(dir_list) != 0 else 0
                        #     nx.drawing.nx_pydot.write_dot(i.graph, '/home/lsf/BlackBox/ccs_23_oars_stateful_attacks/graph_data/raw/anomaly_graph/{}.dot'.format(cnt))
                        #     self.store_graph.append(i.graph)
                        # subgraph_list.remove(i)
        subgraph_list.sort(key = lambda x: x.GetGraphScore(),reverse = True)
        self.g = nx.compose_all([x.graph for x in subgraph_list[:topK]])
        self.g.add_nodes_from([i for i in range(1, self.len_train_set + 1)])
################# PIHA Version

##### BL Version
    # def __init__(self, args):
    #     self.window_size = args["window_size"]
    #     self.num_hashes_keep = args["num_hashes_keep"]
    #     self.round = args["round"]
    #     self.step_size = args["step_size"]
    #     if(args["salt"] != None):
    #         self.salt = args["salt"]
    #     else:
    #         self.salt = np.random.rand(args["input_shape"]) * 255.

    #     self.hash_dict = {}
    #     self.pool = Pool(processes=5)
    #     self.input_idx = 0

    #     self.g = nx.Graph()
    #     self.alerted_nodes = set()
    #     self.threshold = 0
    #     self.model_path = args["model_path"]
    #     self.graph_model = None
    #     self.train_set = None
    #     self.len_train_set = None
    #     self.f_model = None
    #     self.store_graph = []
    #     self.cache_predictions = {}



    # def init_graph(self, model, query):
    #     self.train_set = query
    #     self.len_train_set = len(query)
    #     self.f_model = model
    #     self.graph_model = torch.load(self.model_path)
    #     self.graph_model.eval()
    #     hash_cache = defaultdict(list)

    #     for img in query:
    #         hashes = self.getDigest(img)
    #         hash_cache[self.input_idx] = hashes
    #         self.cache_predictions[self.input_idx] = model.model(img.unsqueeze(0)).detach()
    #         for el in hashes:
    #             if el not in self.hash_dict:
    #                 self.hash_dict[el] = [self.input_idx]
    #             else:
    #                 self.hash_dict[el].append(self.input_idx)
    #         self.g.add_node(self.input_idx)
    #         self.input_idx += 1
    #     dist_list = []
        
    #     def tmp_check_img(hashes):
    #         sets = list(map(self.hash_dict.get, hashes))
    #         sets = [i for i in sets if i is not None]
    #         sets = [item for sublist in sets for item in sublist]
    #         if not sets or len(sets) == 0:
    #             return 0, 0
    #         sets = Counter(sets)
    #         most_common = sets.most_common(2)
    #         if len(most_common) == 1:
    #             return 0, 0
    #         cnt = sets.most_common(2)[1][1]
    #         idx = sets.most_common(2)[1][0]
    #         return idx, cnt
        
    #     for i in range(self.len_train_set):
    #         hashes = hash_cache[i]
    #         idx, cnt = tmp_check_img(hashes)
    #         dist_list.append(cnt)
    #     # self.threshold = np.percentile(dist_list, 90)
    #     self.threshold = 11

    #     print("Threshold: {}".format(self.threshold))
    #     hash_cache.clear()
    #     dist_list.clear()

    # def resetCache(self):
    #     self.cache_predictions = {}
    #     self.hash_dict = {}
    #     self.input_idx = 0
    #     self.g = nx.Graph()
    #     self.alerted_nodes = set()
    #     self.init_graph(self.f_model, self.train_set)

    # @staticmethod
    # def hash_helper(arguments):
    #     img = arguments['img']
    #     idx = arguments['idx']
    #     window_size = arguments['window_size']
    #     return hashlib.sha256(img[idx:idx + window_size]).hexdigest()

    # def preprocess(self, array, salt, round=1, normalized=True):
    #     if len(array.shape) != 3:
    #         raise Exception("expected 3d image")
    #     if (normalized):
    #         # input image normalized to [0,1]
    #         array = np.array(array) * 255.
    #     array = (array + salt) % 255.
    #     array = array.reshape(-1)

    #     array = np.around(array / round, decimals=0) * round
    #     array = array.astype(np.int16)
    #     return array

    # def getDigest(self, img):
    #     img = self.preprocess(img, self.salt, self.round)
    #     total_len = int(len(img))
    #     idx_ls = []

    #     for el in range(int((total_len - self.window_size + 1) / self.step_size)):
    #         idx_ls.append({"idx": el * self.step_size, "img": img, "window_size": self.window_size})
    #     hash_list = self.pool.map(ToolName.hash_helper, idx_ls)
    #     hash_list = list(set(hash_list))
    #     hash_list = [r[::-1] for r in hash_list]
    #     hash_list.sort(reverse=True)
    #     return hash_list[:self.num_hashes_keep]

    # def add(self, img, prediction, cnt, idx):
    #     # cnt, idx, hashes = self.resultsTopk(img, 1)
    #     hashes = self.getDigest(img)
    #     self.cache_predictions[self.input_idx] = prediction
    #     for el in hashes:
    #         if el not in self.hash_dict:
    #             self.hash_dict[el] = [self.input_idx]
    #         else:
    #             self.hash_dict[el].append(self.input_idx)

    #     self.g.add_node(self.input_idx)

    #     res = False
    #     if cnt is not None and cnt > self.threshold:
    #         self.g.add_edge(idx, self.input_idx, label=cnt)
    #         if idx in self.alerted_nodes:
    #             res = True
    #             self.alerted_nodes.add(self.input_idx)
    #     self.input_idx += 1
    #     return cnt, idx ,res


    # def resultsTopk(self, img, k):
    #     hashes = self.getDigest(img)
    #     sets = list(map(self.hash_dict.get, hashes))
    #     sets = [i for i in sets if i is not None]
    #     sets = [item for sublist in sets for item in sublist]
    #     if not sets or len(sets) == 0:
    #         return []
    #     sets = Counter(sets)
    #     x = sets.most_common(k)[0]
    #     result = [(x[1], self.cache_predictions[x[0]], x[0])]
    #     return result


    # def graph_checker(self, graph):
    #     for edge in graph.edges:
    #         graph.edges[edge]['label'] = float(graph.edges[edge]['label'])
    #     converted_largest_connected_subgraph = nx.line_graph(graph)
    #     edge_attr = nx.get_edge_attributes(graph, 'label')
    #     for node in converted_largest_connected_subgraph.nodes:
    #         converted_largest_connected_subgraph.nodes[node]['label'] = edge_attr[node]
    #     pyg_graph = from_networkx(converted_largest_connected_subgraph, group_node_attrs = all)
    #     batch = torch.zeros(pyg_graph.num_nodes, dtype=torch.long)
    #     out = self.graph_model(pyg_graph.x, pyg_graph.edge_index, batch)
    #     pred = out.argmax(dim=1)
    #     # print(pred)
    #     return pred == 1


    # def detector(self, topK = 100):
    #     # print('start detector')
    #     self.alerted_nodes = set()
    #     subgraph_list = [CacheGraph(self.g.subgraph(s).copy()) for s in nx.connected_components(self.g)]
    #     score = [s.GetGraphScore() for s in subgraph_list]
    #     cov = grubbs.max_test_outliers(score, alpha=0.01)
    #     if len(cov) != 0:
    #         flag = True
    #         while flag:
    #             outliers = grubbs.max_test_outliers(cov, alpha=0.01)
    #             if len(outliers) == 0:
    #                 break
    #             cov = outliers
    #         indices = [x for x in subgraph_list if x.GetGraphScore() in cov]
    #         for i in indices:
    #             if i.node_nums > 10:
    #                 if self.graph_checker(i.graph) :
    #                     # print('[Alert]: ', i.GetGraphScore(), i.node_nums, i.edge_nums)
    #                     # print('[Graph]: ', i.graph.nodes(), i.graph.edges())
    #                     self.alerted_nodes |= set(i.graph.nodes())
    #                 else:
    #                     subgraph_list.remove(i)

    #     subgraph_list.sort(key = lambda x: x.GetGraphScore(), reverse = True)
    #     self.g = nx.compose_all([x.graph for x in subgraph_list[:topK]])
    #     self.g.add_nodes_from([i for i in range(0, self.len_train_set)])
        
class StatefulClassifier(torch.nn.Module):

    def __init__(self, model, state_module, hyperparameters):
        super().__init__()
        self.config = hyperparameters
        self.model = model
        self.state_module = state_module
        self.threshold = hyperparameters["threshold"]
        self.aggregation = hyperparameters["aggregation"]
        self.add_cache_hit = hyperparameters["add_cache_hit"]
        self.reset_cache_on_hit = hyperparameters["reset_cache_on_hit"]
        self.dataset = hyperparameters['dataset']
        self.cache_hits = 0
        self.total = 0
        self.distances = []
        self.sequence_metric = []

    def reset(self):
        self.state_module.resetCache()
        self.cache_hits = 0
        self.total = 0
        self.distances = []

    def forward_single(self, x):
        self.total += 1
        # sample = x.cpu().clone()
        # T = torchvision.transforms.ToPILImage()
        # sample = T(sample)
        # img_path = '/home/lsf/BlackBox/ccs_23_oars_stateful_attacks/data/{}/attack_query/{}/{}/{}/{}.png'\
        #     .format(self.dataset, self.state_module.attack_method, self.state_module.targeted ,self.state_module.img_id, self.total)
        # # img_path = '/home/lsf/BlackBox/ccs_23_oars_stateful_attacks/data/mnist/tmp_test/{}.png'.format(self.total)
        # sample.save(img_path)
        t_cached_prediction = None
        similar = False
        if self.aggregation == 'closest':
            similarity_result = self.state_module.resultsTopk(x, 1)
            if len(similarity_result) > 0:
                dist, t_cached_prediction = similarity_result[0]
                self.distances.append(dist)
                if dist <= self.threshold:
                    if self.add_cache_hit:
                        self.state_module.add(x, t_cached_prediction)
                    self.cache_hits += 1
                    if self.reset_cache_on_hit:
                        self.state_module.resetCache()
                    similar = True

        elif self.aggregation == 'average':
            similarity_result = self.state_module.resultsTopk(x, self.config['num_to_average'])
            if len(similarity_result) >= self.config['num_to_average']:
                dist, t_cached_prediction = similarity_result[0]
                dists = [dist for (dist, _) in similarity_result]
                if np.mean(dists) <= self.threshold:
                    if self.add_cache_hit:
                        self.state_module.add(x, t_cached_prediction)
                    self.cache_hits += 1
                    if self.reset_cache_on_hit:
                        self.state_module.resetCache()
                    similar = True

        elif self.aggregation == 'graph':
            if self.total % 50 == 0:
                self.state_module.detector()
            similarity_result = self.state_module.resultsTopk(x, 1)
            ### To label the image is a single node or not
            is_single = True
            if len(similarity_result) > 0:
                t_dist, t_cached_prediction, t_closest_id = similarity_result[0]
                # print(t_dist, t_closest_id)
                graph = self.state_module.g
                subgraph_list = [CacheGraph(graph.subgraph(s).copy()) for s in nx.connected_components(graph)]
                score = [s.GetGraphScore() for s in subgraph_list]
                self.sequence_metric.append(max(score))
                self.distances.append(t_dist)
                is_cached = False
                if t_dist > self.state_module.threshold:
                    is_single = False
                    if t_closest_id in self.state_module.alerted_nodes:
                        is_cached = True
                if is_cached:
                    self.cache_hits += 1
                    self.state_module.add(x, t_cached_prediction, t_dist, t_closest_id)
                    similar = True
            if is_single:
                dist = None
                closest_id = None
            else:
                dist = t_dist
                closest_id = t_closest_id

        # x_prediction = self.model(x.to(next(self.model.parameters()).device).unsqueeze(0)).detach()
        x_prediction = self.model(x.unsqueeze(0)).detach()
        if similar:
            if self.config["action"] != 'rejection_silent':
                # cached_prediction = -1 * torch.ones_like(cached_prediction)
                # return self.state_module.cache_predictions[1].cuda(), True
                return t_cached_prediction, True
            return x_prediction, False
        
        if self.aggregation != 'graph':
            self.state_module.add(x, x_prediction.detach().cpu())
        else:
            self.state_module.add(x, x_prediction.detach().cpu(), dist, closest_id)
        return x_prediction, False
        # cached_prediction = None
        # similar = False
        # if self.aggregation == 'closest':
        #     similarity_result = self.state_module.resultsTopk(x, 1)
        #     if len(similarity_result) > 0:
        #         dist, cached_prediction = similarity_result[0]
        #         print(dist)
        #         self.distances.append(dist)
        #         if dist <= self.threshold:
        #             if self.add_cache_hit:
        #                 self.state_module.add(x, cached_prediction)
        #             self.cache_hits += 1
        #             if self.reset_cache_on_hit:
        #                 self.state_module.resetCache()
        #             similar = True

        # elif self.aggregation == 'average':
        #     similarity_result = self.state_module.resultsTopk(x, self.config['num_to_average'])
        #     if len(similarity_result) >= self.config['num_to_average']:
        #         dist, cached_prediction = similarity_result[0]
        #         dists = [dist for (dist, _) in similarity_result]
        #         if np.mean(dists) <= self.threshold:
        #             if self.add_cache_hit:
        #                 self.state_module.add(x, cached_prediction)
        #             self.cache_hits += 1
        #             if self.reset_cache_on_hit:
        #                 self.state_module.resetCache()
        #             similar = True

        # if similar:
        #     if self.config["action"] != 'rejection_silent':
        #         # cached_prediction = -1 * torch.ones_like(cached_prediction)
        #         return cached_prediction.cuda(), True

        #     prediction = self.model(x.to(next(self.model.parameters()).device).unsqueeze(0)).detach()
        #     return prediction, False

        # prediction = self.model(x.to(next(self.model.parameters()).device).unsqueeze(0)).detach()
        # self.state_module.add(x, prediction.detach().cpu())
        # return prediction, False

    def forward_batch(self, x):
        batch_size = x.shape[0]
        logits, is_cache = [], []
        for i in range(batch_size):
            pred, is_cached = self.forward_single(x[i])
            logits.append(pred)
            is_cache.append(is_cached)
        logits = torch.cat(logits, dim=0)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return probs, is_cache

    def forward(self, x):
        if len(x.shape) == 3:
            return self.forward_single(x)
        else:
            return self.forward_batch(x)


def init_stateful_classifier(config, attack, targeted):
    if config['architecture'] == 'resnet20':
        model = torch.load("models/pretrained/resnet20-12fca82f-single.pth", map_location="cpu")
        model.eval()
    elif config['architecture'] == 'resnet152':
        model = resnet152()
        model.eval()
    elif config['architecture'] == 'iot_sqa':
        model = IOTSQAClassifier()
        # model.load_weights("models/pretrained/iot_sqa_classifier.h5")
        model.eval()
    elif config['architecture'] == 'celebahq':
        class CelebAHQClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torchvision.models.resnet18(pretrained=True)
                num_features = self.model.fc.in_features
                self.model.fc = torch.nn.Linear(num_features, 307)
                self.model.load_state_dict(torch.load(
                    "models/pretrained/facial_identity_classification_transfer_learning_with_ResNet18_resolution_256.pth"))
                self.xform = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            def forward(self, x):
                x = self.xform(x)
                return self.model(x)

        model = CelebAHQClassifier()
        model.eval()
    elif config['architecture'] == 'lenet5':
        model = torch.load("models/pretrained/mnist-classifier.pth", map_location="cpu")
        model.eval()
    else:
        raise NotImplementedError("Architecture not supported.")

    if config["state"]["type"] == "blacklight":
        state_module = BlackLight(config["state"])
    elif config["state"]["type"] == "PIHA":
        state_module = PIHA(config["state"])
    elif config["state"]["type"] == "OSD":
        state_module = OriginalStatefulDetector(config["state"])
    elif config["state"]["type"] == "iot_sqa":
        state_module = IOTSQA(config["state"])
    elif config["state"]["type"] == "no_op":
        state_module = NoOpState(config["state"], attack, targeted)
    elif config["state"]["type"] == "toolname":
        state_module = ToolName(config["state"])
    else:
        raise NotImplementedError("State module not supported.")

    return StatefulClassifier(model, state_module, config)
