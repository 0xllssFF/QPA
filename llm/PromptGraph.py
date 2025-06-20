import networkx as nx
from sentence_transformers import SentenceTransformer
from simhash import Simhash, SimhashIndex
import numpy as np
import re
from CacheGraph import CacheGraph
import smirnov_grubbs as grubbs
import cProfile
import pstats
import io

class PromptNode:
    def __init__(self, prompt, label):
        self.prompt = prompt
        self.label = label
    
    def get_prompt(self):
        return self.prompt
    
    def get_label(self):
        return self.label


def get_bucket_id(x: float, num_bins: int) -> int:

    x_clamped = max(0.0, min(x, 1.0))
    
    if x_clamped == 1.0:
        return num_bins - 1
    
    return int(x_clamped * num_bins)

class PromptGraph:
    def __init__(self, num_bins = 3, f = 128):
        self.g = nx.Graph()
        self.input_idx = 0
        self.num_bins = num_bins
        self.keywords = {'prompt', 'attack'}
        self.simhash_map = {}
        self.f = f
        self.index = SimhashIndex([(str(0), Simhash(self.get_feature("root"), self.f))], k = 8, f = self.f)
        self.simhash_map[str(0)] = Simhash(self.get_feature("root"), f = self.f)
        self.alerted_nodes = set()
        self.ttd = 6

    def _process_numeric_features(self, num_dict):
        features = {}
        for name, value in num_dict.items():
            bin_idx = get_bucket_id(value, self.num_bins)
            features[name] = bin_idx
        return features

    def _process_text_features(self, prompt, width):
        # process n-gram features

        s = re.split(r'[^\w]+', prompt.lower())
        
        ngrams = [''.join(s[i:i + width]) for i in range(max(len(s) - width + 1, 1))]

        # features = {}
        # for gram in set(ngrams):
        #     features[gram] = ngrams.count(gram) / len(ngrams)

        return list(set(ngrams))
    
    def get_feature(self, prompt):


        # feature 1: n-gram
        text_feature = self._process_text_features(prompt, 2)
        
        # feature 2: special token number
        #special_token_number = sum(not c.isalnum() for c in prompt)
        #special_token = [c for c in prompt if not c.isalnum()]
        
        
        # feature 3: key words number
        #key_words_number = sum(prompt.lower().count(keyword) for keyword in self.keywords)

        #num_dict = {
        #    'special': special_token_number,
        #    'keyword': key_words_number
        #}
        #numberic_feature = self._process_numeric_features(num_dict)

        combined = []
        for k in text_feature:
            combined.append(k)
        #for c in special_token:
        #    combined.append(c)
        #for k, v in numberic_feature.items():
        #    combined.append(str(v))
        #combined.append(str(special_token_number))
        #combined.append(str(key_words_number))
        return combined
    
        
    # def getDigest(self, prompt):
    #     # embedding = self.model.encode(prompt)
    #     features = self.get_feature(prompt)
    #     return Simhash(features)
    
    def resultsTopk(self, promptsimhash, k):
        results = self.index.get_near_dups(promptsimhash,k)
        if len(results) == 0:
            return None, None
        idx = [int(results[i][0]) for i in range(len(results))]
        distance = [int(results[i][1]) for i in range(len(results))]
        return idx, distance
    
    def add_node(self, prompt, label, profile):
        # if profile:
        #     pr = cProfile.Profile()
        #     pr.enable()
        
        # Your existing code
        self.input_idx += 1
        prompt_feature = self.get_feature(prompt)
        prompt_simhash = Simhash(prompt_feature, f = self.f)
        idx, distance = self.resultsTopk(prompt_simhash, 1)
        self.g.add_node(self.input_idx, label = label)
        self.index.add(str(self.input_idx), prompt_simhash)
        self.simhash_map[str(self.input_idx)] = prompt_simhash
        if idx is None:
            # if profile:
            #     pr.disable()
            #     s = io.StringIO()
            #     ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
            #     ps.print_stats()
            #     print(s.getvalue())
            return None, None, False
        
        res = False

        matched_idx, d = idx[0], distance[0]
        if d <= 30:
            self.g.add_edge(matched_idx, self.input_idx, distance = d)
            if matched_idx in self.alerted_nodes:
                res = True
                self.alerted_nodes.add(self.input_idx)
        
        # if profile:
        #     pr.disable()
        #     s = io.StringIO()
        #     ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
        #     ps.print_stats()
        #     print(s.getvalue())

        return d, matched_idx, res

        

    def detector(self, topK = 5):
        self.alerted_nodes = set()
        subgraph_list = [CacheGraph(self.g.subgraph(s).copy()) for s in nx.connected_components(self.g)]
        # print(f"subgraph_list: {len(subgraph_list)}")
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
                    self.alerted_nodes |= set(i.graph.nodes())
        else:
            indices = []
        
        removed_graph = sorted(subgraph_list, key = lambda x: x.node_nums, reverse = True)[topK:]
        graph=sorted(subgraph_list, key = lambda x: x.node_nums, reverse = True)[:topK]
        for i in graph:
            print(i.node_nums)
        for i in removed_graph:
            self.g.remove_nodes_from(i.graph.nodes())
            for node in i.graph.nodes():
                self.index.delete(str(node), self.simhash_map[str(node)])
                self.simhash_map.pop(str(node))

        # print(f"removed_graph:{len(removed_graph)}")
        # print(f"graph_size: {len(self.g.nodes())}")
        # print(f"hash_map_size: {len(self.simhash_map)}")
        # print(f"get_index_size: {self.index.get_index_size()}")

        return indices