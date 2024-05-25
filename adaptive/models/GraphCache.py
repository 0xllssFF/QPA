import networkx as nx
import math
class CacheGraph(object):
    def __init__(self, graph):
        self.graph = graph
        self.node_nums = len(graph.nodes)
        self.edge_nums = len(graph.edges)
        self.score = 0
        self.age = 0
        self.decay = 0.9
        for edges in graph.edges():
            self.score += graph.get_edge_data(*edges)['label']


    def GetGraphScore(self):
        return self.score
        
        