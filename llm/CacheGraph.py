import networkx as nx

class CacheGraph(object):
    def __init__(self, graph):
        self.graph = graph
        self.node_nums = len(graph.nodes)
        self.edge_nums = len(graph.edges)
        #self.score = self.node_nums
        self.score = 0
        for edges in graph.edges():
            self.score +=  1/(int(graph.get_edge_data(*edges)['distance'])+1)

    def GetGraphScore(self):
        return self.score
        