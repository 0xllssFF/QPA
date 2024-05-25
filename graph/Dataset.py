from torch_geometric.data import Dataset, download_url
import os
import networkx as nx
import torch
from torch_geometric.utils.convert import from_networkx
from tqdm import tqdm

class MLaaSDataset(Dataset):
    def __init__(self, root, raw_dir, processed_dir, transform=None, pre_transform=None):
        super(MLaaSDataset, self).__init__(root, transform, pre_transform)
        self.root = root
    @property
    def raw_file_names(self):
        file_list = []
        for root, dirs, files in os.walk(self.raw_dir):
            for file in files:
                file_list.append(os.path.join(root, file))
        return sorted(file_list)

    @property
    def processed_file_names(self):
        return [ f'data_{i}.pt'for i in range(len(self.raw_file_names))]

    def download(self):
        """这里不需要下载"""
        pass

    def process(self):
        for idx, file in tqdm(enumerate(self.raw_file_names)):
            print(file)
            graph_data = nx.Graph(nx.nx_pydot.read_dot(file))
            for edge in graph_data.edges:
                graph_data.edges[edge]['label'] = float(graph_data.edges[edge]['label'])
            largest = max(nx.connected_components(graph_data),key=len)
            largest_connected_subgraph = graph_data.subgraph(largest)
            edge_attr = nx.get_edge_attributes(largest_connected_subgraph, 'label')
            converted_largest_connected_subgraph = nx.line_graph(largest_connected_subgraph)
            for node in converted_largest_connected_subgraph.nodes:
                converted_largest_connected_subgraph.nodes[node]['label'] = edge_attr[node]
            pyg_graph = from_networkx(converted_largest_connected_subgraph, group_node_attrs = all)
            if "anomaly" in file:
                pyg_graph.y = torch.tensor([1])
            else:
                pyg_graph.y = torch.tensor([0])
            torch.save(pyg_graph, os.path.join(self.processed_dir, f'data_{idx}.pt'))


    def len(self):
        return len(self.raw_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        return data
