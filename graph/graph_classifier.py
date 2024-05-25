from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, TransformerConv
from torch_geometric.nn import global_mean_pool
import torch
import os
import networkx as nx
from torch_geometric.data import Dataset, download_url
from torch_geometric.data import DataLoader
from Dataset import MLaaSDataset
from torch.utils.data import random_split

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


if __name__ == '__main__':
    graph_path = '../graph_data/'

    dataset = MLaaSDataset(root=graph_path, raw_dir=os.path.join(graph_path, 'raw'), processed_dir=os.path.join(graph_path, 'processed'))

    train_ratio, valid_ratio = 0.7, 0.3
    len_dataset = dataset.__len__()
    train_dataset, test_dataset = random_split(
        dataset=dataset,
        lengths=[round(train_ratio * len_dataset), round(valid_ratio * len_dataset)],
        generator=torch.Generator().manual_seed(0)
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)




    model = GCN(hidden_channels=64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    def train():
        model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
            out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
            loss = criterion(out, data.y)  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

    def test(loader):
        model.eval()

        correct = 0
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(data.x, data.edge_index, data.batch)  
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        return correct / len(loader.dataset)  # Derive ratio of correct predictions.

    best_acc = 0
    for epoch in range(1, 50):
        train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(model, 'graph_model.pt')

        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

