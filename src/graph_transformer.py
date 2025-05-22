import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=2):
        super(GraphEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x

# 示例图构造（测试用）
def build_sample_graph():
    # 5个节点，4条边（双向）
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3]
    ], dtype=torch.long)
    x = torch.rand((5, 16))  # 5个节点，每个16维特征
    return Data(x=x, edge_index=edge_index)

def encode_graph(data: Data):
    encoder = GraphEncoder(in_channels=data.x.shape[1], hidden_channels=32, out_channels=16)
    return encoder(data.x, data.edge_index)  # 输出每个节点的语义向量
