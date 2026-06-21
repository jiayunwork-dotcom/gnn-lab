import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv


def get_activation(name):
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'elu':
        return nn.ELU()
    elif name == 'leakyrelu':
        return nn.LeakyReLU(0.2)
    else:
        return nn.ReLU()


class GCNModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 activation='relu', dropout=0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = get_activation(activation)
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels))
        else:
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.convs.append(GCNConv(hidden_channels, out_channels))

    def forward(self, x, edge_index, return_embeddings=False):
        embeddings = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = self.act(x)
                embeddings.append(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                embeddings.append(x)
        if return_embeddings:
            return x, embeddings
        return x

    def get_embedding(self, x, edge_index, layer=-1):
        _, embeddings = self.forward(x, edge_index, return_embeddings=True)
        return embeddings[layer]


class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 activation='relu', dropout=0.5, heads=8, concat_last=False):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = get_activation(activation)
        self.heads = heads
        self.concat_last = concat_last
        self.convs = nn.ModuleList()

        if num_layers == 1:
            out_heads = 1 if not concat_last else heads
            self.convs.append(GATConv(in_channels, out_channels, heads=out_heads,
                                      concat=concat_last, dropout=dropout))
        else:
            self.convs.append(GATConv(in_channels, hidden_channels, heads=heads,
                                      concat=True, dropout=dropout))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels * heads, hidden_channels,
                                          heads=heads, concat=True, dropout=dropout))
            out_heads = 1 if not concat_last else heads
            last_hidden = hidden_channels * heads if num_layers > 1 else in_channels
            self.convs.append(GATConv(last_hidden, out_channels, heads=out_heads,
                                      concat=concat_last, dropout=dropout))

    def forward(self, x, edge_index, return_embeddings=False, return_attention=False):
        embeddings = []
        attention_weights = []
        for i, conv in enumerate(self.convs):
            if return_attention:
                x, (edge_idx, attn) = conv(x, edge_index, return_attention_weights=True)
                attention_weights.append((edge_idx, attn))
            else:
                x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = self.act(x)
                embeddings.append(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                embeddings.append(x)
        if return_embeddings and return_attention:
            return x, embeddings, attention_weights
        if return_embeddings:
            return x, embeddings
        if return_attention:
            return x, attention_weights
        return x

    def get_embedding(self, x, edge_index, layer=-1):
        _, embeddings = self.forward(x, edge_index, return_embeddings=True)
        return embeddings[layer]

    def get_attention(self, x, edge_index, layer=-1):
        _, attn = self.forward(x, edge_index, return_attention=True)
        return attn[layer]


class GraphSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 activation='relu', dropout=0.5, aggr='mean'):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = get_activation(activation)
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels, aggr=aggr))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.convs.append(SAGEConv(hidden_channels, out_channels, aggr=aggr))

    def forward(self, x, edge_index, return_embeddings=False):
        embeddings = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = self.act(x)
                embeddings.append(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                embeddings.append(x)
        if return_embeddings:
            return x, embeddings
        return x

    def get_embedding(self, x, edge_index, layer=-1):
        _, embeddings = self.forward(x, edge_index, return_embeddings=True)
        return embeddings[layer]


class GINModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 activation='relu', dropout=0.5, eps=0.0, train_eps=True):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.act = get_activation(activation)
        self.convs = nn.ModuleList()

        def build_mlp(in_c, out_c):
            return nn.Sequential(
                nn.Linear(in_c, out_c),
                get_activation(activation),
                nn.Linear(out_c, out_c)
            )

        if num_layers == 1:
            self.convs.append(GINConv(build_mlp(in_channels, out_channels),
                                      eps=eps, train_eps=train_eps))
        else:
            self.convs.append(GINConv(build_mlp(in_channels, hidden_channels),
                                      eps=eps, train_eps=train_eps))
            for _ in range(num_layers - 2):
                self.convs.append(GINConv(build_mlp(hidden_channels, hidden_channels),
                                          eps=eps, train_eps=train_eps))
            self.convs.append(GINConv(build_mlp(hidden_channels, out_channels),
                                      eps=eps, train_eps=train_eps))

    def forward(self, x, edge_index, return_embeddings=False):
        embeddings = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                x = self.act(x)
                embeddings.append(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                embeddings.append(x)
        if return_embeddings:
            return x, embeddings
        return x

    def get_embedding(self, x, edge_index, layer=-1):
        _, embeddings = self.forward(x, edge_index, return_embeddings=True)
        return embeddings[layer]


def create_model(model_name, in_channels, out_channels, hidden_channels=64,
                 num_layers=2, activation='relu', dropout=0.5, **kwargs):
    model_name = model_name.upper()
    if model_name == 'GCN':
        return GCNModel(in_channels, hidden_channels, out_channels,
                        num_layers=num_layers, activation=activation, dropout=dropout)
    elif model_name == 'GAT':
        heads = kwargs.get('heads', 8)
        concat_last = kwargs.get('concat_last', False)
        return GATModel(in_channels, hidden_channels, out_channels,
                        num_layers=num_layers, activation=activation, dropout=dropout,
                        heads=heads, concat_last=concat_last)
    elif model_name == 'GRAPHSAGE':
        aggr = kwargs.get('aggr', 'mean')
        return GraphSAGEModel(in_channels, hidden_channels, out_channels,
                              num_layers=num_layers, activation=activation, dropout=dropout,
                              aggr=aggr)
    elif model_name == 'GIN':
        eps = kwargs.get('eps', 0.0)
        train_eps = kwargs.get('train_eps', True)
        return GINModel(in_channels, hidden_channels, out_channels,
                        num_layers=num_layers, activation=activation, dropout=dropout,
                        eps=eps, train_eps=train_eps)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


MODEL_NAMES = ['GCN', 'GAT', 'GraphSAGE', 'GIN']
ACTIVATIONS = ['ReLU', 'ELU', 'LeakyReLU']
HIDDEN_DIMS = [16, 32, 64, 128, 256]
NUM_LAYERS_RANGE = [1, 2, 3, 4, 5]
GAT_HEADS = [1, 2, 4, 8]
