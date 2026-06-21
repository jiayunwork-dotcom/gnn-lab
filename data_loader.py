import os
import io
import numpy as np
import pandas as pd
import networkx as nx
from torch_geometric.datasets import Planetoid, KarateClub
from torch_geometric.utils import to_networkx, from_networkx, degree
import torch


def load_builtin_dataset(name):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)

    if name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root=data_dir, name=name)
        data = dataset[0]
        G = to_networkx(data, to_undirected=True)
        return data, G, dataset.num_classes
    elif name == 'Karate Club':
        dataset = KarateClub()
        data = dataset[0]
        if data.x is None or data.x.shape[1] == 0:
            data.x = _generate_default_features(data.edge_index, data.num_nodes)
        G = to_networkx(data, to_undirected=True)
        return data, G, dataset.num_classes
    else:
        raise ValueError(f"Unknown dataset: {name}")


def _generate_default_features(edge_index, num_nodes, method='degree_onehot'):
    if method == 'degree_onehot':
        deg = degree(edge_index[0], num_nodes=num_nodes).long()
        max_deg = deg.max().item() + 1
        x = torch.zeros((num_nodes, max_deg))
        x[torch.arange(num_nodes), deg] = 1.0
        return x
    elif method == 'identity':
        return torch.eye(num_nodes)
    else:
        return torch.ones((num_nodes, 1))


def load_from_csv(nodes_csv_content, edges_csv_content, node_id_col='node_id',
                  source_col='source', target_col='target', weight_col=None,
                  label_col=None):
    if isinstance(nodes_csv_content, str):
        nodes_df = pd.read_csv(io.StringIO(nodes_csv_content))
    else:
        nodes_df = pd.read_csv(nodes_csv_content)

    if isinstance(edges_csv_content, str):
        edges_df = pd.read_csv(io.StringIO(edges_csv_content))
    else:
        edges_df = pd.read_csv(edges_csv_content)

    G = nx.Graph()
    node_ids = nodes_df[node_id_col].tolist()
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    num_nodes = len(node_ids)

    for i, nid in enumerate(node_ids):
        G.add_node(i, original_id=nid)

    for _, row in edges_df.iterrows():
        s = id_to_idx.get(row[source_col])
        t = id_to_idx.get(row[target_col])
        if s is not None and t is not None:
            if weight_col and weight_col in row and pd.notna(row[weight_col]):
                G.add_edge(s, t, weight=float(row[weight_col]))
            else:
                G.add_edge(s, t)

    feature_cols = [c for c in nodes_df.columns if c not in [node_id_col, label_col]]
    if len(feature_cols) > 0:
        x = torch.tensor(nodes_df[feature_cols].values, dtype=torch.float)
    else:
        x = None

    y = None
    num_classes = 0
    if label_col and label_col in nodes_df.columns:
        labels_raw = nodes_df[label_col].tolist()
        unique_labels = sorted(set(labels_raw))
        label_map = {l: i for i, l in enumerate(unique_labels)}
        y = torch.tensor([label_map[l] for l in labels_raw], dtype=torch.long)
        num_classes = len(unique_labels)

    data = from_networkx(G)
    data.num_nodes = num_nodes
    if x is not None:
        data.x = x
    else:
        data.x = _generate_default_features(data.edge_index, num_nodes)
    if y is not None:
        data.y = y

    return data, G, num_classes


def load_from_graphml(file_content):
    G = nx.read_graphml(io.BytesIO(file_content) if isinstance(file_content, bytes) else file_content)
    return _convert_networkx_to_data(G)


def load_from_gml(file_content):
    G = nx.read_gml(io.BytesIO(file_content) if isinstance(file_content, bytes) else file_content, destringizer=None)
    return _convert_networkx_to_data(G)


def _convert_networkx_to_data(G):
    nodes = list(G.nodes())
    id_to_idx = {n: i for i, n in enumerate(nodes)}
    G_idx = nx.Graph()
    for i, n in enumerate(nodes):
        attrs = {k: v for k, v in G.nodes[n].items() if k != 'id'}
        G_idx.add_node(i, **attrs)

    for u, v, data in G.edges(data=True):
        G_idx.add_edge(id_to_idx[u], id_to_idx[v], **data)

    num_nodes = len(nodes)
    data = from_networkx(G_idx)
    data.num_nodes = num_nodes

    has_feature = False
    features = []
    for i in range(num_nodes):
        node_attrs = G_idx.nodes[i]
        feat = []
        for k, v in node_attrs.items():
            if k in ['label', 'y', 'class', 'category']:
                continue
            if isinstance(v, (int, float)):
                feat.append(v)
        if feat:
            has_feature = True
            features.append(feat)

    labels = []
    has_label = False
    label_map = {}
    num_classes = 0
    for i in range(num_nodes):
        node_attrs = G_idx.nodes[i]
        lbl = None
        for k in ['label', 'y', 'class', 'category']:
            if k in node_attrs:
                lbl = node_attrs[k]
                break
        if lbl is not None:
            has_label = True
            if lbl not in label_map:
                label_map[lbl] = len(label_map)
            labels.append(label_map[lbl])

    if has_feature:
        data.x = torch.tensor(features, dtype=torch.float)
    else:
        data.x = _generate_default_features(data.edge_index, num_nodes)

    if has_label:
        data.y = torch.tensor(labels, dtype=torch.long)
        num_classes = len(label_map)

    return data, G_idx, num_classes


def graph_statistics(G):
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    if num_nodes == 0:
        return {
            'num_nodes': 0, 'num_edges': 0, 'avg_degree': 0.0,
            'clustering_coeff': 0.0, 'num_components': 0, 'degree_dist': np.array([])
        }

    degrees = [d for _, d in G.degree()]
    avg_degree = np.mean(degrees) if degrees else 0.0

    try:
        clustering_coeff = nx.average_clustering(G)
    except Exception:
        clustering_coeff = 0.0

    num_components = nx.number_connected_components(G)
    degree_dist = np.array(degrees)

    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': round(avg_degree, 4),
        'clustering_coeff': round(clustering_coeff, 4),
        'num_components': num_components,
        'degree_dist': degree_dist
    }


def split_data(data, num_classes, train_ratio=0.6, val_ratio=0.2, random_seed=42):
    num_nodes = data.num_nodes
    has_y = hasattr(data, 'y') and data.y is not None

    if hasattr(data, 'train_mask') and hasattr(data, 'val_mask') and hasattr(data, 'test_mask'):
        if data.train_mask is not None and data.train_mask.sum() > 0:
            return

    if not has_y:
        if not hasattr(data, 'train_mask') or data.train_mask is None:
            data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data.test_mask = torch.ones(num_nodes, dtype=torch.bool)
        return

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    y = data.y.cpu().numpy()
    for c in range(num_classes):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        np.random.shuffle(idx)
        n_train = max(1, int(len(idx) * train_ratio))
        n_val = max(1, int(len(idx) * val_ratio))
        n_train = min(n_train, len(idx) - 2) if len(idx) >= 3 else 1
        n_val = min(n_val, len(idx) - n_train - 1) if len(idx) - n_train > 1 else 0
        train_mask[idx[:n_train]] = True
        val_mask[idx[n_train:n_train + n_val]] = True
        test_mask[idx[n_train + n_val:]] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask


def get_largest_component(G):
    if G.number_of_nodes() == 0:
        return G
    components = list(nx.connected_components(G))
    largest = max(components, key=len)
    return G.subgraph(largest).copy()


def get_subgraph_by_nodes(G, node_indices):
    return G.subgraph(node_indices).copy()


BUILTIN_DATASETS = ['Cora', 'Citeseer', 'Pubmed', 'Karate Club']
