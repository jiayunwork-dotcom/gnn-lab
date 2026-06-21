import numpy as np
import networkx as nx
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.manifold import spectral_embedding

try:
    import community as community_louvain
except ImportError:
    import community.community_louvain as community_louvain


def spectral_clustering(G, k=2):
    adj = nx.to_numpy_array(G)
    if adj.sum() == 0:
        return np.zeros(G.number_of_nodes(), dtype=int), 0.0
    sc = SpectralClustering(n_clusters=k, affinity='precomputed',
                            random_state=42, assign_labels='kmeans')
    labels = sc.fit_predict(adj)
    mod = compute_modularity(G, labels)
    return labels, mod


def louvain_community(G):
    partition = community_louvain.best_partition(G)
    labels = np.array([partition.get(i, 0) for i in range(G.number_of_nodes())])
    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels])
    mod = compute_modularity(G, labels)
    return labels, mod


def label_propagation(G, max_iter=100):
    adj = nx.to_numpy_array(G)
    n = adj.shape[0]
    if n == 0:
        return np.zeros(0, dtype=int), 0.0

    labels = np.arange(n)
    for iteration in range(max_iter):
        changed = False
        order = np.random.permutation(n)
        for node in order:
            neighbors = np.where(adj[node] > 0)[0]
            if len(neighbors) == 0:
                continue
            neighbor_labels = labels[neighbors]
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            max_count = counts.max()
            candidates = unique[counts == max_count]
            new_label = np.random.choice(candidates)
            if new_label != labels[node]:
                labels[node] = new_label
                changed = True
        if not changed:
            break

    unique_labels = np.unique(labels)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    labels = np.array([label_map[l] for l in labels])
    mod = compute_modularity(G, labels)
    return labels, mod


def gnn_kmeans(embeddings, k=2, G=None):
    if isinstance(embeddings, np.ndarray):
        X = embeddings
    else:
        X = embeddings.detach().cpu().numpy()
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    mod = compute_modularity(G, labels) if G is not None else None
    return labels, mod


def compute_modularity(G, labels):
    if G.number_of_edges() == 0:
        return 0.0
    communities = {}
    for node, label in enumerate(labels):
        if label not in communities:
            communities[label] = set()
        communities[label].add(node)
    community_list = list(communities.values())
    try:
        mod = nx.community.modularity(G, community_list)
    except Exception:
        mod = 0.0
    return mod


COMMUNITY_ALGORITHMS = {
    'Spectral Clustering': spectral_clustering,
    'Louvain': louvain_community,
    'Label Propagation': label_propagation,
    'GNN Embedding + K-Means': gnn_kmeans,
}


def run_community_detection(algorithm, G, k=None, embeddings=None):
    alg = algorithm
    if alg == 'Spectral Clustering':
        if k is None:
            k = 2
        labels, mod = spectral_clustering(G, k=k)
    elif alg == 'Louvain':
        labels, mod = louvain_community(G)
        k = len(np.unique(labels))
    elif alg == 'Label Propagation':
        labels, mod = label_propagation(G)
        k = len(np.unique(labels))
    elif alg == 'GNN Embedding + K-Means':
        if embeddings is None:
            labels = np.zeros(G.number_of_nodes(), dtype=int)
            mod = 0.0
        else:
            if k is None:
                k = 2
            labels, mod = gnn_kmeans(embeddings, k=k, G=G)
    else:
        raise ValueError(f"Unknown algorithm: {alg}")

    num_communities = len(np.unique(labels)) if len(labels) > 0 else 0
    return labels, mod, num_communities
