import copy
import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from sklearn.metrics import accuracy_score, f1_score
from torch_geometric.utils import to_networkx


ATTACK_METHODS = {
    '随机攻击': 'random',
    '度数攻击': 'degree',
    '梯度攻击': 'gradient',
}

ATTACK_MODES = {
    '添加边': 'add',
    '删除边': 'remove',
    '翻转边': 'flip',
}

DEFENSE_METHODS = {
    '度数边过滤': 'degree_filter',
    '特征平滑': 'feature_smoothing',
}


def _edge_index_to_set(edge_index):
    edge_set = set()
    for i in range(edge_index.shape[1]):
        u = int(edge_index[0, i])
        v = int(edge_index[1, i])
        if u <= v:
            edge_set.add((u, v))
        else:
            edge_set.add((v, u))
    return edge_set


def _set_to_edge_index(edge_set, num_nodes):
    if len(edge_set) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    edges_u = []
    edges_v = []
    for u, v in edge_set:
        edges_u.extend([u, v])
        edges_v.extend([v, u])
    return torch.tensor([edges_u, edges_v], dtype=torch.long)


def _get_existing_edges(data):
    return _edge_index_to_set(data.edge_index)


def _get_non_existing_edges(data, num_nodes=None):
    if num_nodes is None:
        num_nodes = data.num_nodes
    existing = _get_existing_edges(data)
    non_existing = set()
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if (u, v) not in existing:
                non_existing.add((u, v))
    return non_existing


def _apply_edge_changes(data, edges_to_add, edges_to_remove):
    new_data = data.clone()
    edge_set = _get_existing_edges(data)

    for u, v in edges_to_remove:
        edge_set.discard((min(u, v), max(u, v)))
    for u, v in edges_to_add:
        edge_set.add((min(u, v), max(u, v)))

    new_data.edge_index = _set_to_edge_index(edge_set, data.num_nodes)
    return new_data


def _select_edges_to_remove(data, num_edges, priority_scores=None):
    existing = _get_existing_edges(data)
    existing_list = list(existing)
    if len(existing_list) == 0:
        return []
    if priority_scores is not None:
        scored_edges = []
        for u, v in existing_list:
            score = priority_scores.get((u, v), priority_scores.get((v, u), 0))
            scored_edges.append((u, v, score))
        scored_edges.sort(key=lambda x: x[2], reverse=True)
        n = min(num_edges, len(scored_edges))
        return [(e[0], e[1]) for e in scored_edges[:n]]
    else:
        indices = np.random.choice(len(existing_list), size=min(num_edges, len(existing_list)), replace=False)
        return [existing_list[i] for i in indices]


def _select_edges_to_add(data, num_edges, priority_scores=None):
    num_nodes = data.num_nodes
    existing = _get_existing_edges(data)
    non_existing = []
    for u in range(num_nodes):
        for v in range(u + 1, num_nodes):
            if (u, v) not in existing:
                non_existing.append((u, v))

    if len(non_existing) == 0:
        return []

    if priority_scores is not None:
        scored_edges = []
        for u, v in non_existing:
            score = priority_scores.get((u, v), priority_scores.get((v, u), 0))
            scored_edges.append((u, v, score))
        scored_edges.sort(key=lambda x: x[2], reverse=True)
        n = min(num_edges, len(scored_edges))
        return [(e[0], e[1]) for e in scored_edges[:n]]
    else:
        indices = np.random.choice(len(non_existing), size=min(num_edges, len(non_existing)), replace=False)
        return [non_existing[i] for i in indices]


def random_attack(data, ratio, mode='flip', seed=None):
    if seed is not None:
        np.random.seed(seed)

    num_total_edges = data.edge_index.shape[1] // 2
    num_perturb = max(1, int(num_total_edges * ratio))

    if mode == 'add':
        edges = _select_edges_to_add(data, num_perturb)
        return _apply_edge_changes(data, edges_to_add=edges, edges_to_remove=[])
    elif mode == 'remove':
        edges = _select_edges_to_remove(data, num_perturb)
        return _apply_edge_changes(data, edges_to_add=[], edges_to_remove=edges)
    elif mode == 'flip':
        n_add = num_perturb // 2
        n_remove = num_perturb - n_add
        edges_add = _select_edges_to_add(data, n_add)
        edges_remove = _select_edges_to_remove(data, n_remove)
        return _apply_edge_changes(data, edges_to_add=edges_add, edges_to_remove=edges_remove)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def degree_attack(data, ratio, mode='flip'):
    num_total_edges = data.edge_index.shape[1] // 2
    num_perturb = max(1, int(num_total_edges * ratio))

    edge_index = data.edge_index
    num_nodes = data.num_nodes
    degree_scores = torch.zeros(num_nodes)
    for i in range(edge_index.shape[1]):
        degree_scores[edge_index[0, i]] += 1
    degree_scores = degree_scores.numpy()

    edge_priority = {}
    for i in range(edge_index.shape[1]):
        u = int(edge_index[0, i])
        v = int(edge_index[1, i])
        key = (min(u, v), max(u, v))
        edge_priority[key] = max(degree_scores[u], degree_scores[v])

    if mode == 'add':
        add_priority = {}
        existing = _get_existing_edges(data)
        for u in range(num_nodes):
            for v in range(u + 1, num_nodes):
                if (u, v) not in existing:
                    add_priority[(u, v)] = max(degree_scores[u], degree_scores[v])
        edges = _select_edges_to_add(data, num_perturb, priority_scores=add_priority)
        return _apply_edge_changes(data, edges_to_add=edges, edges_to_remove=[])
    elif mode == 'remove':
        edges = _select_edges_to_remove(data, num_perturb, priority_scores=edge_priority)
        return _apply_edge_changes(data, edges_to_add=[], edges_to_remove=edges)
    elif mode == 'flip':
        n_add = num_perturb // 2
        n_remove = num_perturb - n_add

        add_priority = {}
        existing = _get_existing_edges(data)
        for u in range(num_nodes):
            for v in range(u + 1, num_nodes):
                if (u, v) not in existing:
                    add_priority[(u, v)] = max(degree_scores[u], degree_scores[v])

        edges_add = _select_edges_to_add(data, n_add, priority_scores=add_priority)
        edges_remove = _select_edges_to_remove(data, n_remove, priority_scores=edge_priority)
        return _apply_edge_changes(data, edges_to_add=edges_add, edges_to_remove=edges_remove)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def gradient_attack(data, model, ratio, mode='flip', device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    num_total_edges = data.edge_index.shape[1] // 2
    num_perturb = max(1, int(num_total_edges * ratio))

    x = data.x.to(device)
    num_nodes = data.num_nodes
    y = data.y.to(device)

    if hasattr(data, 'test_mask') and data.test_mask is not None and data.test_mask.sum() > 0:
        target_mask = data.test_mask.to(device)
    elif hasattr(data, 'val_mask') and data.val_mask is not None and data.val_mask.sum() > 0:
        target_mask = data.val_mask.to(device)
    else:
        target_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)

    adj = torch.zeros(num_nodes, num_nodes, device=device)
    adj[data.edge_index[0], data.edge_index[1]] = 1.0

    adj_var = adj.clone().detach().requires_grad_(True)

    adj_with_self = adj_var + torch.eye(num_nodes, device=device)
    deg = adj_with_self.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt = torch.where(torch.isinf(deg_inv_sqrt), torch.zeros_like(deg_inv_sqrt), deg_inv_sqrt)
    adj_norm = deg_inv_sqrt.unsqueeze(1) * adj_with_self * deg_inv_sqrt.unsqueeze(0)

    conv_weights = []
    conv_biases = []
    activation = None
    for name, param in model.named_parameters():
        if 'lin.weight' in name:
            conv_weights.append(param.detach())
        elif 'bias' in name and 'lin' not in name:
            conv_biases.append(param.detach())

    if len(conv_weights) == 0:
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() == 2:
                conv_weights.append(param.detach())
            elif 'bias' in name and param.dim() == 1:
                conv_biases.append(param.detach())

    h = x
    for i in range(len(conv_weights)):
        w = conv_weights[i]
        h = adj_norm @ h @ w.t()
        if i < len(conv_biases) and i < len(conv_weights) - 1:
            h = h + conv_biases[i]
        if i < len(conv_weights) - 1:
            h = F.relu(h)
    if len(conv_biases) == len(conv_weights) and len(conv_biases) > 0:
        h = h + conv_biases[-1]

    out = h

    loss = F.cross_entropy(out[target_mask], y[target_mask])
    loss.backward()

    grad = adj_var.grad.detach()
    grad_score = (grad.abs() + grad.abs().t()) / 2.0

    existing = _get_existing_edges(data)

    if mode == 'remove':
        edge_grad = {}
        for u, v in existing:
            edge_grad[(u, v)] = grad_score[u, v].item()
        edges = _select_edges_to_remove(data, num_perturb, priority_scores=edge_grad)
        return _apply_edge_changes(data, edges_to_add=[], edges_to_remove=edges)

    elif mode == 'add':
        non_existing_grad = {}
        for u in range(num_nodes):
            for v in range(u + 1, num_nodes):
                if (u, v) not in existing:
                    non_existing_grad[(u, v)] = grad_score[u, v].item()
        edges = _select_edges_to_add(data, num_perturb, priority_scores=non_existing_grad)
        return _apply_edge_changes(data, edges_to_add=edges, edges_to_remove=[])

    elif mode == 'flip':
        n_remove = num_perturb // 2
        n_add = num_perturb - n_remove

        edge_grad = {}
        for u, v in existing:
            edge_grad[(u, v)] = grad_score[u, v].item()

        non_existing_grad = {}
        for u in range(num_nodes):
            for v in range(u + 1, num_nodes):
                if (u, v) not in existing:
                    non_existing_grad[(u, v)] = grad_score[u, v].item()

        edges_remove = _select_edges_to_remove(data, n_remove, priority_scores=edge_grad)
        edges_add = _select_edges_to_add(data, n_add, priority_scores=non_existing_grad)
        return _apply_edge_changes(data, edges_to_add=edges_add, edges_to_remove=edges_remove)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def degree_filter_defense(original_data, attacked_data, threshold_percentile=95):
    num_nodes = attacked_data.num_nodes
    edge_index = attacked_data.edge_index

    degree_counts = torch.zeros(num_nodes)
    for i in range(edge_index.shape[1]):
        degree_counts[edge_index[0, i]] += 1
    degree_np = degree_counts.numpy()

    threshold = np.percentile(degree_np[degree_np > 0], threshold_percentile)

    original_edges = _get_existing_edges(original_data)
    attacked_edges = _get_existing_edges(attacked_data)

    new_edges = attacked_edges - original_edges

    edges_to_remove = set()
    for u, v in new_edges:
        if degree_np[u] > threshold or degree_np[v] > threshold:
            edges_to_remove.add((u, v))

    defended_data = _apply_edge_changes(attacked_data, edges_to_add=[], edges_to_remove=list(edges_to_remove))
    return defended_data


def feature_smoothing_defense(attacked_data):
    new_data = attacked_data.clone()
    edge_index = attacked_data.edge_index
    num_nodes = attacked_data.num_nodes

    row = edge_index[0]
    col = edge_index[1]

    deg = torch.zeros(num_nodes)
    deg.scatter_add_(0, row, torch.ones(row.shape[0]))
    deg = deg.clamp(min=1)

    deg_inv = 1.0 / deg
    row_norm = deg_inv[row]

    x = attacked_data.x.float()
    smoothed = torch.zeros_like(x)
    smoothed.index_add_(0, row, x[col] * row_norm.unsqueeze(1))
    smoothed = (x + smoothed) / 2.0

    new_data.x = smoothed
    return new_data


@torch.no_grad()
def evaluate_on_data(model, data, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    out = model(data.x.to(device), data.edge_index.to(device))

    if hasattr(data, 'test_mask') and data.test_mask is not None:
        mask = data.test_mask
    else:
        mask = torch.ones(data.num_nodes, dtype=torch.bool)

    mask = mask.to(torch.bool)
    if mask.sum() == 0:
        return 0.0, 0.0

    preds = out[mask].argmax(dim=-1).cpu().numpy()
    labels = data.y[mask].cpu().numpy()

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    return acc, f1


@torch.no_grad()
def get_prediction_probs(model, data, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    out = model(data.x.to(device), data.edge_index.to(device))
    probs = torch.softmax(out, dim=1).cpu().numpy()
    return probs


def run_attack(data, model, method, ratio, mode, device=None, seed=42):
    method_key = ATTACK_METHODS.get(method, method)
    mode_key = ATTACK_MODES.get(mode, mode)

    if method_key == 'random':
        attacked_data = random_attack(data, ratio, mode=mode_key, seed=seed)
    elif method_key == 'degree':
        attacked_data = degree_attack(data, ratio, mode=mode_key)
    elif method_key == 'gradient':
        attacked_data = gradient_attack(data, model, ratio, mode=mode_key, device=device)
    else:
        raise ValueError(f"Unknown attack method: {method}")

    return attacked_data


def get_perturbed_edge_info(original_data, attacked_data):
    original_edges = _get_existing_edges(original_data)
    attacked_edges = _get_existing_edges(attacked_data)

    added = attacked_edges - original_edges
    removed = original_edges - attacked_edges

    affected_nodes = set()
    for u, v in added | removed:
        affected_nodes.add(u)
        affected_nodes.add(v)

    return added, removed, affected_nodes


def batch_evaluate(model, data, method, ratios, mode, device=None, defense=None, seed=42):
    results = []
    for ratio in ratios:
        attacked_data = run_attack(data, model, method, ratio, mode, device=device, seed=seed)

        acc_before, f1_before = evaluate_on_data(model, data, device)
        acc_after, f1_after = evaluate_on_data(model, attacked_data, device)

        result = {
            'attack_method': method,
            'attack_ratio': ratio,
            'attack_mode': mode,
            'acc_before': acc_before,
            'acc_after': acc_after,
            'acc_drop': acc_before - acc_after,
            'f1_before': f1_before,
            'f1_after': f1_after,
            'f1_drop': f1_before - f1_after,
            'defense_method': defense,
            'acc_after_defense': None,
            'f1_after_defense': None,
            'acc_defense_improvement': None,
            'f1_defense_improvement': None,
        }

        if defense is not None:
            defense_key = DEFENSE_METHODS.get(defense, defense)
            if defense_key == 'degree_filter':
                defended_data = degree_filter_defense(data, attacked_data)
            elif defense_key == 'feature_smoothing':
                defended_data = feature_smoothing_defense(attacked_data)
            else:
                defended_data = attacked_data

            acc_def, f1_def = evaluate_on_data(model, defended_data, device)
            result['acc_after_defense'] = acc_def
            result['f1_after_defense'] = f1_def
            result['acc_defense_improvement'] = acc_def - acc_after
            result['f1_defense_improvement'] = f1_def - f1_after

        results.append(result)
    return results
