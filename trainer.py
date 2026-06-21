import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score


class EarlyStopping:
    def __init__(self, patience=50, min_delta=0.0, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.counter = 0
        self.best_score = None
        self.best_state = None
        self.should_stop = False

    def __call__(self, val_score, model):
        score = val_score
        if self.best_score is None:
            self.best_score = score
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.restore_best and self.best_state is not None:
                    model.load_state_dict(self.best_state)
        else:
            self.best_score = score
            self.best_state = {k: v.clone() for k, v in model.state_dict().items()}
            self.counter = 0


def train_epoch(model, data, optimizer, device):
    model.train()
    optimizer.zero_grad()
    out = model(data.x.to(device), data.edge_index.to(device))
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask].to(device))
    loss.backward()
    optimizer.step()
    preds = out[data.train_mask].argmax(dim=-1).cpu().numpy()
    labels = data.y[data.train_mask].cpu().numpy()
    acc = accuracy_score(labels, preds) if len(labels) > 0 else 0.0
    return loss.item(), acc


@torch.no_grad()
def evaluate(model, data, mask, device):
    model.eval()
    out = model(data.x.to(device), data.edge_index.to(device))
    mask = mask.to(torch.bool)
    if mask.sum() == 0:
        return 0.0, 0.0, np.array([]), np.array([])
    preds = out[mask].argmax(dim=-1).cpu().numpy()
    labels = data.y[mask].cpu().numpy()
    acc = accuracy_score(labels, preds)
    loss = F.cross_entropy(out[mask], data.y[mask].to(device)).item()
    return loss, acc, preds, labels


def train_model(model, data, lr=0.01, weight_decay=5e-4, max_epochs=200,
                patience=50, device=None, verbose=False, progress_callback=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to('cpu')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=patience)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'epoch_times': []
    }

    best_val_acc = 0.0
    best_epoch = 0

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, data, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, data, data.val_mask, device)

        epoch_time = time.time() - epoch_start
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_times'].append(epoch_time)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

        early_stopping(val_acc, model)

        if verbose and epoch % 10 == 0:
            print(f'Epoch {epoch:3d}: Train Loss={train_loss:.4f}, '
                  f'Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}')

        if progress_callback is not None:
            progress_callback(epoch, max_epochs, train_loss, train_acc,
                              val_loss, val_acc, best_val_acc, best_epoch)

        if early_stopping.should_stop:
            if verbose:
                print(f'Early stopping at epoch {epoch}')
            break

    total_time = time.time() - start_time

    test_loss, test_acc, test_preds, test_labels = evaluate(model, data, data.test_mask, device)
    f1_macro = f1_score(test_labels, test_preds, average='macro',
                        zero_division=0) if len(test_labels) > 0 else 0.0
    f1_per_class = f1_score(test_labels, test_preds, average=None,
                            zero_division=0) if len(test_labels) > 0 else np.array([])
    cm = confusion_matrix(test_labels, test_preds) if len(test_labels) > 0 else np.array([[]])

    results = {
        'history': history,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'f1_macro': f1_macro,
        'f1_per_class': f1_per_class,
        'confusion_matrix': cm,
        'total_time': total_time,
        'epochs_run': epoch,
        'test_predictions': test_preds,
        'test_labels': test_labels
    }

    return model, results


@torch.no_grad()
def extract_embeddings(model, data, device=None, layer=-1):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    if hasattr(model, 'get_embedding'):
        emb = model.get_embedding(x, edge_index, layer=layer)
    else:
        _, embeddings = model(x, edge_index, return_embeddings=True)
        emb = embeddings[layer]

    return emb.cpu().numpy()


@torch.no_grad()
def extract_attention(model, data, device=None, layer=-1):
    if not hasattr(model, 'get_attention'):
        return None
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)

    edge_idx, attn = model.get_attention(x, edge_index, layer=layer)
    return edge_idx.cpu().numpy(), attn.cpu().numpy()


def reduce_embeddings(embeddings, method='tsne', n_components=2, random_state=42):
    from sklearn.manifold import TSNE
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=random_state,
                       perplexity=min(30, max(5, len(embeddings) // 4 - 1)))
        reduced = reducer.fit_transform(embeddings)
    elif method.lower() == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=n_components, random_state=random_state)
            reduced = reducer.fit_transform(embeddings)
        except ImportError:
            reducer = TSNE(n_components=n_components, random_state=random_state)
            reduced = reducer.fit_transform(embeddings)
    else:
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components, random_state=random_state)
        reduced = reducer.fit_transform(embeddings)
    return reduced
