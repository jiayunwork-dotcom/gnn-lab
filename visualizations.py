import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots


def compute_layout(G, layout='spring', seed=42, k=None, iterations=50, scale=10):
    if G.number_of_nodes() == 0:
        return {}
    pos = None
    if layout == 'spring':
        pos = nx.spring_layout(G, seed=seed, k=k, iterations=iterations, scale=scale)
    elif layout == 'fruchterman_reingold':
        pos = nx.fruchterman_reingold_layout(G, seed=seed, k=k, iterations=iterations, scale=scale)
    elif layout == 'kamada_kawai':
        try:
            pos = nx.kamada_kawai_layout(G, scale=scale)
        except Exception:
            pos = nx.spring_layout(G, seed=seed, scale=scale)
    elif layout == 'circular':
        pos = nx.circular_layout(G, scale=scale)
    elif layout == 'spectral':
        try:
            pos = nx.spectral_layout(G, scale=scale)
        except Exception:
            pos = nx.spring_layout(G, seed=seed, scale=scale)
    else:
        pos = nx.spring_layout(G, seed=seed, scale=scale)
    return pos


def plot_degree_distribution(degree_dist):
    if len(degree_dist) == 0:
        return go.Figure()
    bins = min(30, int(np.max(degree_dist)) + 1) if np.max(degree_dist) > 0 else 10
    fig = go.Figure(data=[go.Histogram(x=degree_dist, nbinsx=bins,
                                       marker_color='rgba(55, 83, 109, 0.8)',
                                       marker_line_color='white', marker_line_width=1)])
    fig.update_layout(
        title='度分布直方图',
        xaxis_title='度',
        yaxis_title='节点数量',
        template='plotly_white',
        bargap=0.05,
        height=350,
        margin=dict(l=10, r=10, t=50, b=40)
    )
    return fig


def plot_training_curves(history):
    epochs = list(range(1, len(history['train_loss']) + 1))
    fig = make_subplots(rows=1, cols=2, subplot_titles=('损失曲线', '精度曲线'),
                        horizontal_spacing=0.12)

    fig.add_trace(go.Scatter(x=epochs, y=history['train_loss'], mode='lines',
                             name='训练损失', line=dict(color='#636EFA', width=2)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=epochs, y=history['val_loss'], mode='lines',
                             name='验证损失', line=dict(color='#EF553B', width=2, dash='dash')),
                  row=1, col=1)

    fig.add_trace(go.Scatter(x=epochs, y=history['train_acc'], mode='lines',
                             name='训练精度', line=dict(color='#00CC96', width=2)),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=epochs, y=history['val_acc'], mode='lines',
                             name='验证精度', line=dict(color='#AB63FA', width=2, dash='dash')),
                  row=1, col=2)

    fig.update_xaxes(title_text='Epoch', row=1, col=1)
    fig.update_xaxes(title_text='Epoch', row=1, col=2)
    fig.update_yaxes(title_text='Loss', row=1, col=1)
    fig.update_yaxes(title_text='Accuracy', row=1, col=2)
    fig.update_layout(template='plotly_white', height=400,
                      legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5),
                      margin=dict(l=40, r=20, t=50, b=60))
    return fig


def plot_confusion_matrix(cm, class_names=None):
    if cm.size == 0 or cm.shape[0] == 0:
        return go.Figure()
    if class_names is None:
        class_names = [f'类{i}' for i in range(cm.shape[0])]
    cm_text = [[str(y) for y in x] for x in cm.tolist()]
    z = cm.tolist()
    fig = ff.create_annotated_heatmap(z, x=class_names, y=class_names,
                                      annotation_text=cm_text, colorscale='Blues',
                                      showscale=True)
    fig.update_layout(
        title='混淆矩阵',
        xaxis_title='预测标签',
        yaxis_title='真实标签',
        template='plotly_white',
        height=450,
        xaxis=dict(side='bottom'),
        margin=dict(l=80, r=20, t=60, b=80)
    )
    for i in range(len(fig.layout.annotations)):
        fig.layout.annotations[i].font.size = 11
    return fig


def plot_graph(G, pos=None, node_colors=None, node_sizes=None, edge_weights=None,
               title='图可视化', layout='spring', class_names=None, show_labels=False,
               node_ids=None, max_nodes=500):
    if G.number_of_nodes() == 0:
        return go.Figure()

    if pos is None:
        pos = compute_layout(G, layout=layout)

    if node_colors is None:
        node_colors = [0] * G.number_of_nodes()

    degrees = dict(G.degree())
    if node_sizes is None:
        node_sizes = [max(8, min(40, 5 + degrees.get(n, 0) * 2)) for n in G.nodes()]
    else:
        if len(node_sizes) != G.number_of_nodes():
            node_sizes = [max(8, min(40, 5 + degrees.get(n, 0) * 2)) for n in G.nodes()]

    edge_traces = []
    has_weights = edge_weights is not None and len(edge_weights) > 0

    for i, (u, v, data) in enumerate(G.edges(data=True)):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        w = 1.0
        if has_weights and i < len(edge_weights):
            w = max(0.5, min(6.0, edge_weights[i] * 3))
        elif 'weight' in data:
            w = max(0.5, min(6.0, data['weight'] * 2))

        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines',
            line=dict(width=w, color='rgba(150, 150, 150, 0.6)'),
            hoverinfo='none',
            showlegend=False
        ))

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]

    if node_ids is None:
        node_ids = list(G.nodes())

    text_labels = []
    for n, nid in zip(G.nodes(), node_ids):
        lbl = f'ID: {nid}<br>度: {degrees.get(n, 0)}'
        if node_colors is not None and len(node_colors) > 0:
            lbl += f'<br>标签/社区: {node_colors[list(G.nodes()).index(n)]}'
        text_labels.append(lbl)

    unique_colors = sorted(set(node_colors)) if len(node_colors) > 0 else [0]
    color_scale = 'Viridis'
    if class_names is not None and len(class_names) > 0:
        color_scale = 'Plotly3'

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers' + ('+text' if show_labels else ''),
        hoverinfo='text',
        text=[str(nid) for nid in node_ids] if show_labels else None,
        textposition='top center',
        textfont=dict(size=9),
        marker=dict(
            showscale=True,
            colorscale=color_scale,
            reversescale=False,
            size=node_sizes,
            color=node_colors if len(node_colors) == len(node_x) else node_x,
            line_width=1.5,
            line_color='white',
            colorbar=dict(
                thickness=15,
                title='类别/社区',
                xanchor='left',
                titleside='right'
            )
        ),
        hovertext=text_labels
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=title,
        template='plotly_white',
        height=550,
        showlegend=False,
        hovermode='closest',
        margin=dict(l=5, r=5, t=50, b=5),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig


def plot_embeddings_2d(embeddings_2d, labels=None, title='节点嵌入降维可视化', class_names=None):
    if embeddings_2d is None or len(embeddings_2d) == 0:
        return go.Figure()

    if labels is None:
        labels = [0] * len(embeddings_2d)

    fig = go.Figure()
    unique_labels = sorted(set(labels))
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
              '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

    for i, lbl in enumerate(unique_labels):
        idx = [j for j, l in enumerate(labels) if l == lbl]
        name = class_names[lbl] if class_names and lbl < len(class_names) else f'类{lbl}'
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=embeddings_2d[idx, 0], y=embeddings_2d[idx, 1],
            mode='markers', name=name,
            marker=dict(size=8, color=color, line_width=0.5, line_color='white', opacity=0.85),
            hovertext=[f'节点{j}: {name}' for j in idx]
        ))

    fig.update_layout(
        title=title,
        template='plotly_white',
        height=500,
        xaxis_title='t-SNE/UMAP 维度1',
        yaxis_title='t-SNE/UMAP 维度2',
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
        margin=dict(l=40, r=20, t=60, b=80)
    )
    return fig


def plot_attention_graph(G, pos, target_node, edge_index, attn_weights, title='GAT注意力可视化'):
    if G.number_of_nodes() == 0:
        return go.Figure()

    if pos is None:
        pos = compute_layout(G, layout='spring')

    target_neighbors = {}
    edge_list = list(G.edges())

    for i in range(edge_index.shape[1]):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        attn = attn_weights[i].mean() if attn_weights[i].ndim > 0 else float(attn_weights[i])
        if dst == target_node:
            if src not in target_neighbors:
                target_neighbors[src] = 0.0
            target_neighbors[src] += float(attn)
        elif src == target_node:
            if dst not in target_neighbors:
                target_neighbors[dst] = 0.0
            target_neighbors[dst] += float(attn)

    edge_traces = []
    max_attn = max(target_neighbors.values()) if target_neighbors else 1.0

    for (u, v) in edge_list:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        is_target_edge = (u == target_node and v in target_neighbors) or \
                         (v == target_node and u in target_neighbors)
        if is_target_edge:
            nb = v if u == target_node else u
            attn = target_neighbors.get(nb, 0) / max(max_attn, 1e-8)
            w = max(1, min(8, 1 + attn * 7))
            color = f'rgba({int(239 * (1 - attn) + 0 * attn)}, {int(85 * (1 - attn) + 204 * attn)}, ' \
                    f'{int(59 * (1 - attn) + 150 * attn)}, 0.9)'
        else:
            w = 0.8
            color = 'rgba(200, 200, 200, 0.3)'

        edge_traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines',
            line=dict(width=w, color=color),
            hoverinfo='none',
            showlegend=False
        ))

    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]

    node_colors = []
    node_sizes = []
    for n in G.nodes():
        if n == target_node:
            node_colors.append('#FF4D4D')
            node_sizes.append(24)
        elif n in target_neighbors:
            attn_norm = target_neighbors[n] / max(max_attn, 1e-8)
            node_colors.append(f'rgba(0, {int(180 + 50 * attn_norm)}, {int(100 + 155 * attn_norm)}, 1)')
            node_sizes.append(14 + int(attn_norm * 12))
        else:
            node_colors.append('rgba(180, 180, 180, 0.5)')
            node_sizes.append(8)

    text_labels = []
    for n in G.nodes():
        if n == target_node:
            text_labels.append(f'目标节点 {n}')
        elif n in target_neighbors:
            text_labels.append(f'节点{n}<br>注意力: {target_neighbors[n]:.4f}')
        else:
            text_labels.append(f'节点{n}')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(size=node_sizes, color=node_colors,
                    line_width=1.5, line_color='white'),
        hovertext=text_labels
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title=title,
        template='plotly_white',
        height=550,
        showlegend=False,
        hovermode='closest',
        margin=dict(l=5, r=5, t=50, b=5),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
    return fig


def plot_f1_per_class(f1_per_class, class_names=None):
    if f1_per_class is None or len(f1_per_class) == 0:
        return go.Figure()
    if class_names is None or len(class_names) != len(f1_per_class):
        class_names = [f'类{i}' for i in range(len(f1_per_class))]

    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
              '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

    fig = go.Figure(data=[go.Bar(
        x=class_names, y=f1_per_class,
        marker_color=[colors[i % len(colors)] for i in range(len(f1_per_class))],
        text=[f'{v:.3f}' for v in f1_per_class],
        textposition='auto',
    )])
    fig.update_layout(
        title='各类别F1得分',
        xaxis_title='类别',
        yaxis_title='F1 Score',
        template='plotly_white',
        yaxis_range=[0, 1.05],
        height=380,
        margin=dict(l=40, r=20, t=50, b=40)
    )
    return fig


def plot_comparison_bar(df, metric='test_acc', title='对比实验结果'):
    if df is None or len(df) == 0 or metric not in df.columns:
        return go.Figure()

    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
              '#19D3F3', '#FF6692', '#B6E880']

    fig = go.Figure(data=[go.Bar(
        x=df['model_name'], y=df[metric],
        marker_color=colors[:len(df)],
        text=[f'{v:.4f}' if isinstance(v, (int, float)) else str(v) for v in df[metric]],
        textposition='auto',
    )])
    fig.update_layout(
        title=title,
        xaxis_title='模型配置',
        yaxis_title=metric,
        template='plotly_white',
        height=420,
        margin=dict(l=40, r=20, t=50, b=80),
        xaxis_tickangle=-25
    )
    return fig
