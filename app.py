import os
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from torch_geometric.utils import to_networkx

from data_loader import (
    load_builtin_dataset, load_from_csv, load_from_graphml, load_from_gml,
    graph_statistics, split_data, get_largest_component, BUILTIN_DATASETS
)
from models import (
    create_model, count_parameters, MODEL_NAMES, ACTIVATIONS,
    HIDDEN_DIMS, NUM_LAYERS_RANGE, GAT_HEADS
)
from trainer import (
    train_model, extract_embeddings, extract_attention, reduce_embeddings
)
from community_detect import (
    run_community_detection, COMMUNITY_ALGORITHMS
)
from adversarial import (
    run_attack, evaluate_on_data, get_prediction_probs,
    batch_evaluate, get_perturbed_edge_info,
    degree_filter_defense, feature_smoothing_defense,
    ATTACK_METHODS, ATTACK_MODES, DEFENSE_METHODS
)
from visualizations import (
    compute_layout, plot_degree_distribution, plot_training_curves,
    plot_confusion_matrix, plot_graph, plot_embeddings_2d,
    plot_attention_graph, plot_f1_per_class, plot_comparison_bar,
    plot_confidence_bar, COLOR_THEMES
)

st.set_page_config(page_title='GNN实验平台 - 节点分类与社区发现',
                   layout='wide', page_icon='🧠')

st.title('🧠 图神经网络(GNN)实验平台')
st.markdown('#### 节点分类 · 社区发现 · 可解释可视化 · 对比实验')
st.markdown('---')


def init_session_state():
    defaults = {
        'data': None, 'G': None, 'num_classes': 0,
        'dataset_name': None, 'stats': None, 'pos': None,
        'model': None, 'results': None, 'embeddings': None,
        'class_names': None,
        'community_results': {},
        'comparison_experiments': [],
        'training_history': [],
        'selected_history_idx': None,
        'color_theme': 'Plotly默认',
        'graph_edited': False,
        'selected_node': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session_state()


def get_class_names(num_classes, dataset_name):
    if dataset_name == 'Cora':
        return ['Case_Based', 'Genetic_Algorithms', 'Neural_Networks',
                'Probabilistic_Methods', 'Reinforcement_Learning',
                'Rule_Learning', 'Theory']
    elif dataset_name == 'Citeseer':
        return ['AI', 'Agents', 'DB', 'IR', 'ML', 'HCI']
    elif dataset_name == 'Pubmed':
        return ['糖尿病', '缺血性心脏病', '肺癌']
    else:
        return [f'类{i}' for i in range(num_classes)]


def save_training_history(model_name, hyperparams, results, class_names):
    import copy
    record = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_name': model_name,
        'hyperparams': copy.deepcopy(hyperparams),
        'total_time': results['total_time'],
        'test_acc': results['test_acc'],
        'f1_macro': results['f1_macro'],
        'best_val_acc': results['best_val_acc'],
        'best_epoch': results['best_epoch'],
        'history': copy.deepcopy(results['history']),
        'confusion_matrix': results['confusion_matrix'].copy() if results['confusion_matrix'] is not None else None,
        'f1_per_class': results['f1_per_class'].copy() if results['f1_per_class'] is not None else None,
        'class_names': copy.deepcopy(class_names) if class_names else None,
    }
    history = st.session_state.training_history
    history.append(record)
    if len(history) > 20:
        min_idx = min(range(len(history)), key=lambda i: history[i]['test_acc'])
        history.pop(min_idx)
    st.session_state.training_history = history


def add_node_to_graph(feature_method='mean', custom_features=None):
    if st.session_state.G is None or st.session_state.data is None:
        return False, '请先加载数据集'

    G = st.session_state.G.copy()
    data = st.session_state.data.clone()
    new_node_id = G.number_of_nodes()
    G.add_node(new_node_id)

    feat_dim = data.x.shape[1] if data.x is not None else 1
    if feature_method == 'mean':
        if data.x is not None and data.x.shape[0] > 0:
            mean_feat = data.x.mean(dim=0).unsqueeze(0)
        else:
            mean_feat = torch.zeros(1, feat_dim)
        new_x = torch.cat([data.x, mean_feat], dim=0)
    elif feature_method == 'zero':
        new_x = torch.cat([data.x, torch.zeros(1, feat_dim)], dim=0)
    elif feature_method == 'custom' and custom_features is not None:
        custom_tensor = torch.tensor(custom_features, dtype=torch.float).unsqueeze(0)
        if custom_tensor.shape[1] != feat_dim:
            return False, f'特征维度不匹配，应为{feat_dim}维'
        new_x = torch.cat([data.x, custom_tensor], dim=0)
    else:
        new_x = torch.cat([data.x, torch.zeros(1, feat_dim)], dim=0)

    data.x = new_x
    data.num_nodes = new_node_id + 1

    if hasattr(data, 'y') and data.y is not None:
        new_y = torch.tensor([-1], dtype=torch.long)
        data.y = torch.cat([data.y, new_y], dim=0)

    for mask_name in ['train_mask', 'val_mask', 'test_mask']:
        if hasattr(data, mask_name) and getattr(data, mask_name) is not None:
            mask = getattr(data, mask_name)
            new_mask = torch.cat([mask, torch.zeros(1, dtype=torch.bool)], dim=0)
            setattr(data, mask_name, new_mask)

    edge_index = data.edge_index
    data.edge_index = edge_index

    st.session_state.G = G
    st.session_state.data = data
    st.session_state.stats = graph_statistics(G)
    st.session_state.pos = compute_layout(G, layout='spring', seed=42)

    if st.session_state.model is not None:
        st.session_state.graph_edited = True

    return True, f'已添加节点 {new_node_id}'


def add_edge_to_graph(u, v):
    if st.session_state.G is None or st.session_state.data is None:
        return False, '请先加载数据集'

    max_node = st.session_state.G.number_of_nodes() - 1
    if u < 0 or u > max_node or v < 0 or v > max_node:
        return False, f'节点ID超出范围，有效范围: 0 ~ {max_node}'
    if u == v:
        return False, '不能添加自环'
    if st.session_state.G.has_edge(u, v):
        return False, f'边 {u}-{v} 已存在'

    G = st.session_state.G.copy()
    G.add_edge(u, v)

    data = st.session_state.data.clone()
    edge_list = list(G.edges())
    edges_u = [e[0] for e in edge_list]
    edges_v = [e[1] for e in edge_list]
    edge_index = torch.tensor([edges_u + edges_v, edges_v + edges_u], dtype=torch.long)
    data.edge_index = edge_index

    st.session_state.G = G
    st.session_state.data = data
    st.session_state.stats = graph_statistics(G)

    if st.session_state.model is not None:
        st.session_state.graph_edited = True

    return True, f'已添加边 {u}-{v}'


def remove_edge_from_graph(u, v):
    if st.session_state.G is None or st.session_state.data is None:
        return False, '请先加载数据集'

    if not st.session_state.G.has_edge(u, v):
        return False, f'边 {u}-{v} 不存在'

    G = st.session_state.G.copy()
    G.remove_edge(u, v)

    data = st.session_state.data.clone()
    edge_list = list(G.edges())
    if len(edge_list) > 0:
        edges_u = [e[0] for e in edge_list]
        edges_v = [e[1] for e in edge_list]
        edge_index = torch.tensor([edges_u + edges_v, edges_v + edges_u], dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    data.edge_index = edge_index

    st.session_state.G = G
    st.session_state.data = data
    st.session_state.stats = graph_statistics(G)

    if st.session_state.model is not None:
        st.session_state.graph_edited = True

    return True, f'已删除边 {u}-{v}'


with st.sidebar:
    st.header('📊 1. 数据加载')
    data_source = st.radio('数据来源', ['内置数据集', '从文件导入'],
                           horizontal=True)

    if data_source == '内置数据集':
        builtin_choice = st.selectbox('选择数据集', BUILTIN_DATASETS)
        if st.button('📥 加载数据集', type='primary', use_container_width=True):
            with st.spinner('加载中...'):
                try:
                    data, G, num_classes = load_builtin_dataset(builtin_choice)
                    split_data(data, num_classes)
                    st.session_state.data = data
                    st.session_state.G = G
                    st.session_state.num_classes = num_classes
                    st.session_state.dataset_name = builtin_choice
                    st.session_state.stats = graph_statistics(G)
                    st.session_state.class_names = get_class_names(num_classes, builtin_choice)
                    st.session_state.pos = compute_layout(G, layout='spring', seed=42)
                    st.success(f'✅ 已加载 {builtin_choice}！')
                except Exception as e:
                    st.error(f'加载失败: {str(e)}')
    else:
        file_type = st.selectbox('文件格式', ['CSV (节点+边)', 'GraphML', 'GML'])
        if file_type == 'CSV (节点+边)':
            nodes_file = st.file_uploader('节点CSV (node_id, 特征列...)', type=['csv'])
            edges_file = st.file_uploader('边CSV (source, target, [weight])', type=['csv'])
            with st.expander('CSV列名配置'):
                node_id_col = st.text_input('节点ID列名', 'node_id')
                source_col = st.text_input('源节点列名', 'source')
                target_col = st.text_input('目标节点列名', 'target')
                weight_col = st.text_input('权重列名(可选)', '')
                label_col = st.text_input('标签列名(可选)', '')
        elif file_type in ['GraphML', 'GML']:
            graph_file = st.file_uploader(f'上传{file_type}文件',
                                          type=['graphml' if file_type == 'GraphML' else 'gml'])

        if file_type == 'CSV (节点+边)' and nodes_file and edges_file:
            if st.button('📥 导入CSV', type='primary', use_container_width=True):
                with st.spinner('导入中...'):
                    try:
                        nodes_content = nodes_file.getvalue().decode('utf-8')
                        edges_content = edges_file.getvalue().decode('utf-8')
                        wcol = weight_col if weight_col.strip() else None
                        lcol = label_col if label_col.strip() else None
                        data, G, num_classes = load_from_csv(
                            nodes_content, edges_content,
                            node_id_col=node_id_col, source_col=source_col,
                            target_col=target_col, weight_col=wcol, label_col=lcol
                        )
                        split_data(data, num_classes)
                        st.session_state.data = data
                        st.session_state.G = G
                        st.session_state.num_classes = num_classes
                        st.session_state.dataset_name = '自定义CSV'
                        st.session_state.stats = graph_statistics(G)
                        st.session_state.class_names = get_class_names(num_classes, '自定义')
                        st.session_state.pos = compute_layout(G, layout='spring', seed=42)
                        st.success('✅ 导入成功！')
                    except Exception as e:
                        st.error(f'导入失败: {str(e)}')
        elif file_type != 'CSV (节点+边)' and ('graph_file' in dir() and graph_file):
            if st.button(f'📥 导入{file_type}', type='primary', use_container_width=True):
                with st.spinner('导入中...'):
                    try:
                        content = graph_file.getvalue()
                        if file_type == 'GraphML':
                            data, G, num_classes = load_from_graphml(content)
                        else:
                            data, G, num_classes = load_from_gml(content)
                        split_data(data, num_classes)
                        st.session_state.data = data
                        st.session_state.G = G
                        st.session_state.num_classes = num_classes
                        st.session_state.dataset_name = f'自定义{file_type}'
                        st.session_state.stats = graph_statistics(G)
                        st.session_state.class_names = get_class_names(num_classes, '自定义')
                        st.session_state.pos = compute_layout(G, layout='spring', seed=42)
                        st.success('✅ 导入成功！')
                    except Exception as e:
                        st.error(f'导入失败: {str(e)}')

    st.markdown('---')
    st.header('⚙️ 2. 数据集划分')
    if st.session_state.data is not None:
        train_ratio = st.slider('训练集比例', 0.1, 0.9, 0.6, 0.05)
        val_ratio = st.slider('验证集比例', 0.05, 0.5, 0.2, 0.05)
        if st.button('🔀 重新划分', use_container_width=True):
            split_data(st.session_state.data, st.session_state.num_classes,
                       train_ratio=train_ratio, val_ratio=val_ratio)
            st.success('✅ 已重新划分！')

    st.markdown('---')
    st.header('📜 3. 训练历史')
    if len(st.session_state.training_history) == 0:
        st.info('暂无训练记录')
    else:
        history = st.session_state.training_history
        sort_by = st.selectbox('排序方式', ['时间(最新)', '测试精度(高→低)', 'F1分数(高→低)'])
        sorted_indices = list(range(len(history)))
        if sort_by == '测试精度(高→低)':
            sorted_indices.sort(key=lambda i: history[i]['test_acc'], reverse=True)
        elif sort_by == 'F1分数(高→低)':
            sorted_indices.sort(key=lambda i: history[i]['f1_macro'], reverse=True)
        else:
            sorted_indices.reverse()

        st.caption(f'共 {len(history)} 条记录（最多保存20条）')

        for rank, idx in enumerate(sorted_indices):
            rec = history[idx]
            is_selected = st.session_state.selected_history_idx == idx
            btn_label = f"{'▶ ' if is_selected else ''}{rec['model_name']} · {rec['timestamp'].split(' ')[1]}"
            btn_help = f"测试精度: {rec['test_acc']:.4f} | F1: {rec['f1_macro']:.4f} | 用时: {rec['total_time']:.1f}s"
            if st.button(btn_label, key=f'hist_{idx}', help=btn_help, use_container_width=True,
                         type='primary' if is_selected else 'secondary'):
                if st.session_state.selected_history_idx == idx:
                    st.session_state.selected_history_idx = None
                else:
                    st.session_state.selected_history_idx = idx

        if st.button('🗑️ 清空历史', use_container_width=True):
            st.session_state.training_history = []
            st.session_state.selected_history_idx = None
            st.rerun()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    '📊 数据概览', '🧠 GNN训练', '🔍 社区发现',
    '📈 可视化分析', '⚡ 对比实验', '🛡️ 对抗鲁棒性'
])

with tab1:
    st.subheader('📊 数据基本统计')
    if st.session_state.stats is None:
        st.info('请先在左侧加载或导入数据集')
    else:
        s = st.session_state.stats
        c1, c2, c3 = st.columns(3)
        c1.metric('节点数', s['num_nodes'])
        c1.metric('边数', s['num_edges'])
        c2.metric('平均度', s['avg_degree'])
        c2.metric('聚类系数', s['clustering_coeff'])
        c3.metric('连通分量数', s['num_components'])
        if st.session_state.num_classes > 0:
            c3.metric('类别数', st.session_state.num_classes)

        if st.session_state.data is not None:
            st.markdown('#### 节点信息')
            d = st.session_state.data
            info_c1, info_c2, info_c3 = st.columns(3)
            info_c1.info(f'**特征维度**: {d.x.shape[1] if d.x is not None else 0}')
            if hasattr(d, 'train_mask') and d.train_mask is not None:
                train_n = int(d.train_mask.sum().item())
                val_n = int(d.val_mask.sum().item()) if hasattr(d, 'val_mask') else 0
                test_n = int(d.test_mask.sum().item()) if hasattr(d, 'test_mask') else 0
                info_c2.info(f'**Train/Val/Test**: {train_n}/{val_n}/{test_n}')
            info_c3.info(f'**是否有标签**: {"是" if st.session_state.num_classes > 0 else "否"}')

        st.plotly_chart(plot_degree_distribution(s['degree_dist']), use_container_width=True)

        if st.session_state.graph_edited and st.session_state.model is not None:
            st.warning('⚠️ 图结构已修改，现有模型可能不再适用，请重新训练模型')

        with st.expander('✏️ 手动编辑图结构', expanded=False):
            edit_tabs = st.tabs(['添加节点', '添加边', '删除边'])

            with edit_tabs[0]:
                st.markdown('##### 添加新节点')
                feat_method = st.radio('特征初始化方式',
                                       ['均值填充', '零填充', '自定义特征'],
                                       horizontal=True, key='add_node_feat')
                custom_feat_input = None
                if feat_method == '自定义特征':
                    feat_dim = st.session_state.data.x.shape[1] if st.session_state.data.x is not None else 1
                    custom_feat_input = st.text_input(
                        f'输入特征向量（共{feat_dim}维，用逗号分隔）',
                        placeholder=f'例如: {",".join(["0.1"]*min(5, feat_dim))}...',
                        key='custom_feat'
                    )
                if st.button('➕ 添加节点', use_container_width=True, key='btn_add_node'):
                    method_map = {'均值填充': 'mean', '零填充': 'zero', '自定义特征': 'custom'}
                    method = method_map[feat_method]
                    custom_feats = None
                    if method == 'custom' and custom_feat_input:
                        try:
                            custom_feats = [float(x.strip()) for x in custom_feat_input.split(',')]
                        except ValueError:
                            st.error('请输入有效的数字，用逗号分隔')
                            custom_feats = None
                    success, msg = add_node_to_graph(feature_method=method, custom_features=custom_feats)
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

            with edit_tabs[1]:
                st.markdown('##### 添加边')
                max_node = st.session_state.G.number_of_nodes() - 1 if st.session_state.G else 0
                ae_c1, ae_c2 = st.columns(2)
                add_u = ae_c1.number_input('源节点ID', 0, max(0, max_node), 0, 1, key='add_edge_u')
                add_v = ae_c2.number_input('目标节点ID', 0, max(0, max_node), 1, 1, key='add_edge_v')
                if st.button('➕ 添加边', use_container_width=True, key='btn_add_edge'):
                    success, msg = add_edge_to_graph(int(add_u), int(add_v))
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

            with edit_tabs[2]:
                st.markdown('##### 删除边')
                max_node = st.session_state.G.number_of_nodes() - 1 if st.session_state.G else 0
                re_c1, re_c2 = st.columns(2)
                rm_u = re_c1.number_input('源节点ID', 0, max(0, max_node), 0, 1, key='rm_edge_u')
                rm_v = re_c2.number_input('目标节点ID', 0, max(0, max_node), 1, 1, key='rm_edge_v')
                if st.button('🗑️ 删除边', use_container_width=True, key='btn_rm_edge'):
                    success, msg = remove_edge_from_graph(int(rm_u), int(rm_v))
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

            st.caption('💡 提示：编辑图结构后，统计信息和可视化会自动更新')

        st.markdown('#### 图结构预览')
        viz_c1, viz_c2 = st.columns([1, 2])
        with viz_c1:
            layout_choice = st.selectbox('布局算法', ['spring', 'fruchterman_reingold',
                                                        'kamada_kawai', 'circular', 'spectral'])
            show_labels = st.checkbox('显示节点标签')
            display_mode = st.radio('显示模式', ['全部节点', '最大连通分量'],
                                    horizontal=True)
            if st.button('🔄 重算布局', use_container_width=True):
                G_viz = st.session_state.G
                if display_mode == '最大连通分量' and s['num_nodes'] > 500:
                    G_viz = get_largest_component(G_viz)
                st.session_state.pos = compute_layout(G_viz, layout=layout_choice, seed=42)
                st.success('✅ 布局已更新')

            st.markdown('---')
            st.markdown('##### 🔍 节点搜索')
            max_node = s['num_nodes'] - 1
            search_node = st.number_input('输入节点ID', 0, max_node, 0, 1, key='search_node_input')
            sc1, sc2 = st.columns(2)
            if sc1.button('📍 定位', use_container_width=True, key='btn_locate_node'):
                st.session_state.selected_node = int(search_node)
            if sc2.button('❌ 取消', use_container_width=True, key='btn_clear_node'):
                st.session_state.selected_node = None

            color_theme_choice = st.selectbox(
                '配色方案',
                list(COLOR_THEMES.keys()),
                index=list(COLOR_THEMES.keys()).index(st.session_state.color_theme)
                if st.session_state.color_theme in COLOR_THEMES else 0,
                key='color_theme_viz'
            )
            if color_theme_choice != st.session_state.color_theme:
                st.session_state.color_theme = color_theme_choice
                st.rerun()

        with viz_c2:
            G_plot = st.session_state.G
            if display_mode == '最大连通分量' and s['num_nodes'] > 500:
                G_plot = get_largest_component(G_plot)

            has_labels = st.session_state.num_classes > 0
            colors = None
            if has_labels and st.session_state.data.y is not None:
                colors = st.session_state.data.y.cpu().numpy().tolist()

            highlight = st.session_state.selected_node
            if highlight is not None and highlight >= G_plot.number_of_nodes():
                highlight = None

            fig = plot_graph(G_plot, pos=st.session_state.pos,
                             node_colors=colors if has_labels else None,
                             class_names=st.session_state.class_names,
                             show_labels=show_labels,
                             highlight_node=highlight,
                             color_theme=st.session_state.color_theme,
                             title='图可视化（按真实类别着色）' if has_labels else '图可视化')
            st.plotly_chart(fig, use_container_width=True)

            if st.session_state.selected_node is not None:
                node_id = st.session_state.selected_node
                if node_id < s['num_nodes']:
                    st.markdown('---')
                    st.markdown(f'##### 📋 节点详情 · ID: {node_id}')
                    degree = st.session_state.G.degree(node_id) if node_id in st.session_state.G.nodes else 0

                    det_c1, det_c2 = st.columns(2)
                    det_c1.metric('度数', degree)

                    community_label = '-'
                    if len(st.session_state.community_results) > 0:
                        first_alg = list(st.session_state.community_results.keys())[0]
                        labels = st.session_state.community_results[first_alg]['labels']
                        if node_id < len(labels):
                            community_label = int(labels[node_id])
                    det_c2.metric('所属社区', community_label)

                    if st.session_state.num_classes > 0 and st.session_state.data.y is not None:
                        true_label = st.session_state.data.y[node_id].item()
                        true_label_name = st.session_state.class_names[true_label] \
                            if st.session_state.class_names and true_label < len(st.session_state.class_names) \
                            else f'类{true_label}'
                        det_c1.metric('真实标签', true_label_name)

                    if st.session_state.model is not None and not st.session_state.graph_edited:
                        try:
                            device = next(st.session_state.model.parameters()).device
                            st.session_state.model.eval()
                            with torch.no_grad():
                                out = st.session_state.model(
                                    st.session_state.data.x.to(device),
                                    st.session_state.data.edge_index.to(device)
                                )
                                probs = torch.softmax(out, dim=1)[node_id].cpu().numpy()
                                pred_label = int(probs.argmax())
                                pred_conf = float(probs.max())
                                pred_label_name = st.session_state.class_names[pred_label] \
                                    if st.session_state.class_names and pred_label < len(st.session_state.class_names) \
                                    else f'类{pred_label}'
                                det_c2.metric('预测标签', f'{pred_label_name}')
                                st.plotly_chart(
                                    plot_confidence_bar(
                                        probs,
                                        st.session_state.class_names,
                                        color_theme=st.session_state.color_theme
                                    ),
                                    use_container_width=True
                                )
                        except Exception as e:
                            st.info(f'无法获取预测: {str(e)}')

                    st.markdown('##### 👥 一阶邻居')
                    neighbors = list(st.session_state.G.neighbors(node_id)) if node_id in st.session_state.G.nodes else []
                    if len(neighbors) == 0:
                        st.info('该节点没有邻居')
                    else:
                        st.caption(f'共 {len(neighbors)} 个邻居')
                        nb_cols = st.columns(min(5, len(neighbors)))
                        for i, nb in enumerate(neighbors[:10]):
                            with nb_cols[i % 5]:
                                if st.button(f'🔗 {nb}', key=f'nb_{node_id}_{nb}', use_container_width=True):
                                    st.session_state.selected_node = int(nb)
                                    st.rerun()
                        if len(neighbors) > 10:
                            st.caption(f'还有 {len(neighbors) - 10} 个邻居未显示')

with tab2:
    st.subheader('🧠 GNN模型训练与评估')
    if st.session_state.data is None:
        st.info('请先在左侧加载或导入数据集')
    elif st.session_state.num_classes == 0:
        st.warning('⚠️ 当前数据集无标签，无法进行节点分类训练')
    else:
        mc1, mc2 = st.columns(2)
        with mc1:
            model_name = st.selectbox('GNN架构', MODEL_NAMES, key='tab2_model')
            num_layers = st.select_slider('层数', options=NUM_LAYERS_RANGE, value=2,
                                          key='tab2_layers')
            hidden_dim = st.selectbox('隐藏维度', HIDDEN_DIMS, index=2, key='tab2_hd')
            activation = st.selectbox('激活函数', ACTIVATIONS, key='tab2_act')
        with mc2:
            dropout = st.slider('Dropout率', 0.0, 0.8, 0.5, 0.05, key='tab2_drop')
            lr = st.slider('学习率', 0.001, 0.1, 0.01, 0.001, format='%.4f',
                           key='tab2_lr')
            weight_decay = st.slider('权重衰减(L2)', 1e-5, 1e-2, 5e-4, 1e-5,
                                     format='%.5f', key='tab2_wd')
            max_epochs = st.select_slider('最大Epoch', options=[50, 100, 200, 300, 500],
                                          value=200, key='tab2_epochs')
            patience = st.slider('早停Patience', 5, 100, 50, 5, key='tab2_patience')

        extra_kwargs = {}
        if model_name == 'GAT':
            st.markdown('##### GAT专属配置')
            gc1, gc2 = st.columns(2)
            extra_kwargs['heads'] = gc1.selectbox('注意力头数', GAT_HEADS, index=2)
            extra_kwargs['concat_last'] = gc2.selectbox('最后一层', ['平均', '拼接'],
                                                        index=0) == '拼接'
        elif model_name == 'GraphSAGE':
            st.markdown('##### GraphSAGE专属配置')
            aggr_choice = st.selectbox('邻居聚合器', ['mean', 'max'])
            extra_kwargs['aggr'] = aggr_choice
        elif model_name == 'GIN':
            st.markdown('##### GIN专属配置')
            extra_kwargs['train_eps'] = st.checkbox('学习ε参数', value=True)

        if st.button('🚀 开始训练', type='primary', use_container_width=True):
            in_channels = st.session_state.data.x.shape[1]
            model = create_model(model_name, in_channels, st.session_state.num_classes,
                                 hidden_channels=hidden_dim, num_layers=num_layers,
                                 activation=activation, dropout=dropout, **extra_kwargs)
            n_params = count_parameters(model)
            st.info(f'模型参数量: **{n_params:,}**')

            progress_bar = st.progress(0)
            status_text = st.empty()
            train_loss_chart = st.empty()
            train_acc_chart = st.empty()

            history_data = {'epoch': [], 'train_loss': [], 'train_acc': [],
                            'val_loss': [], 'val_acc': []}

            def progress_cb(epoch, max_epoch, tl, ta, vl, va, bva, be):
                pct = min(epoch / max_epoch, 1.0)
                progress_bar.progress(pct)
                status_text.markdown(
                    f'**Epoch {epoch}/{max_epoch}** | '
                    f'Train Loss: {tl:.4f} | Train Acc: {ta:.4f} | '
                    f'Val Acc: {va:.4f} | Best Val Acc: {bva:.4f} @Epoch{be}'
                )

            with st.spinner('训练中...'):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                trained_model, results = train_model(
                    model, st.session_state.data,
                    lr=lr, weight_decay=weight_decay,
                    max_epochs=max_epochs, patience=patience,
                    device=device, verbose=False,
                    progress_callback=progress_cb
                )

            progress_bar.progress(1.0)
            st.session_state.model = trained_model
            st.session_state.results = results
            st.session_state.embeddings = extract_embeddings(
                trained_model, st.session_state.data, device=device, layer=-2
            ) if num_layers > 1 else extract_embeddings(
                trained_model, st.session_state.data, device=device, layer=-1
            )

            hyperparams = {
                'num_layers': num_layers,
                'hidden_dim': hidden_dim,
                'activation': activation,
                'dropout': dropout,
                'lr': lr,
                'weight_decay': weight_decay,
                'max_epochs': max_epochs,
                'patience': patience,
            }
            hyperparams.update(extra_kwargs)
            save_training_history(model_name, hyperparams, results, st.session_state.class_names)
            st.session_state.graph_edited = False

            st.success(f'✅ 训练完成！用时 {results["total_time"]:.2f}s · 已保存到训练历史')

        if st.session_state.selected_history_idx is not None:
            rec = st.session_state.training_history[st.session_state.selected_history_idx]
            st.markdown('---')
            st.subheader(f'📜 历史记录回溯 · {rec["model_name"]}')
            st.caption(f'训练时间: {rec["timestamp"]}')

            hr1, hr2, hr3, hr4 = st.columns(4)
            hr1.metric('最佳验证精度', f'{rec["best_val_acc"]:.4f}',
                       f'Epoch {rec["best_epoch"]}')
            hr2.metric('测试集精度', f'{rec["test_acc"]:.4f}')
            hr3.metric('Macro F1', f'{rec["f1_macro"]:.4f}')
            hr4.metric('训练用时', f'{rec["total_time"]:.1f}s')

            with st.expander('🔧 超参数配置', expanded=False):
                hp = rec['hyperparams']
                hp_cols = st.columns(3)
                for i, (k, v) in enumerate(hp.items()):
                    hp_cols[i % 3].info(f'**{k}**: {v}')

            st.plotly_chart(plot_training_curves(rec['history']), use_container_width=True)

            hpc1, hpc2 = st.columns(2)
            with hpc1:
                if rec['f1_per_class'] is not None:
                    st.plotly_chart(plot_f1_per_class(
                        rec['f1_per_class'], rec['class_names']
                    ), use_container_width=True)
            with hpc2:
                if rec['confusion_matrix'] is not None:
                    st.plotly_chart(plot_confusion_matrix(
                        rec['confusion_matrix'], rec['class_names']
                    ), use_container_width=True)

        if st.session_state.results is not None and st.session_state.selected_history_idx is None:
            r = st.session_state.results
            st.markdown('---')
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric('最佳验证精度', f'{r["best_val_acc"]:.4f}',
                       f'Epoch {r["best_epoch"]}')
            rc2.metric('测试集精度', f'{r["test_acc"]:.4f}')
            rc3.metric('Macro F1', f'{r["f1_macro"]:.4f}')
            rc4.metric('训练用时', f'{r["total_time"]:.1f}s')

            st.plotly_chart(plot_training_curves(r['history']), use_container_width=True)

            pc1, pc2 = st.columns(2)
            with pc1:
                st.plotly_chart(plot_f1_per_class(
                    r['f1_per_class'], st.session_state.class_names
                ), use_container_width=True)
            with pc2:
                st.plotly_chart(plot_confusion_matrix(
                    r['confusion_matrix'], st.session_state.class_names
                ), use_container_width=True)

            st.markdown('---')
            st.subheader('📤 嵌入导出')
            ec1, ec2 = st.columns(2)
            with ec1:
                if st.session_state.embeddings is not None:
                    emb_df = pd.DataFrame(st.session_state.embeddings)
                    emb_df.insert(0, 'node_id', range(len(emb_df)))
                    csv = emb_df.to_csv(index=False).encode('utf-8')
                    st.download_button('💾 导出节点嵌入(CSV)', csv,
                                       'node_embeddings.csv', 'text/csv',
                                       use_container_width=True)
            with ec2:
                if st.session_state.embeddings is not None:
                    with st.spinner('降维中...'):
                        reduced = reduce_embeddings(st.session_state.embeddings, method='tsne')
                        red_df = pd.DataFrame(reduced, columns=['x', 'y'])
                        red_df.insert(0, 'node_id', range(len(red_df)))
                        if st.session_state.data.y is not None:
                            red_df['label'] = st.session_state.data.y.cpu().numpy()
                        csv2 = red_df.to_csv(index=False).encode('utf-8')
                        st.download_button('💾 导出2D降维坐标(CSV)', csv2,
                                           'embeddings_2d.csv', 'text/csv',
                                           use_container_width=True)

with tab3:
    st.subheader('🔍 社区发现')
    if st.session_state.G is None:
        st.info('请先在左侧加载或导入数据集')
    else:
        algos = list(COMMUNITY_ALGORITHMS.keys())
        selected_algos = st.multiselect('选择社区发现算法', algos, default=algos[:2])

        spec_k = None
        gnn_k = None
        need_gnn = 'GNN Embedding + K-Means' in selected_algos
        if 'Spectral Clustering' in selected_algos:
            default_k = max(2, st.session_state.num_classes) if st.session_state.num_classes > 0 else 2
            spec_k = st.slider('谱聚类 - 社区数k', 2, 20, default_k)
        if need_gnn:
            if st.session_state.embeddings is None:
                st.warning('⚠️ 请先在"GNN训练"页训练模型以获取嵌入')
                gnn_k = None
            else:
                default_k = max(2, st.session_state.num_classes) if st.session_state.num_classes > 0 else 2
                gnn_k = st.slider('GNN+KMeans - 社区数k', 2, 20, default_k)

        if st.button('🔍 开始社区发现', type='primary', use_container_width=True):
            results_dict = {}
            for alg in selected_algos:
                with st.spinner(f'运行 {alg}...'):
                    try:
                        k_arg = None
                        if alg == 'Spectral Clustering':
                            k_arg = spec_k
                        elif alg == 'GNN Embedding + K-Means':
                            k_arg = gnn_k
                        labels, mod, n_comm = run_community_detection(
                            alg, st.session_state.G, k=k_arg,
                            embeddings=st.session_state.embeddings
                        )
                        results_dict[alg] = {
                            'labels': labels, 'modularity': mod,
                            'num_communities': n_comm
                        }
                    except Exception as e:
                        st.error(f'{alg} 出错: {str(e)}')
            st.session_state.community_results = results_dict
            st.success('✅ 社区发现完成！')

        if len(st.session_state.community_results) > 0:
            st.markdown('#### 社区结果统计')
            summary_rows = []
            for alg, res in st.session_state.community_results.items():
                summary_rows.append({
                    '算法': alg,
                    '社区数量': res['num_communities'],
                    '模块度 Q': round(res['modularity'], 4) if res['modularity'] else '-'
                })
            st.table(pd.DataFrame(summary_rows))

            st.markdown('---')
            st.markdown('#### 社区可视化对比')
            display_cols = st.columns(min(len(st.session_state.community_results), 2))
            for idx, (alg, res) in enumerate(st.session_state.community_results.items()):
                with display_cols[idx % len(display_cols)]:
                    colors = res['labels'].tolist()
                    fig = plot_graph(
                        st.session_state.G, pos=st.session_state.pos,
                        node_colors=colors,
                        color_theme=st.session_state.color_theme,
                        title=f'{alg} 社区划分'
                    )
                    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.subheader('📈 深度可视化分析')
    if st.session_state.data is None:
        st.info('请先加载数据集并训练模型')
    else:
        va_tabs = st.tabs([
            '🎯 标签 vs 社区对比',
            '🧩 嵌入降维(t-SNE/UMAP)',
            '👁️ GAT注意力可视化'
        ])

        with va_tabs[0]:
            st.markdown('#### 真实标签与社区划分对比')
            has_labels = st.session_state.num_classes > 0
            has_communities = len(st.session_state.community_results) > 0
            if not has_labels and not has_communities:
                st.info('请先训练模型或运行社区发现')
            else:
                comp_c1, comp_c2 = st.columns(2)
                with comp_c1:
                    colors = None
                    if has_labels and st.session_state.data.y is not None:
                        colors = st.session_state.data.y.cpu().numpy().tolist()
                    fig1 = plot_graph(
                        st.session_state.G, pos=st.session_state.pos,
                        node_colors=colors, class_names=st.session_state.class_names,
                        color_theme=st.session_state.color_theme,
                        title='真实类别标签' if has_labels else '图结构'
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                with comp_c2:
                    if has_communities:
                        alg_options = list(st.session_state.community_results.keys())
                        chosen_alg = st.selectbox('选择社区算法', alg_options,
                                                  key='viz_compare_alg')
                        if chosen_alg:
                            res = st.session_state.community_results[chosen_alg]
                            fig2 = plot_graph(
                                st.session_state.G, pos=st.session_state.pos,
                                node_colors=res['labels'].tolist(),
                                color_theme=st.session_state.color_theme,
                                title=f'{chosen_alg} 社区划分'
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.info('请先运行社区发现')

        with va_tabs[1]:
            st.markdown('#### 节点嵌入降维可视化')
            if st.session_state.embeddings is None:
                st.info('请先在GNN训练页训练模型')
            else:
                dr_c1, dr_c2 = st.columns([1, 3])
                with dr_c1:
                    dr_method = st.radio('降维方法', ['t-SNE', 'UMAP'], key='dr_method')
                    if st.button('🔄 重新降维', use_container_width=True):
                        with st.spinner('降维中...'):
                            reduced = reduce_embeddings(
                                st.session_state.embeddings,
                                method=dr_method.lower()
                            )
                            st.session_state.reduced_embeddings = reduced

                if 'reduced_embeddings' not in st.session_state or \
                        st.session_state.reduced_embeddings is None:
                    with st.spinner('初次降维中...'):
                        reduced = reduce_embeddings(
                            st.session_state.embeddings, method='tsne'
                        )
                        st.session_state.reduced_embeddings = reduced

                reduced = st.session_state.reduced_embeddings
                labels = None
                if st.session_state.data.y is not None:
                    labels = st.session_state.data.y.cpu().numpy()
                fig = plot_embeddings_2d(
                    reduced, labels=labels,
                    title=f'节点嵌入 {dr_method} 降维可视化',
                    class_names=st.session_state.class_names,
                    color_theme=st.session_state.color_theme
                )
                st.plotly_chart(fig, use_container_width=True)

        with va_tabs[2]:
            st.markdown('#### GAT注意力权重可视化')
            is_gat = st.session_state.model is not None and \
                     hasattr(st.session_state.model, 'get_attention')
            if not is_gat:
                st.info('⚠️ 请先使用GAT架构训练模型')
            else:
                att_c1, att_c2 = st.columns([1, 3])
                with att_c1:
                    max_n = st.session_state.G.number_of_nodes() - 1
                    target_node = st.number_input('目标节点ID', 0, max_n, 0, 1)
                    attn_layer = st.slider('注意力层', 0,
                                           st.session_state.model.num_layers - 1, 0)
                with att_c2:
                    with st.spinner('计算注意力权重...'):
                        try:
                            device = next(st.session_state.model.parameters()).device
                            edge_idx, attn = extract_attention(
                                st.session_state.model, st.session_state.data,
                                device=device, layer=attn_layer
                            )
                            fig = plot_attention_graph(
                                st.session_state.G, st.session_state.pos,
                                target_node=target_node,
                                edge_index=edge_idx, attn_weights=attn,
                                title=f'节点 {target_node} 的邻居注意力权重'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f'可视化失败: {str(e)}')

with tab5:
    st.subheader('⚡ 对比实验')
    if st.session_state.data is None:
        st.info('请先加载数据集')
    elif st.session_state.num_classes == 0:
        st.warning('⚠️ 当前数据集无标签，无法进行对比实验')
    else:
        st.markdown('##### 配置多个实验')
        num_exps = st.slider('实验数量', 2, 6, 2, 1, key='num_exps')

        exps_config = []
        for i in range(num_exps):
            with st.expander(f'🧪 实验{i+1} 配置', expanded=(i == 0)):
                ec1, ec2, ec3, ec4 = st.columns(4)
                mname = ec1.selectbox('架构', MODEL_NAMES, index=i % 4, key=f'exp{i}_m')
                nlayers = ec2.select_slider('层数', NUM_LAYERS_RANGE, value=2, key=f'exp{i}_l')
                hdim = ec3.selectbox('隐层', HIDDEN_DIMS, index=2, key=f'exp{i}_h')
                lrate = ec4.slider('学习率', 0.001, 0.1, 0.01, 0.001, format='%.4f', key=f'exp{i}_lr')

                ec5, ec6 = st.columns(2)
                do = ec5.slider('Dropout', 0.0, 0.8, 0.5, 0.05, key=f'exp{i}_d')
                act = ec6.selectbox('激活', ACTIVATIONS, key=f'exp{i}_a')

                extra_kw = {}
                if mname == 'GAT':
                    eg1, eg2 = st.columns(2)
                    extra_kw['heads'] = eg1.selectbox('头数', GAT_HEADS, index=2, key=f'exp{i}_head')
                    extra_kw['concat_last'] = eg2.selectbox(
                        '最后', ['平均', '拼接'], index=0, key=f'exp{i}_cl'
                    ) == '拼接'
                elif mname == 'GraphSAGE':
                    extra_kw['aggr'] = st.selectbox('聚合器', ['mean', 'max'], key=f'exp{i}_ag')

                exps_config.append({
                    'idx': i,
                    'name': f'{mname}-L{nlayers}-H{hdim}',
                    'model_name': mname,
                    'num_layers': nlayers,
                    'hidden_dim': hdim,
                    'dropout': do,
                    'activation': act,
                    'lr': lrate,
                    'extra': extra_kw
                })

        train_common_c1, train_common_c2, train_common_c3 = st.columns(3)
        max_epochs_common = train_common_c1.select_slider(
            '最大Epoch', [50, 100, 200, 300], value=150, key='comp_epochs'
        )
        patience_common = train_common_c2.slider('早停Patience', 5, 100, 30, 5, key='comp_pat')
        weight_decay_common = train_common_c3.slider(
            '权重衰减', 1e-5, 1e-2, 5e-4, 1e-5, format='%.5f', key='comp_wd'
        )

        feature_mode = st.radio('节点特征模式', ['原始特征', '单位矩阵(Identity)'],
                                horizontal=True)

        if st.button('⚡ 一键开始对比实验', type='primary', use_container_width=True):
            all_results = []
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            data_mod = st.session_state.data.clone()
            if feature_mode == '单位矩阵(Identity)':
                data_mod.x = torch.eye(data_mod.num_nodes)

            overall_bar = st.progress(0)
            overall_status = st.empty()

            for idx, cfg in enumerate(exps_config):
                overall_status.markdown(f'### 🧪 实验 {idx+1}/{num_exps}: {cfg["name"]}')
                exp_bar = st.progress(0)
                exp_status = st.empty()

                in_ch = data_mod.x.shape[1]
                model = create_model(
                    cfg['model_name'], in_ch, st.session_state.num_classes,
                    hidden_channels=cfg['hidden_dim'], num_layers=cfg['num_layers'],
                    activation=cfg['activation'], dropout=cfg['dropout'],
                    **cfg['extra']
                )
                n_params = count_parameters(model)

                def make_cb(exp_idx):
                    def _cb(epoch, max_ep, tl, ta, vl, va, bva, be):
                        pct = epoch / max_ep
                        exp_bar.progress(min(pct, 1.0))
                        exp_status.markdown(
                            f'Epoch {epoch}/{max_ep} | Val Acc: {va:.4f} | Best: {bva:.4f}'
                        )
                    return _cb

                _, results = train_model(
                    model, data_mod, lr=cfg['lr'], weight_decay=weight_decay_common,
                    max_epochs=max_epochs_common, patience=patience_common,
                    device=device, verbose=False,
                    progress_callback=make_cb(idx)
                )

                row = {
                    'model_name': cfg['name'],
                    '架构': cfg['model_name'],
                    '层数': cfg['num_layers'],
                    '隐层维度': cfg['hidden_dim'],
                    '参数量': n_params,
                    '训练时间(s)': round(results['total_time'], 2),
                    '测试精度': round(results['test_acc'], 4),
                    'Macro F1': round(results['f1_macro'], 4),
                    '最佳Val精度': round(results['best_val_acc'], 4),
                }
                if st.session_state.class_names:
                    for ci, cn in enumerate(st.session_state.class_names):
                        if ci < len(results['f1_per_class']):
                            row[f'F1_{cn}'] = round(results['f1_per_class'][ci], 4)
                all_results.append(row)

                exp_bar.progress(1.0)
                overall_bar.progress((idx + 1) / num_exps)

            overall_status.markdown('### ✅ 全部实验完成！')
            st.session_state.comparison_experiments = all_results

        if len(st.session_state.comparison_experiments) > 0:
            st.markdown('---')
            st.markdown('#### 📋 对比结果表格')
            df = pd.DataFrame(st.session_state.comparison_experiments)
            st.dataframe(df.set_index('model_name'), use_container_width=True)

            bc1, bc2 = st.columns(2)
            with bc1:
                metric_choice = st.selectbox(
                    '对比指标',
                    ['测试精度', 'Macro F1', '训练时间(s)', '参数量', '最佳Val精度'],
                    key='comp_metric'
                )
                st.plotly_chart(
                    plot_comparison_bar(df, metric=metric_choice,
                                        title=f'各模型{metric_choice}对比'),
                    use_container_width=True
                )
            with bc2:
                layer_unique = df['层数'].nunique() > 1
                if layer_unique:
                    layer_df = df.groupby('层数')['测试精度'].mean().reset_index()
                    st.plotly_chart(
                        plot_comparison_bar(layer_df, metric='测试精度',
                                            title='层数 vs 平均精度（观察过平滑现象）'),
                        use_container_width=True
                    )
                else:
                    st.info('💡 提示：配置不同层数的实验可观察过平滑现象')

with tab6:
    st.subheader('🛡️ 图对抗鲁棒性分析')
    if st.session_state.data is None:
        st.info('请先在左侧加载数据集')
    elif st.session_state.num_classes == 0:
        st.warning('⚠️ 当前数据集无标签，无法进行鲁棒性评估')
    elif len(st.session_state.training_history) == 0 and st.session_state.model is None:
        st.info('请先在"GNN训练"页训练模型')
    else:
        available_models = []
        if len(st.session_state.training_history) > 0:
            for i, rec in enumerate(st.session_state.training_history):
                available_models.append(f'{rec["model_name"]} · {rec["timestamp"]} (精度:{rec["test_acc"]:.4f})')
        if st.session_state.model is not None and st.session_state.results is not None:
            available_models.insert(0, '当前模型')

        adv_tabs = st.tabs(['🎯 单次攻击评估', '📊 批量评估', '⚔️ 防御策略对比', '👁️ 攻击可视化', '💾 结果导出'])

        with adv_tabs[0]:
            st.markdown('#### 单次攻击评估')
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                selected_model_label = st.selectbox('选择模型', available_models, key='adv_model')
                attack_method = st.selectbox('攻击方法', list(ATTACK_METHODS.keys()), key='adv_method')
            with sc2:
                attack_ratio = st.slider('攻击比例(%)', 1, 50, 10, 1, key='adv_ratio',
                                         format='%d%%')
                attack_mode = st.selectbox('攻击模式', list(ATTACK_MODES.keys()), key='adv_mode')
            with sc3:
                enable_defense = st.checkbox('启用防御', value=False, key='adv_enable_def')
                defense_method = None
                if enable_defense:
                    defense_method = st.selectbox('防御方法', list(DEFENSE_METHODS.keys()), key='adv_defense')

            if st.button('🚀 执行攻击', type='primary', use_container_width=True, key='btn_adv_attack'):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = st.session_state.model

                data = st.session_state.data
                ratio = attack_ratio / 100.0

                with st.spinner('执行攻击中...'):
                    attacked_data = run_attack(
                        data, model, attack_method, ratio, attack_mode, device=device
                    )

                acc_before, f1_before = evaluate_on_data(model, data, device)
                acc_after, f1_after = evaluate_on_data(model, attacked_data, device)

                acc_def = None
                f1_def = None
                if enable_defense and defense_method is not None:
                    defense_key = DEFENSE_METHODS.get(defense_method, defense_method)
                    if defense_key == 'degree_filter':
                        defended_data = degree_filter_defense(data, attacked_data)
                    else:
                        defended_data = feature_smoothing_defense(attacked_data)
                    acc_def, f1_def = evaluate_on_data(model, defended_data, device)

                st.session_state.adv_last_result = {
                    'attack_method': attack_method,
                    'attack_ratio': attack_ratio,
                    'attack_mode': attack_mode,
                    'acc_before': acc_before,
                    'acc_after': acc_after,
                    'acc_drop': acc_before - acc_after,
                    'f1_before': f1_before,
                    'f1_after': f1_after,
                    'f1_drop': f1_before - f1_after,
                    'defense_method': defense_method if enable_defense else None,
                    'acc_after_defense': acc_def,
                    'f1_after_defense': f1_def,
                    'attacked_data': attacked_data,
                }

                st.success('✅ 攻击评估完成！')

            if 'adv_last_result' in st.session_state and st.session_state.adv_last_result is not None:
                res = st.session_state.adv_last_result
                st.markdown('---')
                st.markdown('#### 📊 攻击结果')

                rc1, rc2, rc3, rc4 = st.columns(4)
                rc1.metric('攻击前精度', f'{res["acc_before"]:.4f}')
                rc2.metric('攻击后精度', f'{res["acc_after"]:.4f}',
                           f'↓ {res["acc_drop"]:.4f}', delta_color='inverse')
                rc3.metric('攻击前F1', f'{res["f1_before"]:.4f}')
                rc4.metric('攻击后F1', f'{res["f1_after"]:.4f}',
                           f'↓ {res["f1_drop"]:.4f}', delta_color='inverse')

                if res['defense_method'] is not None and res['acc_after_defense'] is not None:
                    st.markdown('##### 🛡️ 防御效果')
                    dc1, dc2, dc3 = st.columns(3)
                    dc1.metric('防御方法', res['defense_method'])
                    dc2.metric('防御后精度', f'{res["acc_after_defense"]:.4f}',
                               f'↑ {res["acc_after_defense"] - res["acc_after"]:.4f}')
                    dc3.metric('防御后F1', f'{res["f1_after_defense"]:.4f}',
                               f'↑ {res["f1_after_defense"] - res["f1_after"]:.4f}')

                if 'adv_batch_results' not in st.session_state:
                    st.session_state.adv_batch_results = []
                st.session_state.adv_batch_results.append(res)

        with adv_tabs[1]:
            st.markdown('#### 批量攻击评估')
            bc1, bc2 = st.columns(2)
            with bc1:
                batch_method = st.selectbox('攻击方法', list(ATTACK_METHODS.keys()), key='batch_method')
                batch_mode = st.selectbox('攻击模式', list(ATTACK_MODES.keys()), key='batch_mode')
            with bc2:
                batch_ratios_str = st.text_input(
                    '攻击比例列表(%, 逗号分隔)',
                    '5,10,15,20,25',
                    key='batch_ratios'
                )
                batch_defense = st.checkbox('启用防御', value=False, key='batch_enable_def')
                batch_defense_method = None
                if batch_defense:
                    batch_defense_method = st.selectbox(
                        '防御方法', list(DEFENSE_METHODS.keys()), key='batch_defense'
                    )

            if st.button('📊 一键批量评估', type='primary', use_container_width=True, key='btn_batch_adv'):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = st.session_state.model
                data = st.session_state.data

                try:
                    ratios_pct = [float(x.strip()) for x in batch_ratios_str.split(',')]
                    ratios = [r / 100.0 for r in ratios_pct]
                except ValueError:
                    st.error('请输入有效的逗号分隔数字')
                    ratios = []

                if len(ratios) > 0:
                    with st.spinner('批量评估中...'):
                        batch_results = batch_evaluate(
                            model, data, batch_method, ratios, batch_mode,
                            device=device,
                            defense=batch_defense_method if batch_defense else None
                        )

                    st.session_state.adv_batch_results = batch_results
                    st.success('✅ 批量评估完成！')

            if 'adv_batch_results' in st.session_state and len(st.session_state.adv_batch_results) > 0:
                batch_res = st.session_state.adv_batch_results
                valid_ratios = [r for r in batch_res if isinstance(r.get('attack_ratio'), (int, float))]
                if len(valid_ratios) > 0:
                    ratios_x = [r['attack_ratio'] * 100 if r['attack_ratio'] <= 1 else r['attack_ratio']
                                for r in valid_ratios]
                    acc_drops = [r['acc_drop'] for r in valid_ratios]
                    f1_drops = [r['f1_drop'] for r in valid_ratios]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=ratios_x, y=acc_drops, mode='lines+markers',
                        name='精度下降', line=dict(color='#EF553B', width=2),
                        marker=dict(size=8)
                    ))
                    fig.add_trace(go.Scatter(
                        x=ratios_x, y=f1_drops, mode='lines+markers',
                        name='F1下降', line=dict(color='#636EFA', width=2, dash='dash'),
                        marker=dict(size=8)
                    ))

                    has_defense = any(r.get('acc_after_defense') is not None for r in valid_ratios)
                    if has_defense:
                        defense_drops = []
                        for r in valid_ratios:
                            if r.get('acc_after_defense') is not None:
                                defense_drops.append(r['acc_before'] - r['acc_after_defense'])
                            else:
                                defense_drops.append(None)
                        fig.add_trace(go.Scatter(
                            x=ratios_x, y=defense_drops, mode='lines+markers',
                            name='精度下降(有防御)', line=dict(color='#00CC96', width=2),
                            marker=dict(size=8, symbol='diamond')
                        ))

                    fig.update_layout(
                        title='攻击比例 vs 精度/F1下降幅度',
                        xaxis_title='攻击比例 (%)',
                        yaxis_title='下降幅度',
                        template='plotly_white',
                        height=450,
                        legend=dict(orientation='h', yanchor='bottom', y=-0.25,
                                    xanchor='center', x=0.5),
                        margin=dict(l=50, r=20, t=50, b=80)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown('#### 📋 批量评估结果表')
                    table_rows = []
                    for r in valid_ratios:
                        row = {
                            '攻击比例': f'{r["attack_ratio"]*100:.0f}%' if r['attack_ratio'] <= 1 else f'{r["attack_ratio"]:.0f}%',
                            '攻击前精度': f'{r["acc_before"]:.4f}',
                            '攻击后精度': f'{r["acc_after"]:.4f}',
                            '精度下降': f'{r["acc_drop"]:.4f}',
                            '攻击前F1': f'{r["f1_before"]:.4f}',
                            '攻击后F1': f'{r["f1_after"]:.4f}',
                            'F1下降': f'{r["f1_drop"]:.4f}',
                        }
                        if r.get('acc_after_defense') is not None:
                            row['防御后精度'] = f'{r["acc_after_defense"]:.4f}'
                            row['防御后F1'] = f'{r["f1_after_defense"]:.4f}'
                        table_rows.append(row)
                    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        with adv_tabs[2]:
            st.markdown('#### 防御策略对比')
            dc1, dc2 = st.columns(2)
            with dc1:
                def_attack_method = st.selectbox('攻击方法', list(ATTACK_METHODS.keys()), key='def_method')
                def_attack_ratio = st.slider('攻击比例(%)', 1, 50, 15, 1, key='def_ratio')
                def_attack_mode = st.selectbox('攻击模式', list(ATTACK_MODES.keys()), key='def_mode')
            with dc2:
                def_enable_deg = st.checkbox('度数边过滤', value=True, key='def_deg')
                def_enable_smooth = st.checkbox('特征平滑', value=True, key='def_smooth')
                def_deg_percentile = st.slider('度数过滤百分位', 80, 99, 95, 1, key='def_deg_pct')

            if st.button('⚔️ 执行防御对比', type='primary', use_container_width=True, key='btn_def_compare'):
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = st.session_state.model
                data = st.session_state.data
                ratio = def_attack_ratio / 100.0

                with st.spinner('执行攻击...'):
                    attacked_data = run_attack(
                        data, model, def_attack_method, ratio, def_attack_mode, device=device
                    )

                acc_before, f1_before = evaluate_on_data(model, data, device)
                acc_after, f1_after = evaluate_on_data(model, attacked_data, device)

                defense_results = {}
                if def_enable_deg:
                    defended_deg = degree_filter_defense(data, attacked_data, threshold_percentile=def_deg_percentile)
                    acc_deg, f1_deg = evaluate_on_data(model, defended_deg, device)
                    defense_results['度数边过滤'] = {'acc': acc_deg, 'f1': f1_deg}
                if def_enable_smooth:
                    defended_smooth = feature_smoothing_defense(attacked_data)
                    acc_smooth, f1_smooth = evaluate_on_data(model, defended_smooth, device)
                    defense_results['特征平滑'] = {'acc': acc_smooth, 'f1': f1_smooth}

                st.session_state.adv_defense_result = {
                    'acc_before': acc_before,
                    'f1_before': f1_before,
                    'acc_after': acc_after,
                    'f1_after': f1_after,
                    'defense_results': defense_results,
                }
                st.success('✅ 防御对比完成！')

            if 'adv_defense_result' in st.session_state and st.session_state.adv_defense_result is not None:
                dres = st.session_state.adv_defense_result

                st.markdown('---')
                m1, m2 = st.columns(2)
                m1.metric('原始精度', f'{dres["acc_before"]:.4f}')
                m2.metric('攻击后精度(无防御)', f'{dres["acc_after"]:.4f}',
                          f'↓ {dres["acc_before"] - dres["acc_after"]:.4f}', delta_color='inverse')

                if len(dres['defense_results']) > 0:
                    categories = ['原始', '攻击后(无防御)']
                    acc_vals = [dres['acc_before'], dres['acc_after']]
                    f1_vals = [dres['f1_before'], dres['f1_after']]
                    bar_colors = ['#636EFA', '#EF553B']

                    for def_name, def_res in dres['defense_results'].items():
                        categories.append(f'攻击后+{def_name}')
                        acc_vals.append(def_res['acc'])
                        f1_vals.append(def_res['f1'])
                        bar_colors.append('#00CC96' if '度数' in def_name else '#AB63FA')

                    fig = make_subplots(rows=1, cols=2, subplot_titles=('精度对比', 'F1对比'))
                    fig.add_trace(go.Bar(
                        x=categories, y=acc_vals, marker_color=bar_colors,
                        text=[f'{v:.4f}' for v in acc_vals], textposition='auto',
                        name='Accuracy'
                    ), row=1, col=1)
                    fig.add_trace(go.Bar(
                        x=categories, y=f1_vals, marker_color=bar_colors,
                        text=[f'{v:.4f}' for v in f1_vals], textposition='auto',
                        name='F1'
                    ), row=1, col=2)
                    fig.update_layout(
                        template='plotly_white', height=420, showlegend=False,
                        margin=dict(l=40, r=20, t=60, b=100),
                        xaxis_tickangle=-20,
                    )
                    fig.update_xaxes(tickangle=-20, row=1, col=1)
                    fig.update_xaxes(tickangle=-20, row=1, col=2)
                    st.plotly_chart(fig, use_container_width=True)

        with adv_tabs[3]:
            st.markdown('#### 攻击可视化')
            if 'adv_last_result' not in st.session_state or st.session_state.adv_last_result is None:
                st.info('请先在"单次攻击评估"或"批量评估"中执行攻击')
            else:
                last_res = st.session_state.adv_last_result
                attacked_data = last_res.get('attacked_data')
                if attacked_data is None:
                    st.info('未找到攻击后的图数据，请重新执行攻击')
                else:
                    data = st.session_state.data
                    model = st.session_state.model
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                    added, removed, affected_nodes = get_perturbed_edge_info(data, attacked_data)

                    st.markdown(f'**扰动统计**: 添加 {len(added)} 条边, 删除 {len(removed)} 条边, 受影响节点 {len(affected_nodes)} 个')

                    vc1, vc2 = st.columns(2)
                    with vc1:
                        has_labels = st.session_state.num_classes > 0
                        colors_orig = None
                        if has_labels and data.y is not None:
                            colors_orig = data.y.cpu().numpy().tolist()
                        fig_orig = plot_graph(
                            st.session_state.G, pos=st.session_state.pos,
                            node_colors=colors_orig,
                            class_names=st.session_state.class_names,
                            color_theme=st.session_state.color_theme,
                            title='原始图（按真实标签着色）'
                        )
                        st.plotly_chart(fig_orig, use_container_width=True)

                    with vc2:
                        G_attacked = to_networkx(attacked_data, to_undirected=True)
                        try:
                            pos_attacked = compute_layout(G_attacked, layout='spring', seed=42)
                        except Exception:
                            pos_attacked = st.session_state.pos

                        has_labels = st.session_state.num_classes > 0
                        colors_att = None
                        if has_labels and data.y is not None:
                            colors_att = data.y.cpu().numpy().tolist()

                        edge_traces = []
                        perturbed_edge_set = added | removed

                        for u, v in G_attacked.edges():
                            x0, y0 = pos_attacked.get(u, (0, 0))
                            x1, y1 = pos_attacked.get(v, (0, 0))
                            key = (min(u, v), max(u, v))
                            is_perturbed = key in perturbed_edge_set
                            ec = '#FF0000' if is_perturbed else 'rgba(150, 150, 150, 0.4)'
                            ew = 2.5 if is_perturbed else 0.8

                            edge_traces.append(go.Scatter(
                                x=[x0, x1, None], y=[y0, y1, None],
                                mode='lines',
                                line=dict(width=ew, color=ec),
                                hoverinfo='none',
                                showlegend=False
                            ))

                        node_x = [pos_attacked.get(n, (0, 0))[0] for n in G_attacked.nodes()]
                        node_y = [pos_attacked.get(n, (0, 0))[1] for n in G_attacked.nodes()]

                        normal_nodes_x, normal_nodes_y, normal_colors = [], [], []
                        affected_nodes_x, affected_nodes_y, affected_colors = [], [], []

                        for idx_n, n in enumerate(G_attacked.nodes()):
                            px, py = pos_attacked.get(n, (0, 0))
                            c = colors_att[idx_n] if colors_att and idx_n < len(colors_att) else 0
                            if n in affected_nodes:
                                affected_nodes_x.append(px)
                                affected_nodes_y.append(py)
                                affected_colors.append(c)
                            else:
                                normal_nodes_x.append(px)
                                normal_nodes_y.append(py)
                                normal_colors.append(c)

                        theme = COLOR_THEMES.get(st.session_state.color_theme, COLOR_THEMES['Plotly默认'])
                        color_scale = theme['color_scale']

                        if normal_nodes_x:
                            edge_traces.append(go.Scatter(
                                x=normal_nodes_x, y=normal_nodes_y,
                                mode='markers',
                                marker=dict(
                                    size=10, colorscale=color_scale,
                                    color=normal_colors, line_width=1, line_color='white',
                                    showscale=True,
                                    colorbar=dict(thickness=15, title=dict(text='类别', side='right'), xanchor='left')
                                ),
                                hoverinfo='text',
                                hovertext=[f'节点{n}' for n in G_attacked.nodes() if n not in affected_nodes],
                                showlegend=False,
                                name='正常节点'
                            ))

                        if affected_nodes_x:
                            edge_traces.append(go.Scatter(
                                x=affected_nodes_x, y=affected_nodes_y,
                                mode='markers',
                                marker=dict(
                                    size=14, colorscale=color_scale,
                                    color=affected_colors, symbol='triangle-up',
                                    line_width=2, line_color='red',
                                    showscale=False
                                ),
                                hoverinfo='text',
                                hovertext=[f'受影响节点{n}' for n in affected_nodes],
                                showlegend=False,
                                name='受影响节点'
                            ))

                        fig_att = go.Figure(data=edge_traces)
                        fig_att.update_layout(
                            title='攻击后的图（红色=被扰动边，三角形=受影响节点）',
                            template=theme['template'],
                            height=550, showlegend=False, hovermode='closest',
                            margin=dict(l=5, r=5, t=50, b=5),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                        )
                        st.plotly_chart(fig_att, use_container_width=True)

                    st.markdown('---')
                    st.markdown('#### 节点级影响热力图')
                    try:
                        probs_before = get_prediction_probs(model, data, device)
                        probs_after = get_prediction_probs(model, attacked_data, device)
                        prob_change = np.abs(probs_after - probs_before)
                        max_change = prob_change.max(axis=1)

                        num_nodes = len(max_change)
                        sample_size = min(200, num_nodes)
                        if num_nodes > 200:
                            sample_indices = np.linspace(0, num_nodes - 1, sample_size, dtype=int)
                            sample_change = max_change[sample_indices]
                            sample_ids = sample_indices
                        else:
                            sample_change = max_change
                            sample_ids = np.arange(num_nodes)

                        heatmap_fig = go.Figure(data=go.Heatmap(
                            z=prob_change[sample_ids].T,
                            x=[f'{i}' for i in sample_ids],
                            y=[f'类{c}' for c in range(prob_change.shape[1])],
                            colorscale='Reds',
                            colorbar=dict(title='概率变化'),
                        ))
                        heatmap_fig.update_layout(
                            title='节点级预测概率变化热力图',
                            xaxis_title='节点ID',
                            yaxis_title='类别',
                            template='plotly_white',
                            height=350,
                            margin=dict(l=60, r=20, t=50, b=60)
                        )
                        st.plotly_chart(heatmap_fig, use_container_width=True)

                        top_k = 10
                        top_indices = np.argsort(max_change)[-top_k:][::-1]
                        st.markdown(f'##### 受影响最大的 {top_k} 个节点')
                        top_data = []
                        for idx_n in top_indices:
                            pred_before = int(probs_before[idx_n].argmax())
                            pred_after = int(probs_after[idx_n].argmax())
                            top_data.append({
                                '节点ID': int(idx_n),
                                '攻击前预测类别': pred_before,
                                '攻击后预测类别': pred_after,
                                '是否改变预测': '是' if pred_before != pred_after else '否',
                                '最大概率变化': f'{max_change[idx_n]:.4f}',
                            })
                        st.dataframe(pd.DataFrame(top_data), use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.warning(f'热力图生成失败: {str(e)}')

        with adv_tabs[4]:
            st.markdown('#### 结果导出')
            if 'adv_batch_results' not in st.session_state or len(st.session_state.adv_batch_results) == 0:
                st.info('暂无评估结果，请先执行攻击评估')
            else:
                export_rows = []
                for r in st.session_state.adv_batch_results:
                    if not isinstance(r.get('attack_ratio'), (int, float)):
                        continue
                    ratio_display = f'{r["attack_ratio"]*100:.0f}%' if r['attack_ratio'] <= 1 else f'{r["attack_ratio"]:.0f}%'
                    row = {
                        '攻击方法': r.get('attack_method', ''),
                        '攻击比例': ratio_display,
                        '攻击模式': r.get('attack_mode', ''),
                        '攻击前精度': round(r['acc_before'], 4),
                        '攻击后精度': round(r['acc_after'], 4),
                        '精度变化': round(r['acc_drop'], 4),
                        '攻击前F1': round(r['f1_before'], 4),
                        '攻击后F1': round(r['f1_after'], 4),
                        'F1变化': round(r['f1_drop'], 4),
                    }
                    if r.get('defense_method') is not None:
                        row['防御方法'] = r['defense_method']
                        if r.get('acc_after_defense') is not None:
                            row['防御后精度'] = round(r['acc_after_defense'], 4)
                            row['防御后F1'] = round(r.get('f1_after_defense', 0), 4)
                    export_rows.append(row)

                if len(export_rows) > 0:
                    export_df = pd.DataFrame(export_rows)
                    st.dataframe(export_df, use_container_width=True, hide_index=True)

                    csv_bytes = export_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        '💾 导出攻击实验结果(CSV)',
                        csv_bytes,
                        'adversarial_results.csv',
                        'text/csv',
                        use_container_width=True
                    )

st.markdown('---')
st.caption('GNN实验平台 v1.0 · PyTorch Geometric + Streamlit + NetworkX + Plotly')
