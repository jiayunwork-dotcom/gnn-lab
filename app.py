import os
import io
import time
import numpy as np
import pandas as pd
import streamlit as st
import torch

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
from visualizations import (
    compute_layout, plot_degree_distribution, plot_training_curves,
    plot_confusion_matrix, plot_graph, plot_embeddings_2d,
    plot_attention_graph, plot_f1_per_class, plot_comparison_bar
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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    '📊 数据概览', '🧠 GNN训练', '🔍 社区发现',
    '📈 可视化分析', '⚡ 对比实验'
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

        with viz_c2:
            G_plot = st.session_state.G
            if display_mode == '最大连通分量' and s['num_nodes'] > 500:
                G_plot = get_largest_component(G_plot)

            has_labels = st.session_state.num_classes > 0
            colors = None
            if has_labels and st.session_state.data.y is not None:
                colors = st.session_state.data.y.cpu().numpy().tolist()
            fig = plot_graph(G_plot, pos=st.session_state.pos,
                             node_colors=colors if has_labels else None,
                             class_names=st.session_state.class_names,
                             show_labels=show_labels,
                             title='图可视化（按真实类别着色）' if has_labels else '图可视化')
            st.plotly_chart(fig, use_container_width=True)

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

            st.success(f'✅ 训练完成！用时 {results["total_time"]:.2f}s')

        if st.session_state.results is not None:
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
                    class_names=st.session_state.class_names
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

st.markdown('---')
st.caption('GNN实验平台 v1.0 · PyTorch Geometric + Streamlit + NetworkX + Plotly')
