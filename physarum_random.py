import numpy as np
import random
import os
from matplotlib import pyplot as plt
import networkx as nx
from networkx.algorithms import approximation

####################
# 超参数
# I0 默认2.0
I = 2.0
# gamma 默认1.8
g = 1.8
# 设置时间步长
dt = 0.1

name = 'best_#5'
plot_step = 3
dont_plot = False
####################
# 随机生成图

# 所有节点个数
n = 100

# FS点：地图城市（食物）点 其中将重要城市设置为5个点
FS_dict = {}
capital = random.randint(0, n - 1)
for i in range(5):
    FS_dict[i] = capital
for i in range(5, 11):
    FS_dict[i] = random.randint(0, n - 1)

# 传导原生质的质管，[节点1，节点2，长度]
graph = []
for p in range(n):
    i = p // 10
    j = p % 10
    flag1 = j < 9
    flag2 = i < 9
    l = random.randint(30, 50) / 10.0
    if flag1:
        graph.append([p, p + 1, l])
    if flag2:
        graph.append([p, p + 10, l])
    if flag1 and flag2 and random.random() > 0.5:
        l += random.randint(10, 20) / 10.0
        graph.append([p, p + 11, l])
####################

if not dont_plot:
    save_dir = './' + name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    FS_color_dict = {'y': [], 'b': [], 'r': []}
    for i in range(n):
        if i not in FS_dict.values():
            FS_color_dict['y'].append(i)
        elif i == capital:
            FS_color_dict['r'].append(i)
        else:
            FS_color_dict['b'].append(i)


def L_init(graph, n):
    # 图的边（导管）的权值（长度）矩阵
    L = np.ones((n, n), dtype=float) * np.inf
    for link in graph:
        i, j, l = link
        L[i, j] = l
        L[j, i] = l
    return L


def Q_step(L, D, I0):
    # 随机选取一个FS点p作为source点，此处总流量 Σj(Q_pj)=I0
    # 随机选取另一个FS点q作为sink点，此处总流量 Σj(Q_qj)=-I0
    # 由于流量平衡，对其他所有点i(i≠p, i≠q)，总流量 Σj(Q_qj)=0
    n = L.shape[0]
    # 质管参数 D=πr^4/8η
    if D is None:
        # D = np.random.rand(n, n)
        D = np.ones((n, n))
    # Q_ij = D_ij(p_i - p_j)/L_ij = K_ij(p_i - p_j)
    K = np.true_divide(D, L)

    for i in range(n):
        K[i, i] -= np.sum(K[i, :])

    K *= -1

    # 随机选取source点和sink点
    sourceFS, sinkFS = random.sample(FS_dict.keys(), 2)
    source, sink = FS_dict[sourceFS], FS_dict[sinkFS]
    while source == sink:
        sourceFS, sinkFS = random.sample(FS_dict.keys(), 2)
        source, sink = FS_dict[sourceFS], FS_dict[sinkFS]

    replace = random.randint(0, n - 1)
    while replace == source or replace == sink:
        replace = random.randint(0, n - 1)

    for i in range(n):
        K[replace, i] = 1

    # 各节点压力组成向量 pres = [p_1, p_2, ..., p_n]
    # Q矩阵的各行和ΣQ_i为节点i的流量和，由迭代条件
    # ΣQ_i = I0 (i=source)
    # ΣQ_i = -I0 (i=sink)
    # ΣQ_i = 0 (others)
    # ΣQ构成以ΣQ_i为元素的列向量
    # 将K变换成关于pres的参数矩阵K'，建立方程组 K'·pres=ΣQ

    # K'是奇异矩阵，因为其任意一行都与其他行线性相关
    # 新增约束 Σi(p_i)=0，以全1行替换K'的非source非sink行
    # 以0替换ΣQ对应元素
    # 若K'仍奇异，微调其对角线使方程组可解

    b = np.zeros(n)
    b[source] = I0
    b[sink] = -I0
    b[replace] = 0
    if np.linalg.matrix_rank(K) < n:
        for i in range(n):
            K[i, i] += 1e-3

    pres = np.linalg.solve(K, b)
    dpres = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dpres[i, j] = pres[i] - pres[j]
    Q = np.true_divide(D, L) * dpres
    return Q, source, sink


def D_step(Q, gamma):
    # 对每一步D满足变化规律
    # dD_ij/dt = f(|Q_ij|) - D_ij)
    # D_new_ij = D_ij + (f(|Q_ij|) - D_ij) * dt
    # f(|x|) = |x|^gamma/(1+|x|^gamma)
    n = Q.shape[0]
    absQ = np.array(Q)
    for i in range(n):
        for j in range(n):
            absQ[i, j] = pow(abs(Q[i, j]), gamma)
    f = np.true_divide(absQ, absQ + 1)
    return D + (f - D) * dt

def step(L, D, I0=2.0, gamma=1.8):
    # 步进
    # 从上一步D计算下一步Q，进而计算下一步D
    Q, source, sink = Q_step(L, D, I0=I0)
    return D_step(Q, gamma=gamma), source, sink


def MST(G, edges, pos, node_colors=None, save=None):
    # 生成过局部点的最小生成树（Steiner树）
    edge_labels = {}
    for e in edges:
        i, j, l = e
        edge_labels[(i, j)] = l

    ST = approximation.steinertree.steiner_tree(G, list(set(FS_dict.values())))
    if not dont_plot:
        if node_colors is None:
            node_colors = FS_color_dict
        for c in node_colors.keys():
            nx.draw_networkx_nodes(G, pos, nodelist=node_colors[c], node_color=c, node_size=50)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=ST.edges,
            width=10,
            alpha=0.5,
            edge_color="g",
        )

        if save is not None:
            # plt.show(block=False)
            plt.savefig(save_dir + 'MST.png', format='PNG')
            plt.clf()
        else:
            plt.show()
    print('MST ', end='')
    return measure(ST.edges, edge_labels)


def plot_graph(n, G, edges, pos, D, node_colors=None, save=None):
    # 调用NetworkX模块，可视化图
    edge_labels = {}
    for e in edges:
        i, j, l = e
        edge_labels[(i, j)] = l
    PT = []
    if dont_plot:
        for i in range(n):
            for j in range(i + 1, n):
                if D[i, j] > 1e-5:
                    PT.append((i, j))
    else:
        if node_colors is None:
            node_colors = FS_color_dict
        for c in node_colors.keys():
            nx.draw_networkx_nodes(G, pos, nodelist=node_colors[c], node_color=c, node_size=50)
        nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)

        for i in range(n):
            for j in range(i + 1, n):
                d = D[i, j]
                if d > 1e-5:
                    PT.append((i, j))
                    w = np.clip(d * 15, 10, 20)
                    nx.draw_networkx_edges(
                        G,
                        pos,
                        edgelist=[(i, j)],
                        width=w,
                        alpha=0.5,
                        edge_color="c",
                    )

        if save is not None:
            plt.savefig(save_dir + str(save) + '.png', format='PNG')
            plt.clf()
        else:
            plt.show()
    return measure(PT, edge_labels)


def measure(edges, labels):
    FS_list = list(set(FS_dict.values()))
    FSs = len(FS_list)

    cost = 0
    node_list = []
    edge_list = []
    for e in edges:
        i, j = e
        if i not in node_list:
            node_list.append(i)
        if j not in node_list:
            node_list.append(j)
        e_ = e if i < j else (j, i)
        l = labels[e_]
        edge_list.append((i ,j, l))
        cost += l

    subG = nx.Graph()
    subG.add_nodes_from(node_list)
    subG.add_weighted_edges_from(edge_list)
    min_dis_sum = 0
    for i in range(FSs):
        for j in range(i, FSs):
            try:
                min_dis_sum += nx.dijkstra_path_length(
                    subG,
                    source=FS_list[i],
                    target=FS_list[j],
                    weight='weight'
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                min_dis_sum += np.inf

    fault_tol = len(edge_list)

    def cut_l(l, i):
        l_ = []
        for k, e in enumerate(l):
            if k != i:
                l_.append(e)
        return l_

    for i in range(len(edges)):
        cutG = nx.Graph()
        cutG.add_nodes_from(node_list)
        cutG.add_weighted_edges_from(cut_l(edge_list, i))
        flag = False
        for p in range(FSs):
            if flag:
                break
            for q in range(p + 1, FSs):
                n1, n2 = FS_list[p], FS_list[q]
                f1 = n1 not in node_list
                f2 = n2 not in node_list
                if f1 or f2 or not nx.has_path(cutG, n1, n2):
                    fault_tol -= 1.0
                    flag = True
                    break
        del cutG

    fault_tol /= float(len(edge_list))

    print('TL:%.2f MD:%.2f FT:%.2f' % (cost, min_dis_sum, fault_tol))

    return cost, min_dis_sum, fault_tol


if __name__ == '__main__':
    L = L_init(graph, n)

    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            l = L[i, j]
            if l != np.inf:
                edges.append((i, j, l))
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_weighted_edges_from(edges)
    pos = nx.spring_layout(G)
    # 计算最小生成树作为性能参照
    TL_mst, MD_mst, FT_mst = MST(G, edges, pos, save='MST')

    # 初始化D：每个有连接质管系数为1
    D = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if L[i, j] == np.inf:
                D[i, j] = 0
    print('t0')
    TL, MD, FT = plot_graph(n, G, edges, pos, D, save=0)
    for i in range(1, 301):
        # 循环迭代
        D, source, sink = step(L, D, I0=I, gamma=g)
        print('t%.1f source:%d sink:%d' % (i * dt, source, sink))
        if not i % plot_step:
            TL, MD, FT = plot_graph(n, G, edges, pos, D, save=i)

    c = TL / TL_mst
    p = MD / MD_mst
    e = FT / c
    info_dict = {
        'name': name,
        'I0': I,
        'gamma': g,
        'TL_mst': TL_mst,
        'MD_mst': MD_mst,
        'TL': TL,
        'MD': MD,
        'cost': c,
        'performance': p,
        'fault tolerance': FT,
        'efficiency': e
    }

    print(info_dict)
    log = open('log.txt', 'a')
    log.write(str(info_dict) + '\n')
    log.close()
    data = '%.2f %.1f %.3f %.3f %.3f %.3f\n' % (I, g, c, p, FT, e)
    dfile = open('data.txt', 'a')
    dfile.write(data)
    dfile.close()