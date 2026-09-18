import igraph
import torch
import dgl
import numpy as np

FEATURE_PROTOCOLS = ("ones", "degree", "degree32")


def feature_input_dim(protocol, ones_dim=32):
    if protocol in {"ones", "1", "degree32"}:
        return int(ones_dim)
    if protocol in {"degree", "deg"}:
        return 3
    raise ValueError(f"unsupported feature protocol: {protocol}")


# ==================== GRAPH LOADERS ====================
def get_rev_graph(graph, train_graph_is_directed):
    g = igraph.Graph(directed=True)
    g.add_vertices(graph.vcount())
    src, dst = zip(*graph.get_edgelist())
    g.add_edges(list(zip(dst, src)))
    if not train_graph_is_directed:
        g.add_edges(list(zip(src, dst)))
    return g


def gen_zero_feature(graph, feature_dim):
    """Generate all-zero features
    """
    return torch.zeros(graph.vcount(), feature_dim)


def gen_one_feature(graph, feature_dim):
    """Generate all-one features
    """
    return torch.ones(graph.vcount(), feature_dim)


def gen_deg_feature(graph, *args):  # args is only a placeholder
    """Constant plus graph-normalized in/out degree structural features."""
    indegree = torch.tensor(graph.indegree(), dtype=torch.float32)
    outdegree = torch.tensor(graph.outdegree(), dtype=torch.float32)
    indegree = indegree / indegree.max().clamp_min(1.0)
    outdegree = outdegree / outdegree.max().clamp_min(1.0)
    return torch.stack([torch.ones(graph.vcount()), indegree, outdegree], dim=1)


def gen_deg_feature_padded(graph, feature_dim):
    """Degree diagnostic with the released input width kept unchanged."""
    if feature_dim < 3:
        raise ValueError("degree32 requires feature_dim >= 3")
    structural = gen_deg_feature(graph)
    features = torch.zeros(graph.vcount(), feature_dim, dtype=torch.float32)
    features[:, :3] = structural
    return features


def gen_one_hot_feayture(graph, *args):  # args is only a placeholder
    return torch.eye(graph.vcount()).float()


FEATURE_TYPE_DICT = {
    "0": gen_zero_feature,
    "1": gen_one_feature,
    "ones": gen_one_feature,
    "onehot": gen_one_hot_feayture,
    "deg": gen_deg_feature,
    "degree": gen_deg_feature,
    "degree32": gen_deg_feature_padded,
}

def get_rev_dgl(graph, feature_type='0', feature_dim=None, is_directed=False, use_cuda=False):
    """get dgl graph from igraph
    """
    edges = graph.get_edgelist()
    if edges:
        src, dst = zip(*edges)
    else:
        src, dst = [], []

    if use_cuda:
        dglgraph = dgl.graph((dst, src), num_nodes=graph.vcount()).to(torch.device("cuda:0"))
    else:
        dglgraph = dgl.graph((dst, src), num_nodes=graph.vcount())
        
    if not is_directed:
        dglgraph.add_edges(src, dst)

    if use_cuda:
        dglgraph.ndata['feat'] = FEATURE_TYPE_DICT[feature_type](graph, feature_dim).cuda()
        dglgraph.ndata['degree'] = torch.tensor(graph.degree()).float().cuda()

    else:
        dglgraph.ndata['feat'] = FEATURE_TYPE_DICT[feature_type](graph, feature_dim)
        dglgraph.ndata['degree'] = torch.tensor(graph.degree()).float()
        
    return dglgraph
# ==================== END OF GRAPH LOADERS ====================


# ==================== GRAPH GENERATORS ====================

def gen_erdos_graph(nodes_num, p, directed):
    """Generate an Erdos-Renyi graph

    Args:
        nodes_num (int): number of nodes
        p (any): probability of having an edge
        train_graph_is_directed (bool): directed/undirected graph

    Returns:
        igraph object
    """
    g = igraph.Graph.Erdos_Renyi(
        nodes_num, p, directed=directed)
    return g


def gen_erdos_graphs(graphs_num, nodes_num, p, directed):
    return [gen_erdos_graph(nodes_num, p, directed) for _ in range(graphs_num)]
# ==================== END OF GRAPH GENERATORS ====================


def get_adj_mat(graph, d=1):
    if d == 1:
        adj_mat = np.array(graph.get_adjacency().data, dtype=bool)
        adj_mat = torch.from_numpy(np.array(adj_mat, dtype=int)).float()
        adj_mat += torch.eye(graph.vcount())
        return adj_mat
    elif d == 2:
        return get_adj_mat_2(graph)


def get_adj_mat_2(graph):
    adj_mat = np.array(graph.get_adjacency().data, dtype=bool)
    adj_mat = torch.from_numpy(np.array(adj_mat, dtype=int)).float()
    adj_mat = adj_mat.matmul(adj_mat) + adj_mat
    adj_mat += torch.eye(graph.vcount())
    return adj_mat


def get_adj_mat_3(graph):
    adj_mat = np.array(graph.get_adjacency().data, dtype=bool)
    adj_mat = torch.from_numpy(np.array(adj_mat, dtype=int)).float()
    adj_mat = adj_mat.matmul(adj_mat) + adj_mat
    adj_mat += torch.eye(graph.vcount())
    return adj_mat
