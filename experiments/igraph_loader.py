import os
import igraph
from graph_util import get_rev_dgl


# dglgraph parameters
INPUT_DIM = 32
FEATURE_TYPE="1"

DIRECTED_GRAPHS = None  # all
UNDIRECTED_GRAPHS = []  # empty
BLACK_LIST = ["bitcoinotc_train.txt"]
PATH_TO_DIRECTED_GRAPHS = "dataset/directed/"
PATH_TO_UNDIRECTED_GRAPHS = "dataset/undirected/"

# DIRECTED_GRAPHS = [  # small directed graphs
#     "soc-sign-jazz.txt",
#     "soc-advogato.txt",
#     "soc-wiki-elec.txt",
#     "soc-anybeat.txt",
#     "soc-jazz.txt",
#     "soc-epinions.txt"
# ]


# DIRECTED_GRAPHS = [
#     "soc-email.txt",
#     "soc-slashdot.txt",
#     "soc-douban.txt",
#     "soc-themarker.txt",
#     "soc-sign-epinions.txt",
#     "soc-gowalla.txt",
#     "soc-twitter-follows.txt",
#     "soc-twitter-follows-mun.txt",
#     "soc-delicious.txt",
# ]


def get_graph_names(path_to_test):
    graph_names = []
    for rt, _, files in os.walk(path_to_test):
        if rt == path_to_test:
            for file in files:
                if file.endswith('.txt'):
                    graph_names.append(file)
    return graph_names


def load_igraphs(graph_names, path, is_directed, black_list=BLACK_LIST):
    for name in graph_names:
        if name in black_list:
            continue
        print(f"Current graph: {name}", flush=True)

        graph = igraph.Graph().Read_Edgelist(
            f"{path}{name}", directed=is_directed)
        yield name, graph, is_directed


def next_graph():
    for path, is_directed, graph_names in [(PATH_TO_DIRECTED_GRAPHS, True, DIRECTED_GRAPHS), (PATH_TO_UNDIRECTED_GRAPHS, False, UNDIRECTED_GRAPHS)]:
        if graph_names is None:
            graph_names = get_graph_names(path)
        for x in load_igraphs(graph_names, path, is_directed):
            yield x


def load_dglgraphs(graph_names, path, is_directed, black_list=BLACK_LIST, input_dim=INPUT_DIM, feature_type=FEATURE_TYPE, use_cuda=False):
    for name in graph_names:
        if name in black_list:
            continue
        print(f"Current graph: {name}", flush=True)

        # Load igraph
        graph = igraph.Graph().Read_Edgelist(
            f"{path}{name}", directed=is_directed)
        
        # Load dglgraph
        dglgraph = get_rev_dgl(graph, feature_type,
                               input_dim, is_directed, use_cuda)
        yield name, graph, dglgraph, is_directed


def next_dglgraph(input_dim=32, use_cuda=False):
    for path, is_directed, graph_names in [(PATH_TO_DIRECTED_GRAPHS, True, DIRECTED_GRAPHS), (PATH_TO_UNDIRECTED_GRAPHS, False, UNDIRECTED_GRAPHS)]:
        if graph_names is None:
            graph_names = get_graph_names(path)
        for x in load_dglgraphs(graph_names, path, is_directed, input_dim=input_dim, use_cuda=use_cuda):
            yield x
