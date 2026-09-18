import os
import json
from pathlib import Path
import igraph
from graph_util import get_rev_dgl


# dglgraph parameters
INPUT_DIM = 32
FEATURE_TYPE="1"

DIRECTED_GRAPHS = None  # all
UNDIRECTED_GRAPHS = []  # empty
BLACK_LIST = ["bitcoin_train_edges.txt"]
PATH_TO_DIRECTED_GRAPHS = "dataset/directed/"
PATH_TO_UNDIRECTED_GRAPHS = "dataset/undirected/"
PREPARED_ROOT = Path(__file__).resolve().parents[1] / "dataset" / "prepared"
PAPER_DATASETS = ("email", "bitcoin", "lastfm", "hepph", "facebook", "gowalla")

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


def load_prepared_graph(dataset, split="test", prepared_root=PREPARED_ROOT):
    directory = Path(prepared_root) / dataset
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    graph = igraph.Graph(
        n=int(metadata.get(f"{split}_nodes", metadata[f"{split}_assigned_nodes"])),
        directed=bool(metadata["directed"]),
    )
    edge_path = directory / f"{dataset}_{split}_edges.txt"
    edges = []
    with edge_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                source, target = line.split()[:2]
                edges.append((int(source), int(target)))
    graph.add_edges(edges)
    return edge_path.name, graph, graph.is_directed()


def next_graph(datasets=None, split="test"):
    for dataset in datasets or PAPER_DATASETS:
        yield load_prepared_graph(dataset, split)


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


def next_dglgraph(input_dim=32, use_cuda=False, datasets=None, split="test"):
    for name, graph, is_directed in next_graph(datasets=datasets, split=split):
        dglgraph = get_rev_dgl(
            graph, FEATURE_TYPE, input_dim, is_directed, use_cuda
        )
        yield name, graph, dglgraph, is_directed
