import re

import time
import sys
from pathlib import Path
import logging

# Allow this launcher to be run directly from the repository root:
# `python experiments/evaluate_models.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PATH_TO_MODELS = PROJECT_ROOT / "outputs"
PATH_TO_OUTPUT = PROJECT_ROOT / "outputs" / "evaluation"

from util import get_filename, get_model, get_memory
from glob import glob
from furl import furl
import torch
import pandas as pd
from experiments.igraph_loader import next_graph, next_dglgraph
from baselines.heuristics import d_greedy, greedy, bfs, get_influence, get_influence_d
import numpy as np
from timeout_decorator import timeout


# ==================== LOG SETTING ====================
DATE = time.strftime('%m-%d', time.localtime())
TIME = time.strftime('%H.%M.%S', time.localtime())
Path(f"log/{DATE}").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"log/{DATE}/debug-{DATE}_{TIME}.log"),
        logging.StreamHandler()
    ]
)
# ==================== END OF LOG SETTING ====================
TIME_LIMIT = 300
HIDDEN_FEATS = [32] * 6
USE_CUDA = True
KS = [10 * n for n in range(1, 11)]
bfs = timeout(seconds=TIME_LIMIT)(bfs)


def get_modelnames(path_to_models=PATH_TO_MODELS):
    return glob(str(Path(path_to_models) / "**" / "*.pt"), recursive=True)


def load_net(param_file, use_cuda=USE_CUDA, debug=False):
    # f = furl(param_file)
    param_file_name = get_filename(param_file)
    pattern = r"model_(?P<model>.*)_d_(?P<d>\d+)_seed_(?P<seed>\d+)"
    match = re.search(pattern, param_file_name)
    d = int(match.group(2))
    model_name = match.group(1)
    # model_name = f.args["model"]
    net = get_model(model_name, *HIDDEN_FEATS)
    net.load_state_dict(torch.load(param_file))
    if use_cuda:
        net.cuda()
    return net


def calculate_celf_coverages(graph, d, ks):
    """Calculate CELF influence spread for each k on the evaluation graph."""
    try:
        celf_graph = graph if d == 1 else bfs(graph, d)
    except Exception:
        logging.exception(f"Failed to build the {d}-hop graph for CELF.")
        return {k: np.nan for k in ks}

    celf_coverages = {}
    for k in ks:
        try:
            _, celf_n_covered = greedy(celf_graph, k)
            celf_coverages[k] = celf_n_covered
            # logging.info(
            #     f"CELF baseline: d: {d}, k: {k}. "
            #     f"Coverage: {celf_n_covered}/{graph.vcount()}="
            #     f"{celf_n_covered/graph.vcount():.2f}."
            # )
        except Exception:
            logging.exception(f"Failed to calculate CELF coverage for d={d}, k={k}.")
            celf_coverages[k] = np.nan

    return celf_coverages


def evaluate_model(net, d, model_name, seed, graph_name, graph, dglgraph, ks,
                   celf_coverages, repeat=1, closed_graph=None, debug=False):
    n, m = graph.vcount(), graph.ecount()
    net.eval()

    records = []
    with torch.no_grad():
        try:
            for k in ks:
                ts = np.zeros(repeat)
                for i in range(repeat):
                    # Select seeds
                    t_start = time.time()
                    out = net.grat(dglgraph, dglgraph.ndata['feat']).squeeze(1)
                    # 排除nan
                    out[torch.isnan(out)] = -1e9
                    _, nn_seeds = torch.topk(out, k)
                    ts[i] = (time.time() - t_start)

                # Evaluate time
                t_mean = ts.mean()
                t_std = ts.std() / np.sqrt(repeat)

                # Evaluate memory
                memory = get_memory()

                # Evaluate covereage
                # if closed_graph is None:
                #     n_covered = get_influence(graph, nn_seeds)
                # else:
                #     n_covered = get_influence(closed_graph, nn_seeds)
                n_covered = get_influence_d(graph, nn_seeds, d)
                celf_n_covered = celf_coverages.get(k, np.nan)
                coverage_ratio = n_covered / celf_n_covered
                coverage_ratio_percent = coverage_ratio * 100
                logging.info(
                    f"k: {k}."
                    f"N covered: {n_covered}. "
                    f"CELF N covered: {celf_n_covered}. "
                    f"Coverage ratio percent: {n_covered}/{celf_n_covered}={coverage_ratio_percent:.2f}. "
                )

                # Write to records
                records.append({
                    "graph": graph_name,
                    "model": model_name,
                    "seed": seed,
                    "n": n,
                    "m": m,
                    "d": d,
                    "k": k,
                    "n_covered": n_covered,
                    # "coverage": n_covered/n,
                    "celf_n_covered": celf_n_covered,
                    "coverage_ratio_percent": coverage_ratio_percent,
                    # "t_mean": t_mean,
                    # "t_std": t_std,
                    # "memory": memory,
                    "gpu": USE_CUDA,
                })
        except:
            logging.info(f"Failed to evaluate on {graph_name}!")
            records.append({
                "graph": graph_name,
                "model": model_name,
                "seed": seed,
                "n": n,
                "m": m,
                "d": d,
                "k": np.nan,
                "n_covered": np.nan,
                # "coverage": np.nan,
                "celf_n_covered": np.nan,
                "coverage_ratio_percent": np.nan,
                # "t_mean": np.nan,
                # "t_std": np.nan,
                # "memory": np.nan,
                "gpu": USE_CUDA
            })
    return records


def evaluate_models(debug=False):
    model_param_names = get_modelnames()
    PATH_TO_OUTPUT.mkdir(parents=True, exist_ok=True)

    for x in next_dglgraph(input_dim=HIDDEN_FEATS[0], n_limit=None, m_limit=None, use_cuda=USE_CUDA):
        name, graph, dglgraph, is_directed = x
        graph_name_body = get_filename(name)

        records = []
        for d in [1, 2, 3]:
            # closed_graph = bfs_K__r_(graph, d=d)
            closed_graph = None
            celf_coverages = None

            for param_name in model_param_names:
                param_name_body = get_filename(param_name)
                pattern = r"model_(?P<model>.*)_d_(?P<d>\d+)_seed_(?P<seed>\d+)_round_(?P<round>\d+)"
                match = re.search(pattern, param_name_body)
                net_d = int(match.group(2))
                net_round = int(match.group(4))
                final_name = f'{graph_name_body}_{net_round}'
                # net_d = int(furl(param_name_body).args["d"])
                if net_d != d:
                    continue

                if celf_coverages is None:
                    celf_coverages = calculate_celf_coverages(graph, d, KS)
                
                logging.info(f"Graph: {name}. Model: {param_name_body}. d: {d}.")
                net = load_net(param_name)
                model_name = match.group(1)
                seed = int(match.group(3))
                # model_name = furl(param_name_body).args["model"]
                # seed = furl(param_name_body).args["seed"]
                # 清空records
                records = []
                records.extend(
                    evaluate_model(
                        net, d, model_name, seed, name, graph, dglgraph,
                        ks=KS, celf_coverages=celf_coverages, repeat=1,
                        closed_graph=closed_graph
                    )
                )  # name => graph name
                df_result = pd.DataFrame(records)
                output_file = PATH_TO_OUTPUT / f"{final_name}.csv"
                df_result.to_csv(output_file)
                logging.info(f"Saving results of {graph_name_body} to {output_file}!")
    return


if __name__ == '__main__':
    evaluate_models(debug=False)
