import re
import argparse
import json

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
from experiments.igraph_loader import next_graph
from experiments.graph_util import get_rev_dgl
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


def ranking_scores(net, graph, features):
    """Return pre-sigmoid scores for numerically stable top-k selection."""
    logits = getattr(net, "logits", None)
    if logits is not None:
        return logits(graph, features)
    # Historical wrappers expose their pre-sigmoid backbone as ``grat``.
    return net.grat(graph, features)


def ranking_diagnostics(scores, k):
    """Summarize score saturation and ambiguity at the top-k boundary."""
    flat_scores = scores.detach().float().reshape(-1)
    finite = torch.isfinite(flat_scores)
    safe_scores = torch.nan_to_num(
        flat_scores, nan=-torch.inf, neginf=-torch.inf, posinf=torch.inf
    )
    probabilities = torch.sigmoid(safe_scores)
    finite_probabilities = probabilities[finite]
    saturation_fraction = (
        float(
            (
                (finite_probabilities <= 1e-4)
                | (finite_probabilities >= 1 - 1e-4)
            ).float().mean()
        )
        if finite_probabilities.numel()
        else np.nan
    )
    if k <= 0 or k > safe_scores.numel():
        raise ValueError("k must be in [1, number of scores]")
    ordered = torch.sort(safe_scores, descending=True).values
    boundary = ordered[k - 1]
    boundary_tie_count = int((safe_scores == boundary).sum())
    margin = (
        float(boundary - ordered[k])
        if k < ordered.numel()
        else np.nan
    )
    finite_scores = safe_scores[finite]
    return {
        "ranking_score_type": "logits",
        "ranking_finite_fraction": float(finite.float().mean()),
        "ranking_logit_std": (
            float(finite_scores.std(unbiased=False))
            if finite_scores.numel()
            else np.nan
        ),
        "ranking_sigmoid_saturation_fraction": saturation_fraction,
        "ranking_boundary_tie_count": boundary_tie_count,
        "ranking_topk_margin": margin,
    }


def select_topk_nodes(scores, k):
    """Select top-k logits with node id as a deterministic tie-breaker."""
    flat_scores = scores.reshape(-1)
    safe_scores = torch.nan_to_num(
        flat_scores, nan=-torch.inf, neginf=-torch.inf, posinf=torch.inf
    )
    # Stable sorting preserves the original node-id order for equal logits.
    return torch.argsort(safe_scores, descending=True, stable=True)[:k]


def get_modelnames(path_to_models=PATH_TO_MODELS):
    return glob(str(Path(path_to_models) / "**" / "*.pt"), recursive=True)


def checkpoint_metadata(param_file):
    metadata_path = Path(param_file).with_suffix(".json")
    if not metadata_path.is_file():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_net(param_file, use_cuda=USE_CUDA, debug=False):
    # f = furl(param_file)
    param_file_name = get_filename(param_file)
    pattern = r"model_(?P<model>.*)_d_(?P<d>\d+)_seed_(?P<seed>\d+)"
    match = re.search(pattern, param_file_name)
    d = int(match.group(2))
    model_name = match.group(1)
    device = "cuda" if use_cuda else "cpu"
    state_dict = torch.load(param_file, map_location=device)
    metadata = checkpoint_metadata(param_file)
    model_variant = None
    if model_name in {"GRAT2", "GRAT3", "GRAT4"}:
        if metadata:
            model_variant = metadata.get("model_variant")
        if model_variant is None:
            # Historical checkpoints predate the explicit variant metadata.
            model_variant = (
                "legacy_released"
                if any(key.endswith("fc.bias") for key in state_dict)
                else "paper_appendix_g"
            )
    # model_name = f.args["model"]
    input_dim = int(metadata.get("input_dim") or HIDDEN_FEATS[0])
    net = get_model(
        model_name, input_dim, *HIDDEN_FEATS[1:],
        grat_variant=model_variant,
    )
    net.load_state_dict(state_dict)
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
                    out = ranking_scores(
                        net, dglgraph, dglgraph.ndata['feat']
                    ).squeeze(1)
                    score_diagnostics = ranking_diagnostics(out, k)
                    nn_seeds = select_topk_nodes(out, k)
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
                    "inference_seconds_mean": t_mean,
                    "inference_seconds_std": t_std,
                    "memory_mib": memory,
                    "gpu": dglgraph.device.type == "cuda",
                    **score_diagnostics,
                })
        except Exception:
            logging.exception(f"Failed to evaluate on {graph_name}!")
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
                "inference_seconds_mean": np.nan,
                "inference_seconds_std": np.nan,
                "memory_mib": np.nan,
                "gpu": dglgraph.device.type == "cuda",
                "ranking_score_type": "logits",
                "ranking_finite_fraction": np.nan,
                "ranking_logit_std": np.nan,
                "ranking_sigmoid_saturation_fraction": np.nan,
                "ranking_boundary_tie_count": np.nan,
                "ranking_topk_margin": np.nan,
            })
    return records


def _model_experiment(param_name, path_to_models=PATH_TO_MODELS):
    relative = Path(param_name).resolve().relative_to(Path(path_to_models).resolve())
    parts = relative.parts
    if (
        len(parts) < 7
        or parts[0] == "evaluation"
        or not parts[3].startswith("n_")
        or not parts[4].startswith("Ng_")
    ):
        return None
    return {
        "dataset": parts[0],
        "setting": parts[1],
        "epsilon": parts[2],
        "subgraph_size": parts[3],
        "occurrence_bound": parts[4],
        "relative_parent": relative.parent,
    }


def evaluate_models(
    datasets=None, settings=None, ks=None, use_cuda=None, debug=False,
    path_to_models=PATH_TO_MODELS, path_to_output=PATH_TO_OUTPUT,
):
    use_cuda = torch.cuda.is_available() if use_cuda is None else use_cuda
    ks = ks or [50]
    model_param_names = []
    for name in get_modelnames(path_to_models):
        experiment = _model_experiment(name, path_to_models)
        if experiment is None:
            continue
        if datasets and experiment["dataset"] not in datasets:
            continue
        if settings and experiment["setting"] not in settings:
            continue
        model_param_names.append(name)
    path_to_output = Path(path_to_output)
    path_to_output.mkdir(parents=True, exist_ok=True)
    all_records = []

    graph_datasets = sorted({
        _model_experiment(name, path_to_models)["dataset"]
        for name in model_param_names
    })
    for name, graph, is_directed in next_graph(datasets=graph_datasets):
        suffix = "_test_edges.txt"
        dataset_name = name[:-len(suffix)] if name.endswith(suffix) else name
        graph_name_body = get_filename(name)
        feature_graphs = {}

        records = []
        for d in [1, 2, 3]:
            # closed_graph = bfs_K__r_(graph, d=d)
            closed_graph = None
            celf_coverages = None

            for param_name in model_param_names:
                experiment = _model_experiment(param_name, path_to_models)
                if experiment["dataset"] != dataset_name:
                    continue
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
                    celf_coverages = calculate_celf_coverages(graph, d, ks)
                
                logging.info(f"Graph: {name}. Model: {param_name_body}. d: {d}.")
                metadata = checkpoint_metadata(param_name)
                feature_protocol = metadata.get("feature_protocol", "ones")
                input_dim = int(metadata.get("input_dim") or HIDDEN_FEATS[0])
                feature_key = (feature_protocol, input_dim)
                if feature_key not in feature_graphs:
                    feature_graphs[feature_key] = get_rev_dgl(
                        graph, feature_protocol, input_dim, is_directed,
                        use_cuda,
                    )
                dglgraph = feature_graphs[feature_key]
                net = load_net(param_name, use_cuda=use_cuda)
                model_name = match.group(1)
                seed = int(match.group(3))
                # model_name = furl(param_name_body).args["model"]
                # seed = furl(param_name_body).args["seed"]
                # 清空records
                records = []
                records.extend(
                    evaluate_model(
                        net, d, model_name, seed, name, graph, dglgraph,
                        ks=ks, celf_coverages=celf_coverages, repeat=1,
                        closed_graph=closed_graph
                    )
                )  # name => graph name
                df_result = pd.DataFrame(records)
                df_result.insert(2, "setting", experiment["setting"])
                df_result.insert(3, "epsilon", experiment["epsilon"])
                df_result.insert(4, "subgraph_size", experiment["subgraph_size"])
                df_result.insert(5, "occurrence_bound", experiment["occurrence_bound"])
                df_result.insert(6, "feature_protocol", feature_protocol)
                df_result.insert(7, "input_dim", input_dim)
                df_result["protocol_profile"] = metadata.get(
                    "protocol_profile", "strict"
                )
                df_result["sampling_protocol"] = metadata.get(
                    "sampling_protocol", "paper"
                )
                df_result["dp_noise_stream"] = metadata.get(
                    "dp_noise_stream", "independent"
                )
                df_result["training_protocol"] = metadata.get(
                    "training_protocol", "fixed-iterations"
                )
                df_result["strict_privacy_claim"] = bool(
                    metadata.get("strict_privacy_claim", False)
                )
                df_result["actual_optimizer_steps"] = metadata.get(
                    "actual_optimizer_steps", metadata.get("iterations")
                )
                df_result["optimizer_update_cap"] = metadata.get(
                    "optimizer_update_cap"
                )
                df_result["privacy_accounting_steps"] = metadata.get(
                    "privacy_accounting_steps"
                )
                df_result["calibration_epsilon"] = metadata.get(
                    "calibration_epsilon", metadata.get("achieved_epsilon")
                )
                df_result["actual_steps_epsilon_nominal"] = metadata.get(
                    "actual_steps_epsilon_nominal"
                )
                df_result["actual_steps_epsilon_effective_noise"] = metadata.get(
                    "actual_steps_epsilon_effective_noise"
                )
                output_dir = path_to_output / experiment["relative_parent"]
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{param_name_body}.csv"
                df_result.to_csv(output_file, index=False)
                all_records.extend(df_result.to_dict("records"))
                logging.info(f"Saving results of {graph_name_body} to {output_file}!")
    if all_records:
        frame = pd.DataFrame(all_records)
        summary = summarize_records(frame)
        summary.to_csv(path_to_output / "summary.csv", index=False)
    return


def summarize_records(frame):
    """Aggregate per-checkpoint evaluation rows without dropping baselines.

    Non-private checkpoints intentionally have no DP-noise stream.  Pandas
    drops groups containing a null grouping key by default, which previously
    removed every ``eps_10000`` row from the run summary even though the
    individual evaluation CSV files had been written successfully.
    """
    return (
            frame.groupby(
                [
                    "graph", "setting", "epsilon", "subgraph_size",
                    "occurrence_bound", "feature_protocol", "input_dim",
                    "protocol_profile", "sampling_protocol",
                    "training_protocol", "dp_noise_stream",
                    "strict_privacy_claim", "model", "d", "k",
                ],
                as_index=False,
                dropna=False,
            )
            .agg(
                runs=("seed", "count"),
                ranking_score_type=("ranking_score_type", "first"),
                actual_optimizer_steps=("actual_optimizer_steps", "first"),
                optimizer_update_cap=("optimizer_update_cap", "first"),
                privacy_accounting_steps=("privacy_accounting_steps", "first"),
                calibration_epsilon=("calibration_epsilon", "first"),
                actual_steps_epsilon_nominal=(
                    "actual_steps_epsilon_nominal", "first"
                ),
                actual_steps_epsilon_effective_noise=(
                    "actual_steps_epsilon_effective_noise", "first"
                ),
                influence_mean=("n_covered", "mean"),
                influence_std=("n_covered", "std"),
                coverage_ratio_mean=("coverage_ratio_percent", "mean"),
                coverage_ratio_std=("coverage_ratio_percent", "std"),
                ranking_finite_fraction=(
                    "ranking_finite_fraction", "mean"
                ),
                ranking_logit_std=("ranking_logit_std", "mean"),
                ranking_sigmoid_saturation_fraction=(
                    "ranking_sigmoid_saturation_fraction", "mean"
                ),
                ranking_boundary_tie_count_mean=(
                    "ranking_boundary_tie_count", "mean"
                ),
                ranking_topk_margin_mean=("ranking_topk_margin", "mean"),
            )
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+", choices=("email", "bitcoin", "lastfm", "hepph", "facebook", "gowalla"))
    parser.add_argument("--settings", nargs="+")
    parser.add_argument("--k", nargs="+", type=int, default=[50])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--models-root", type=Path, default=PATH_TO_MODELS)
    parser.add_argument("--evaluation-root", type=Path, default=PATH_TO_OUTPUT)
    cli_args = parser.parse_args()
    cuda = cli_args.device == "cuda" or (
        cli_args.device == "auto" and torch.cuda.is_available()
    )
    evaluate_models(
        cli_args.dataset, cli_args.settings, cli_args.k, use_cuda=cuda,
        debug=False, path_to_models=cli_args.models_root,
        path_to_output=cli_args.evaluation_root,
    )
