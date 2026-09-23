import argparse
import json
import random
import sys
from pathlib import Path

# Allow this launcher to be run directly from the repository root:
# `python experiments/train_models.py ...`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

import torch
from graph_util import FEATURE_PROTOCOLS, feature_input_dim, get_adj_mat
from model.functional.MaxCoverLoss import KSetMaxCoverAdjLoss
from load_graph import load_train, bfs_split, load_val, load_train_2000, bfs_split_two_parts, bfs_split_dp, rate_split, rate_split_dp, random_walking_with_restart
from util import get_model
from model.layer.GRATLayer import GRAT_INITIALIZATIONS
from model.GRAT import GRAT_VARIANTS
from train import train
from config import (
    PROTOCOL_PROFILES,
    RELEASED_COMPATIBILITY_PROTOCOL,
    select_indicator_parameters,
)
from privacy import (
    ACCOUNTANTS,
    DP_NOISE_STREAMS,
    LEGACY_GLOBAL_DP_NOISE_STREAM,
    PAPER_BINOMIAL,
)
from seeding import (
    SAMPLING_SEED_MODES,
    SEED_MODES,
    resolve_round_seeds,
    seed_protocol,
)
from sampling import (
    PAPER_SAMPLING_PROTOCOL,
    RELEASED_LEGACY_SAMPLING_PROTOCOL,
    SAMPLING_PROTOCOLS,
)

import time
import logging
import pickle
import logging
import pandas as pd
from furl import furl
from baselines.heuristics import d_greedy, d_closure, greedy, bfs
from torch.utils.data import Dataset, DataLoader


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


# ==================== PARAMETERS ====================
TRAIN = True  # TRAIN MODE
HIDDEN_FEATS = [32, 32, 32, 32, 32, 32]
C_LOSS = 1
ROUND = 2

class GraphDataset(Dataset):
    def __init__(self, edges):
        self.edges = edges

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        return {
            'idx': idx,
            'edge': self.edges[idx]
        }

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_upper_bound(args):
    setting = args.settings[0]
    if setting == "PrivIM":
        if args.base_occurrence_bound is not None:
            return int(args.base_occurrence_bound)
        # Lemma 1: a node may occur in at most sum(theta**i, i=0..r)
        # Algorithm 1 subgraphs.
        upper_bound = sum(args.theta ** i for i in range(args.r + 1))
    elif setting in {"PrivIM+SCS", "PrivIM+", "PrivIM*"}:
        upper_bound = args.score_upper_bound
    else:
        raise ValueError(f"unsupported setting: {setting}")
    return int(upper_bound)

def get_dir_name(args):
    setting = args.settings[0]
    model_name = args.model_name[0]
    dataset = args.dataset[0]
    return f"{dataset}_dp/{setting}"

def get_gnns_name(args):
    model_name = args.model_name[0]
    setting = args.settings[0]
    dataset = args.dataset[0]
    return f"{dataset}_dp/{model_name}"


def train_models(debug=False):
    continuous_sampling_rng = None
    continuous_sampling_numpy_rng = None
    if args.sampling_seed_mode == "continuous":
        continuous_sampling_rng = random.Random(args.sampling_seed_base)
        continuous_sampling_numpy_rng = np.random.RandomState(
            args.sampling_seed_base
        )

    # Prepare training instances
    for idx in range(args.round):
        round_seeds = resolve_round_seeds(
            round_index=idx,
            model_seed=args.seed,
            model_seed_mode=args.model_seed_mode,
            sampling_seed=args.sampling_seed_base,
            batch_seed=args.batch_seed,
            batch_seed_mode=args.batch_seed_mode,
            sampling_seed_mode=args.sampling_seed_mode,
            dp_noise_seed=args.dp_noise_seed,
            dp_noise_seed_mode=args.dp_noise_seed_mode,
        )
        args.current_seed = round_seeds.sampling
        args.sampling_seed = round_seeds.sampling
        args.continuous_sampling_rng = continuous_sampling_rng
        args.continuous_sampling_numpy_rng = continuous_sampling_numpy_rng
        args.input_dim = feature_input_dim(
            args.feature_protocol, ones_dim=HIDDEN_FEATS[0]
        )
        args.use_cuda = args.device == "cuda" or (
            args.device == "auto" and torch.cuda.is_available()
        )
        print(
            f"Round: {idx}, model_seed={round_seeds.model}, "
            f"sampling_seed={round_seeds.sampling}, "
            f"batch_seed={round_seeds.batch}, "
            f"dp_noise_seed={round_seeds.dp_noise}"
        )

        raw_train_graphs, raw_train_dglgraphs = load_train(
            input_dim=args.input_dim,
            dataset=args.dataset[0],
            feature_type=args.feature_protocol,
            use_cuda=args.use_cuda,
        )
        metadata_path = (
            PROJECT_ROOT / "dataset" / "prepared" / args.dataset[0] / "metadata.json"
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if args.auto_parameters and args.settings[0] != "PrivIM":
            # Appendix H defines the indicator with the original dataset size
            # |V|, not the size of the 50% training split.
            original_node_count = int(metadata["original_nodes"])
            args.subgraph_size, args.score_upper_bound = select_indicator_parameters(
                args.dataset[0], original_node_count
            )
            print(
                f"Indicator parameters: n={args.subgraph_size}, "
                f"M={args.score_upper_bound}, |V|={original_node_count}"
            )
        elif args.auto_parameters:
            print(
                "Gamma indicator applies to PrivIM*; base PrivIM keeps "
                f"n={args.subgraph_size}."
            )
        # train_graphs, train_dglgraphs, num_layers, min_subgraphs_per_layer = rate_split_dp(raw_train_graphs, raw_train_dglgraphs,
        #             num_subgraphs=args.num_subgraph, min_node_threshold=args.subgraph_size, K=args.theta,
        #             r=args.r, k_training=args.k_train, args=args)

        train_graphs, train_dglgraphs, sampling_metadata = random_walking_with_restart(
            raw_train_graphs, raw_train_dglgraphs, args=args
        )
        if len(train_graphs) < args.batchsize:
            raise RuntimeError(
                f"sampling produced {len(train_graphs)} subgraphs, fewer than "
                f"batchsize={args.batchsize} (dataset={args.dataset[0]}, "
                f"setting={args.settings[0]}, n={args.subgraph_size}, "
                f"theta={args.theta}, r={args.r}, L={args.walk_length}). "
                "Use a smaller --subgraph_size/--batchsize or a larger "
                "--walk-length after checking the paper configuration."
            )

        upper_bound = calculate_upper_bound(args=args)
        dir_name = get_dir_name(args=args)
        gnns_name = get_gnns_name(args=args)
        # train_graphs, train_dglgraphs= bfs_split(raw_train_graphs, raw_train_dglgraphs,
        #                                           num_subgraphs=args.num_subgraph,
        #                                           min_node_threshold=args.subgraph_size,
        #                                           K=args.theta,
        #                                           r=args.r, k_training=args.k_train,
        #                                           args=args)
        #
        # train_graphs, train_dglgraphs = rate_split(raw_train_graphs, raw_train_dglgraphs,
        #                                             num_subgraphs=args.num_subgraph,
        #                                             min_node_threshold=args.subgraph_size,
        #                                             K=args.theta,
        #                                             r=args.r, k_training=args.k_train,
        #                                             args=args)
        #
        # num_layers = 999
        # min_subgraphs_per_layer = 999

        # 计算一下一个点平均度数
        degree_list = []
        for i in range(len(train_graphs)):
            degree_list.append(train_graphs[i].ecount() / train_graphs[i].vcount())
        print(f"Average Degree：{np.mean(degree_list)}, Upper Bound: {upper_bound}")

        # train_graphs, train_dglgraphs = raw_train_graphs, raw_train_dglgraphs

        # Train models
        for (k, d) in zip([args.k_train], args.d):
            print(f"d: {d}, k: {k}")
            adj_matrices = []
            greedy_perfs = []
            train_closed_graphs = []

            i = 0
            # 这部分是为了计算greedy的baseline
            for train_graph in train_graphs:
                i = i + 1
                # calculate graph closure
                train_closed_graph = bfs(train_graph, d=d)
                train_closed_graphs.append(train_closed_graph)

                # calculate the adj matrix of each graph
                adj_matrices.append(get_adj_mat(train_closed_graph))

                # KSetMaxCoverAdjLoss does not consume a per-subgraph CELF
                # target. Keep the positional placeholder expected by the
                # legacy train() interface; the evaluation CELF denominator
                # is computed independently in evaluate_models.py.
                greedy_perfs.append(None)

            for model_name in args.model_name:
                # Sampling uses its own local random.Random. Reset all global
                # RNGs immediately before model construction so preprocessing
                # cannot perturb initialization or DP-noise randomness.
                seed_everything(round_seeds.model)
                print(
                    f"Parameters: model: {model_name}, d: {d}, k: {k}, "
                    f"model_seed: {round_seeds.model}"
                )
                param_file = furl("params").add({
                    "model": model_name,
                    "d": d,
                    "seed": round_seeds.model,
                })

                model_variant = (
                    args.grat_variant
                    if model_name in {"GRAT2", "GRAT3", "GRAT4"}
                    else None
                )
                net = get_model(
                    model_name, args.input_dim, *HIDDEN_FEATS[1:],
                    grat_variant=model_variant,
                    grat_initialization=args.grat_initialization,
                )
                if debug:
                    logging.debug("Model loaded. Model info:")
                    logging.debug(net)

                # 把图转成list形式
                edges = []
                closed_edges = []

                for i in range(len(train_graphs)):
                    raw_edges = train_graphs[i].get_edgelist()
                    raw_closed_edges = train_closed_graphs[i].get_edgelist()

                    start_points = [edge[0] for edge in raw_edges]
                    end_points = [edge[1] for edge in raw_edges]
                    closed_start_points = [edge[0] for edge in raw_closed_edges]
                    closed_end_points = [edge[1] for edge in raw_closed_edges]
                    edges.append([start_points, end_points])
                    closed_edges.append([closed_start_points, closed_end_points])

                dataset = GraphDataset(edges)

                loss_list, _, _ = train(
                    net=net,
                    dataset=dataset,
                    loss_function=KSetMaxCoverAdjLoss(args.loss_penalty),
                    n_iteration=args.n_iteration,
                    batchsize=args.batchsize,
                    lr=args.lr,
                    k_train=args.k_train,
                    adj_matrices=adj_matrices,
                    greedy_perfs=greedy_perfs,
                    closed_edges=closed_edges,
                    dglgraphs=train_dglgraphs,
                    graphs=train_graphs,
                    closed_graphs=train_closed_graphs,
                    model_name=model_name,
                    d=d,
                    seed=round_seeds.model,
                    subgraph_size=args.subgraph_size,
                    num_subgraph=args.num_subgraph,
                    score_upper_bound=upper_bound,
                    rate = args.rate,
                    settings = args.settings[0],
                    round = idx,
                    with_dp = args.with_dp,
                    dataset_name = args.dataset[0],
                    target_epsilon = args.target_epsilon,
                    dir_name = dir_name,
                    gnns_name = gnns_name,
                    K = args.theta,
                    use_cuda=args.use_cuda,
                    train_node_count=raw_train_graphs[0].vcount(),
                    privacy_accountant=args.privacy_accountant,
                    l2_norm_clip=args.clip_norm,
                    sampler_metadata=sampling_metadata,
                    experiment_metadata={
                        "data_source_sha256": metadata.get("source_sha256"),
                        "prepared_seed": metadata.get("seed"),
                        "split_source": metadata.get("split_source", "generated"),
                        "artifact_revision": metadata.get("artifact_revision"),
                        "train_ratio": metadata.get("train_ratio"),
                        "isolated_policy": metadata.get(
                            "isolated_policy", "legacy-assigned"
                        ),
                        "feature_protocol": args.feature_protocol,
                        "input_dim": args.input_dim,
                        "sampling_protocol": args.sampling_protocol,
                        "sampling_hops": (
                            args.r
                            if args.sampling_protocol == PAPER_SAMPLING_PROTOCOL
                            else None
                        ),
                        "configured_sampling_hops": args.r,
                        "base_occurrence_bound_override": (
                            args.base_occurrence_bound
                        ),
                        "seeds": round_seeds.as_dict(),
                        "seed_protocol": seed_protocol(
                            model_seed=args.seed,
                            model_seed_mode=args.model_seed_mode,
                            sampling_seed=args.sampling_seed_base,
                            batch_seed=args.batch_seed,
                            batch_seed_mode=args.batch_seed_mode,
                            sampling_seed_mode=args.sampling_seed_mode,
                            dp_noise_seed=args.dp_noise_seed,
                            dp_noise_seed_mode=args.dp_noise_seed_mode,
                        ),
                    },
                    output_root=args.output_root,
                    lr_decay=args.lr_decay,
                    lr_decay_steps=args.lr_decay_steps,
                    model_variant=model_variant,
                    model_initialization=(
                        args.grat_initialization
                        if model_variant is not None else None
                    ),
                    batch_seed=round_seeds.batch,
                    sampling_seed=round_seeds.sampling,
                    dp_noise_seed=round_seeds.dp_noise,
                    dp_noise_stream=args.dp_noise_stream,
                    clip_non_private=args.clip_non_private,
                    protocol_profile=args.protocol_profile,
                    training_protocol=args.training_protocol,
                    privacy_accounting_steps=args.privacy_accounting_steps,
                    optimizer_momentum=args.optimizer_momentum,
                    sensitivity_noise_factor=args.sensitivity_noise_factor,
                    optimizer_update_cap=args.optimizer_update_cap,
                    parameter_averaging=args.parameter_averaging,
                    ema_decay=args.ema_decay,
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        default=0,
        type=int,
        help="base model seed; round i uses seed + i",
    )
    parser.add_argument(
        "--model-seed-mode", choices=SEED_MODES, default="per-round",
    )
    parser.add_argument(
        "--sampling-seed", default=0, type=int,
        help="base seed used only for subgraph-container sampling",
    )
    parser.add_argument(
        "--sampling-seed-mode", choices=SAMPLING_SEED_MODES, default="fixed",
        help="reuse one sampled container or vary its seed by round",
    )
    parser.add_argument(
        "--batch-seed", default=0, type=int,
        help="base RandomSampler seed; round i uses batch-seed + i",
    )
    parser.add_argument(
        "--batch-seed-mode", choices=SEED_MODES, default="per-round",
    )
    parser.add_argument(
        "--dp-noise-seed", default=0, type=int,
        help="base seed used only for Algorithm 2 Gaussian gradient noise",
    )
    parser.add_argument(
        "--dp-noise-seed-mode", choices=SEED_MODES, default="per-round",
        help="hold DP noise fixed across rounds or vary it independently",
    )
    parser.add_argument(
        "--dp-noise-stream", choices=DP_NOISE_STREAMS, default="independent",
        help=(
            "independent uses the dedicated seeded generator; legacy-global "
            "reproduces revision 31166ca torch.randn_like behavior"
        ),
    )
    parser.add_argument("--d", nargs="+", default=[1], type=int)
    parser.add_argument("--k_train", default=50, type=int)
    parser.add_argument("--batchsize", default=16, type=int)
    parser.add_argument("--lr", default=0.005, type=float)
    parser.add_argument(
        "--loss-penalty", default=C_LOSS, type=float,
        help=(
            "lambda in paper Equation (5); the released-code value is 1.0"
        ),
    )
    parser.add_argument(
        "--n_iteration",
        default=500,
        type=int,
        help=(
            "total optimizer updates T; 500 is the validated reproduction "
            "default because 50 updates leave the non-private model severely "
            "underfit"
        ),
    )
    parser.add_argument("--subgraph_size", default=40, type=int)
    parser.add_argument("--num_subgraph", default=256, type=int)
    parser.add_argument(
        "--theta", "--K", dest="theta", default=10, type=int,
        help="paper theta: maximum node in-degree for base PrivIM (default: 10)",
    )
    parser.add_argument("--r", default=3, type=int,
                        help="maximum sampling/message-passing hops")
    parser.add_argument("--score_upper_bound", default=10, type=int)
    parser.add_argument("--target_epsilon", default=3, type=int)
    parser.add_argument(
        "--lr-decay", "--gamma", dest="lr_decay", default=1.0, type=float,
        help=(
            "StepLR multiplier; 1.0 keeps the paper's reported 0.005 "
            "learning rate fixed; use 0.99 for released-code compatibility"
        ),
    )
    parser.add_argument(
        "--lr-decay-steps", default=5, type=int,
        help="apply learning-rate decay after this many optimizer updates",
    )
    parser.add_argument("--round", default=5, type=int)
    parser.add_argument(
        "--settings", nargs='+', default=["PrivIM+"],
        choices=["PrivIM", "PrivIM+SCS", "PrivIM+", "PrivIM*"],
    )
    parser.add_argument(
        "--model_name", nargs='+', default=["GRAT3"],
        choices=["GRAT3", "GCN3", "GAT3", "GIN3", "GraphSAGE3"],
    )
    parser.add_argument(
        "--grat-variant",
        choices=GRAT_VARIANTS,
        default="legacy_released",
        help=(
            "released layer used for Table II reproduction, or the explicit "
            "Appendix G equations for architecture diagnosis"
        ),
    )
    parser.add_argument(
        "--grat-initialization",
        choices=GRAT_INITIALIZATIONS,
        default="released_default",
        help="released PyTorch initialization or structural diagnostic mode",
    )
    parser.add_argument(
        "--feature-protocol",
        choices=FEATURE_PROTOCOLS,
        default="ones",
        help=(
            "ones reproduces released 32-D constant inputs; degree uses "
            "a 3-D structural vector; degree32 pads that vector to the "
            "released 32-D width for a controlled comparison"
        ),
    )
    parser.add_argument("--dataset", nargs='+', default=["lastfm"],help="[email, bitcoin, lastfm, hepph, facebook, gowalla]")
    parser.set_defaults(with_dp=True)
    parser.add_argument("--with-dp", "--with_dp", dest="with_dp", action="store_true")
    parser.add_argument("--no-dp", dest="with_dp", action="store_false")
    parser.add_argument(
        "--clip-non-private",
        action="store_true",
        help=(
            "with --no-dp, retain Algorithm 2 per-subgraph clipping while "
            "setting Gaussian noise to zero (epsilon approaches infinity)"
        ),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--restart-probability", default=0.3, type=float)
    parser.add_argument("--walk-length", default=200, type=int)
    parser.add_argument("--frequency-decay", default=1.0, type=float)
    parser.add_argument("--boundary-divisor", default=2, type=int)
    parser.add_argument(
        "--sampling-protocol",
        choices=SAMPLING_PROTOCOLS,
        default=PAPER_SAMPLING_PROTOCOL,
        help=(
            "paper implements Algorithms 1/3; released-legacy reproduces "
            "revision 31166ca's retry sampler and is diagnostic-only"
        ),
    )
    parser.add_argument("--auto-parameters", action="store_true")
    parser.add_argument(
        "--privacy-accountant",
        choices=ACCOUNTANTS,
        default=PAPER_BINOMIAL,
    )
    parser.add_argument("--clip-norm", default=3.0, type=float)
    parser.add_argument(
        "--protocol-profile", choices=PROTOCOL_PROFILES, default="strict",
    )
    parser.add_argument(
        "--training-protocol",
        choices=("fixed-iterations", "released-epochs"),
        default="fixed-iterations",
        help=(
            "fixed-iterations implements Algorithm 2 T updates; released-epochs "
            "reproduces revision 31166ca's full DataLoader traversal per epoch"
        ),
    )
    parser.add_argument(
        "--privacy-accounting-steps", type=int,
        help="calibration steps; defaults to actual fixed iterations",
    )
    parser.add_argument(
        "--optimizer-update-cap", type=int,
        help=(
            "diagnostic cap on actual released-epoch optimizer updates; "
            "only valid for the released-compatibility profile"
        ),
    )
    parser.add_argument("--optimizer-momentum", default=0.0, type=float)
    parser.add_argument(
        "--parameter-averaging", choices=("none", "ema"), default="none",
        help=(
            "optionally save an exponential moving average of the private "
            "optimizer iterates; this is DP post-processing"
        ),
    )
    parser.add_argument("--ema-decay", default=0.99, type=float)
    parser.add_argument(
        "--sensitivity-noise-factor", default=1.0, type=float,
        help=(
            "compatibility multiplier applied only to released gradient noise; "
            "strict unbounded node-DP uses 1 per Lemma 2"
        ),
    )
    parser.add_argument(
        "--base-occurrence-bound", type=int,
        help="compatibility override for base PrivIM instead of Equation 6",
    )
    parser.add_argument(
        "--output-root", type=Path, default=PROJECT_ROOT / "outputs"
    )

    parser.add_argument("--sample_mode", default="bfs")
    parser.add_argument("--rate", default=0, type=float)
    parser.add_argument("--method", default='ours')
    parser.add_argument("--baseline", default='ours')

    args = parser.parse_args()
    if args.loss_penalty <= 0:
        parser.error("--loss-penalty must be positive")
    args.sampling_seed_base = args.sampling_seed
    if (
        args.dp_noise_stream == LEGACY_GLOBAL_DP_NOISE_STREAM
        and args.protocol_profile != RELEASED_COMPATIBILITY_PROTOCOL
    ):
        parser.error(
            "--dp-noise-stream legacy-global requires "
            "--protocol-profile released-compatibility"
        )
    if args.n_iteration <= 0:
        parser.error("--n_iteration must be positive")
    if not 0 < args.lr_decay <= 1:
        parser.error("--lr-decay must be in (0, 1]")
    if args.lr_decay_steps <= 0:
        parser.error("--lr-decay-steps must be positive")
    if args.privacy_accounting_steps is not None and args.privacy_accounting_steps <= 0:
        parser.error("--privacy-accounting-steps must be positive")
    if args.optimizer_update_cap is not None and args.optimizer_update_cap <= 0:
        parser.error("--optimizer-update-cap must be positive")
    if args.optimizer_momentum < 0:
        parser.error("--optimizer-momentum must be non-negative")
    if not 0 <= args.ema_decay < 1:
        parser.error("--ema-decay must be in [0, 1)")
    if (
        args.protocol_profile == RELEASED_COMPATIBILITY_PROTOCOL
        and args.parameter_averaging != "none"
    ):
        parser.error(
            "released-compatibility requires --parameter-averaging none"
        )
    if args.sensitivity_noise_factor <= 0:
        parser.error("--sensitivity-noise-factor must be positive")
    if (
        args.protocol_profile == RELEASED_COMPATIBILITY_PROTOCOL
        and args.training_protocol != "released-epochs"
    ):
        parser.error(
            "released-compatibility requires --training-protocol released-epochs"
        )
    if (
        args.sampling_protocol == RELEASED_LEGACY_SAMPLING_PROTOCOL
        and args.protocol_profile != RELEASED_COMPATIBILITY_PROTOCOL
    ):
        parser.error(
            "--sampling-protocol released-legacy requires "
            "--protocol-profile released-compatibility"
        )
    if (
        args.optimizer_update_cap is not None
        and args.protocol_profile != RELEASED_COMPATIBILITY_PROTOCOL
    ):
        parser.error(
            "--optimizer-update-cap requires "
            "--protocol-profile released-compatibility"
        )

    train_models(debug=False)
