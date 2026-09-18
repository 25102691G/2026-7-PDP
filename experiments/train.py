import time
from torch import optim

from util import get_model, get_memory
from load_graph import load_train, load_val, bfs_split
from algo import greedy
import torch
from copy import deepcopy
from tqdm import tqdm
import logging
from pytorchtools import EarlyStopping
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler
import os
import json
from pathlib import Path
from privacy import (
    FIXED_BATCH_HYPERGEOMETRIC,
    PrivateSGD,
    multiterm_rdp,
    rdp_to_dp,
    sensitivity_occurrences,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ==================== PARAMETERS ====================
PATH_TO_PARAMS = str(PROJECT_ROOT / "params") + os.sep
USE_CUDA_TRAIN = True
HIDDEN_FEATS = [32, 32, 32, 32, 32, 32]
# ==================== END OF PARAMETERS ====================


# ==================== HELPER FUNCTIONS ====================
def model_file(model_name, path_to_params=PATH_TO_PARAMS):
    time_str = time.strftime("%m-%d", time.localtime())
    return f"{path_to_params}{time_str}{model_name}.pkl"


def get_influence(graph, seeds):
    covered = set()
    for seed in seeds:
        covered.add(int(seed))
        for u in graph.successors(seed):  # Add all the covered seeds
            covered.add(int(u))
    return len(covered)


def get_k_top(net, dglgraph, test_k):
    """Get top k nodes based on the net
    """
    dglgraph_ = deepcopy(dglgraph)
    seeds = set()

    cum_n_covered = 0
    for i in range(test_k):
        # bar.set_description('GetKTop')
        out = net(dglgraph_, dglgraph_.ndata['feat']).squeeze(1)
        _, top_node = torch.max(out, 0)  # index of the best-scored node
        seeds.add(top_node)
        srcs = []  # the src node of the i_th edge
        dsts = []  # the dst node of the i_th edge
        covered = dglgraph_.predecessors(top_node)
        parents = dglgraph_.successors(top_node)

        # add edges (top_node, succ)
        srcs.extend([top_node]*len(parents))
        dsts.extend(parents)

        # add edges (pred, top_node)
        srcs.extend(covered)
        dsts.extend([top_node]*len(covered))

        for node in covered:
            parents = dglgraph_.successors(node)
            srcs.extend([node]*len(parents))
            dsts.extend(parents)
        dglgraph_.remove_edges(dglgraph_.edge_ids(srcs, dsts))

        n_covered = len(covered)
        cum_n_covered += n_covered
        logging.info(f"Node {i+1}: {n_covered}/{cum_n_covered} points are covered.")

    return list(seeds)


def get_x_top(net, dglgraph, test_k, x):  # get top 1/x points.
    d = deepcopy(dglgraph)
    seeds = []

    for _ in range((test_k + x - 1) // x):
        # bar.set_description(' Get'+str(x)+'Top')
        out = net(d, d.ndata['feat']).squeeze(1)
        y = x if test_k >= x else test_k
        test_k -= x
        _, topNodes = torch.topk(out, x)
        seeds.extend(topNodes.tolist())
        srcs = []
        dsts = []
        nodes = []
        nodes.extend(topNodes)
        for topNode in topNodes:
            nodes.extend(d.predecessors(topNode))
        nodes = list(set(nodes))
        for node in nodes:
            for succ in d.successors(node):
                srcs.append(node)
                dsts.append(succ)
        d.remove_edges(d.edge_ids(srcs, dsts))

    seeds = list(set(seeds))
    return seeds


def get_x_top_plus(net, graph, dglgraph, test_k, x):
    graph_ = deepcopy(graph)
    graph_.to_directed()
    seeds = []
    if test_k % x == 0:
        bar = tqdm(range(test_k//x))
        # bar = list(range(test_k//x))
    else:
        bar = tqdm(range(test_k//x+1))
        # bar = list(range(test_k//x+1))
    for i in bar:
        bar.set_description(' Get'+str(x)+'TopPlus')

        out = net(dglgraph, dglgraph.ndata['feat']).squeeze(1)

        y = x if test_k >= x else test_k
        test_k -= x

        _, topNodes = torch.topk(out, x)

        # 删边可以有重复，但不能删除不存在的边
        nodes = topNodes.tolist()
        topNodesNum = len(nodes)
        seeds.extend(nodes)
        for i in range(topNodesNum):
            nodes.extend(graph_.successors(nodes[i]))
        nodes = list(set(nodes))

        edges = []
        for node in nodes:
            for pred in graph_.predecessors(node):
                edges.append((pred, node))

        if edges != []:
            srcs, dsts = zip(*edges)
            dglgraph.remove_edges(dglgraph.edge_ids(dsts, srcs))
        graph_.delete_edges(edges)
        dglgraph.ndata['degree'] = torch.tensor(graph.degree()).float()
    seeds = list(set(seeds))
    return seeds


def get_x_top_plus_plus(net, graph, dglgraph, test_k, x):
    dglgraph_ = deepcopy(dglgraph)
    graph_ = deepcopy(graph)
    graph_.to_directed()
    seeds = set()
    if test_k % x == 0:
        bar = tqdm(range(test_k//x))
    else:
        bar = tqdm(range(test_k//x+1))
    for i in bar:
        bar.set_description(' GetXTopPlusPlus')
        out = net(dglgraph_, dglgraph_.ndata['feat']).squeeze(1)
        y = x if test_k >= x else test_k
        test_k -= x
        r = 10  # 查找范围
        _, indices = torch.topk(out, r*x)
        count = 0
        i = 0
        topNodes = []
        while count < x and i < r*x:
            idx = int(indices[i])
            if idx in seeds:
                pass
            else:
                seeds.add(idx)
                topNodes.append(idx)
                count += 1
            i += 1
        # 删边可以有重复，但不能删除不存在的边
        nodes = []
        nodes.extend(topNodes)
        for topNode in topNodes:
            nodes.extend(graph_.successors(topNode))
        nodes = list(set(nodes))
        edges = []
        for node in nodes:
            for pred in graph_.predecessors(node):
                edges.append((pred, node))
        if edges != []:
            srcs, dsts = zip(*edges)
            dglgraph_.remove_edges(dglgraph_.edge_ids(dsts, srcs))
        graph_.delete_edges(edges)
    seeds = list(seeds)
    return seeds
# ==================== END OF HELPER FUNCTIONS ====================

def compute_base_sensitivity(
        num_message_passing_steps, max_degree, batch_size, num_hops,
        upper_bound, dp_method, train_num=None,
        privacy_accountant=FIXED_BATCH_HYPERGEOMETRIC):
    """Returns the base sensitivity which is multiplied to the clipping threshold.
    Args:
    """
    if dp_method == "DPGNN4GC":
        max_terms_per_edge = batch_size
    elif dp_method == 'DP_ipd_0617':
        if train_num is None:
            raise ValueError("train_num is required for PrivIM sensitivity")
        max_terms_per_edge = sensitivity_occurrences(
            privacy_accountant, upper_bound, train_num, batch_size
        )
    # Lemma 2 gives Delta_g <= C * N_g.  The optimizer multiplies this
    # occurrence multiplier by the clipping bound C when adding noise.
    return float(max_terms_per_edge)

def account_privacy(num_message_passing_steps,
                    max_node_degree,
                    num_hops,
                    step_num,
                    batch_size,
                    train_num,
                    sigma,
                    orders,
                    target_delta=1e-5,
                    upper_bound=None,
                    privacy_accountant=FIXED_BATCH_HYPERGEOMETRIC,
                    ):
    # max_terms_per_edge = compute_max_terms_per_edge(num_message_passing_steps,
    #                                                 max_node_degree,
    #                                                 num_hops)
    max_terms_per_node = upper_bound
    # assert max_terms_per_node <= len(train_dataset), "#affected terms must <= #samples"
    rdp_every_epoch = multiterm_rdp(
        orders, step_num, sigma, train_num, max_terms_per_node, batch_size,
        accountant=privacy_accountant,
    )
    # rdp_every_epoch_org = compute_rdp(args.batch_size / len(train_dataset), args.sigma, 1 * epoch, orders)
    epsilon, best_alpha = rdp_to_dp(np.asarray(orders), rdp_every_epoch, target_delta)
    return epsilon, best_alpha


def get_noise_multiplier(
        target_epsilon: float,
        target_delta: float,
        step_num: int,
        orders: list,
        num_message_passing_steps,
        max_node_degree,
        num_hops,
        batch_size,
        train_num,
        upper_bound,
        privacy_accountant=FIXED_BATCH_HYPERGEOMETRIC,
        epsilon_tolerance: float = 0.01,
        dp_method="DPGNN4GC",
) -> float:
    r"""
    Computes the noise level sigma to reach a total budget of (target_epsilon, target_delta)
    at the end of epochs, with a given sample_rate
    Args:
        target_epsilon: the privacy budget's epsilon
        target_delta: the privacy budget's delta
        sample_rate: the sampling rate (usually batch_size / n_data)
        steps: number of steps to run
        epsilon_tolerance: precision for the binary search
    Returns:
        The noise level sigma to ensure privacy budget of (target_epsilon, target_delta)
    """

    sigma_low, sigma_high = 0, 100
    eps_high, best_alpha = account_privacy(num_message_passing_steps=num_message_passing_steps,
                                           max_node_degree=max_node_degree,
                                           num_hops=num_hops,
                                           step_num=step_num,
                                           batch_size=batch_size,
                                           train_num=train_num,
                                           sigma=sigma_high,
                                           target_delta=target_delta,
                                           orders=orders,
                                           upper_bound=upper_bound,
                                           privacy_accountant=privacy_accountant)

    if eps_high > target_epsilon:
        raise ValueError("The target privacy budget is too low. 当前可供搜索的最大的sigma只到100")

    while target_epsilon - eps_high > epsilon_tolerance:
        sigma = (sigma_low + sigma_high) / 2
        eps, best_alpha = account_privacy(
            num_message_passing_steps=num_message_passing_steps,
            max_node_degree=max_node_degree,
            num_hops=num_hops,
            step_num=step_num,
            batch_size=batch_size,
            train_num=train_num,
            sigma=sigma,
            target_delta=target_delta,
            orders=orders,
            upper_bound=upper_bound,
            privacy_accountant=privacy_accountant,
        )

        if eps < target_epsilon:
            sigma_high = sigma
            eps_high = eps
        else:
            sigma_low = sigma

    return sigma_high


def my_collate_fn(batch):
    return{'idx': [item['idx'] for item in batch],
           'edge': [item['edge'] for item in batch]}

def cal_target_delta(dataset_name, train_nodes=None):
    if dataset_name == "email":
        target_delta_value = 2e-3
    elif dataset_name == "bitcoin":
        target_delta_value = 4e-4
    elif dataset_name == "lastfm":
        target_delta_value = 3e-4
    elif dataset_name == "hepph":
        target_delta_value = 1e-4
    elif dataset_name == "facebook":
        target_delta_value = 9e-5
    elif dataset_name == "gowalla":
        target_delta_value = 1e-5

    if train_nodes:
        target_delta_value = min(target_delta_value, 0.9 / train_nodes)
    return target_delta_value


def _tensor_diagnostics(value):
    value = value.detach().float().reshape(-1)
    finite = torch.isfinite(value)
    finite_values = value[finite]
    if finite_values.numel() == 0:
        return {"finite_fraction": 0.0}
    return {
        "mean": float(finite_values.mean()),
        "std": float(finite_values.std(unbiased=False)),
        "min": float(finite_values.min()),
        "max": float(finite_values.max()),
        "positive_fraction": float((finite_values > 0).float().mean()),
        "zero_fraction": float((finite_values == 0).float().mean()),
        "finite_fraction": float(finite.float().mean()),
    }


def collect_model_diagnostics(net, probe_graph):
    """Collect initialization/convergence diagnostics on one fixed subgraph."""
    layer_outputs = {}
    handles = []

    def capture(name):
        def hook(_module, _inputs, output):
            layer_outputs[name] = _tensor_diagnostics(output)
        return hook

    for name, module in net.named_modules():
        if module.__class__.__name__ in {
            "GRATLayer", "LegacyGRATLayer", "LegacyLeakyGRATLayer"
        }:
            handles.append(module.register_forward_hook(capture(name)))

    was_training = net.training
    net.eval()
    try:
        with torch.no_grad():
            output = net(probe_graph, probe_graph.ndata["feat"])
    finally:
        for handle in handles:
            handle.remove()
        net.train(was_training)

    output_stats = _tensor_diagnostics(output)
    flattened_output = output.detach().float().reshape(-1)
    finite_output = flattened_output[torch.isfinite(flattened_output)]
    output_stats["sigmoid_saturation_fraction"] = (
        float(
            ((finite_output <= 1e-4) | (finite_output >= 1 - 1e-4))
            .float().mean()
        )
        if finite_output.numel() else None
    )
    squared_norm = 0.0
    max_abs = 0.0
    for parameter in net.parameters():
        detached = parameter.detach().float()
        squared_norm += float(torch.sum(detached * detached))
        if detached.numel():
            max_abs = max(max_abs, float(detached.abs().max()))
    return {
        "layers": layer_outputs,
        "output": output_stats,
        "parameter_l2_norm": squared_norm ** 0.5,
        "parameter_max_abs": max_abs,
    }

def train(net, dataset, loss_function, n_iteration, batchsize, lr, k_train=32,
          adj_matrices=None, graphs=None, dglgraphs=None, greedy_perfs=None, closed_edges=None, closed_graphs=None,
          model_name=None, d=None, seed=None, subgraph_size=None, num_subgraph=None,
          score_upper_bound=None, rate=None, settings=None, round=None, with_dp=False, dataset_name=None,
          target_epsilon=None, gamma=None, dir_name=None,gnns_name=None, K=None,
          use_cuda=True, train_node_count=None,
          privacy_accountant=FIXED_BATCH_HYPERGEOMETRIC,
          l2_norm_clip=3.0, sampler_metadata=None,
          experiment_metadata=None, output_root=None,
          lr_decay=1.0, lr_decay_steps=5, model_variant=None,
          model_initialization=None, batch_seed=0, sampling_seed=None,
          clip_non_private=False):

    if use_cuda:
        net.cuda()
    # ``gamma`` is retained as a compatibility keyword for older callers.
    if gamma is not None:
        lr_decay = gamma
    print(
        f"Learning rate: {lr:.6g}; StepLR every {lr_decay_steps} "
        f"updates with gamma={lr_decay:.6g}"
    )

    target_delta_value = cal_target_delta(dataset_name, train_node_count)

    avg_train_losses = []
    avg_train_perfs = []
    model_diagnostics = {
        "initial": collect_model_diagnostics(net, dglgraphs[0])
    }

    # Early stopping
    train_losses=[]
    train_perfs=[]
    early_stopping = EarlyStopping(patience=3, verbose=True)

    # # parmas for dp
    achieved_epsilon = None
    best_alpha = None
    sigma = None
    sensitivity_multiplier = None
    clip_gradients = with_dp or clip_non_private
    if with_dp:
        target_delta = target_delta_value
        print(f"Target epsilon: {target_epsilon}, target delta: {target_delta}")

        orders = np.arange(1, 100, 0.1)[1:]
        sigma = get_noise_multiplier(
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            step_num=n_iteration,
            orders=orders,
            num_message_passing_steps=2,
            max_node_degree=10,
            num_hops=2,
            batch_size=batchsize,
            train_num=len(dataset),
            upper_bound=score_upper_bound,
            privacy_accountant=privacy_accountant,
            dp_method="DP_ipd_0617")
        if settings == 'HP':
            W = np.random.exponential(scale=1.0)
            sigma = np.sqrt(W) * sigma
        print(f"Given target epsilon, sigma is set to:{sigma}")
        achieved_epsilon, best_alpha = account_privacy(
            num_message_passing_steps=2,
            max_node_degree=K,
            num_hops=d,
            step_num=n_iteration,
            batch_size=batchsize,
            train_num=len(dataset),
            sigma=sigma,
            orders=orders,
            target_delta=target_delta,
            upper_bound=score_upper_bound,
            privacy_accountant=privacy_accountant,
        )
        print(
            f"Accounted epsilon: {achieved_epsilon:.4f} "
            f"(delta={target_delta:.6g}, alpha={best_alpha:g})"
        )
        if achieved_epsilon > target_epsilon + 0.01:
            raise AssertionError("calibrated noise exceeds the requested privacy budget")
        sens = compute_base_sensitivity(max_degree=10,
                                        num_message_passing_steps=3,
                                        num_hops=2, batch_size=batchsize,
                                        upper_bound=score_upper_bound,
                                        train_num=len(dataset),
                                        privacy_accountant=privacy_accountant,
                                        dp_method='DP_ipd_0617')
        sensitivity_multiplier = sens
        print(f"All the noise: {sens * sigma}")
        optimizer = PrivateSGD(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=sens * sigma,
            minibatch_size=batchsize,
            microbatch_size=1,
            params=net.parameters(),
            lr=lr,
        )
    elif clip_non_private:
        # Non-Private is PrivIM* with epsilon -> infinity (Section V-A).
        # Algorithm 2 still clips each subgraph gradient; only its Gaussian
        # perturbation vanishes in this limit.
        optimizer = PrivateSGD(
            l2_norm_clip=l2_norm_clip,
            noise_multiplier=0.0,
            minibatch_size=batchsize,
            microbatch_size=1,
            params=net.parameters(),
            lr=lr,
        )
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=lr_decay_steps, gamma=lr_decay
    )

    if len(dataset) < batchsize:
        raise ValueError(
            f"batchsize ({batchsize}) must not exceed dataset size ({len(dataset)})"
        )

    # Algorithm 2 defines T as the total number of private optimizer
    # iterations.  At every iteration, draw one fresh, fixed-size batch
    # uniformly from the complete subgraph container.  This also matches the
    # fixed batch size used by the privacy accountant.
    batch_generator = torch.Generator()
    batch_generator.manual_seed(batch_seed)
    train_sampler = RandomSampler(
        dataset,
        replacement=False,
        num_samples=batchsize,
        generator=batch_generator,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=batchsize,
        sampler=train_sampler,
        collate_fn=my_collate_fn,
        generator=batch_generator,
    )
    timer_list = []
    diagnostic_records = []
    # best_model_wts = net.state_dict()
    # best_loss = float('inf')

    for iteration in range(n_iteration):
        timer = time.time()
        train_perfs = []
        train_losses = []

        batch_data = next(iter(train_loader))
        batch_idx = batch_data['idx']

        batch_adj = []
        batch_graph = []
        batch_dglgraph = []
        batch_greedy_perf = []
        batch_closed_edges = []
        batch_closed_graph = []

        for idx in batch_idx:
            batch_adj.append(adj_matrices[idx])
            batch_graph.append(graphs[idx])
            batch_dglgraph.append(dglgraphs[idx])
            batch_greedy_perf.append(greedy_perfs[idx])
            batch_closed_edges.append(closed_edges[idx])
            batch_closed_graph.append(closed_graphs[idx])

        loss_list = train_single(
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            loss_function=loss_function,
            batch_adj_matrices=batch_adj,
            batch_graphs=batch_graph,
            batch_dglgraphs=batch_dglgraph,
            batch_greedy_perfs=batch_greedy_perf,
            batch_closed_edges=batch_closed_edges,
            batch_closed_graphs=batch_closed_graph,
            k_train=k_train,
            iteration=iteration,
            clip_gradients=clip_gradients,
        )

        # train_perfs.append(perf)
        train_losses.append(sum(loss_list))
        
        train_loss = np.sum(train_losses)
        avg_train_losses.append(train_loss)
        avg_train_loss = train_loss / len(loss_list)



        # if train_loss < best_loss:
        #     best_loss = train_loss
        #     best_model_wts = net.state_dict()
        #
        # train_perf = np.average(train_perfs)
        # avg_train_perfs.append(train_perf)

        if iteration == 0 or (iteration + 1) % 10 == 0 or iteration + 1 == n_iteration:
            print(
                f"Train Iteration {iteration} | Loss: {avg_train_loss:.2f} | "
                f"Elapsed Time: {time.time() - timer:.2f}"
            )
        timer_list.append(time.time() - timer)
        record = {
            "iteration": iteration,
            "mean_loss": float(avg_train_loss),
            "elapsed_seconds": float(timer_list[-1]),
        }
        if clip_gradients:
            norms = optimizer.last_microbatch_norms
            record.update({
                "mean_microbatch_grad_norm": (
                    float(np.mean(norms)) if norms else None
                ),
                "max_microbatch_grad_norm": (
                    float(np.max(norms)) if norms else None
                ),
                "clip_fraction": float(optimizer.last_clip_fraction),
            })
        diagnostic_records.append(record)

        # early_stopping(train_loss, net)
        # early_stopping(-train_perf, net)
        # if early_stopping.early_stop:
        #     logging.info("Early stopping")
        #     break

    model_diagnostics["final"] = collect_model_diagnostics(net, dglgraphs[0])

    # 保存权重
    epsilon_dir = f"eps_{target_epsilon}" if with_dp else "eps_10000"
    output_root = Path(output_root) if output_root else PROJECT_ROOT / "outputs"
    model_dir = (
        output_root
        / dataset_name
        / settings
        / epsilon_dir
        / f"n_{subgraph_size}"
        / f"Ng_{score_upper_bound}"
        / "model"
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / (
        f"model_{model_name}_d_{d}_seed_{seed}_round_{round}.pt"
    )
    if with_dp:
        torch.save(net.state_dict(), model_path)
    else:
        torch.save(net.state_dict(), model_path)
    metadata = {
        "dataset": dataset_name,
        "setting": settings,
        "model": model_name,
        "model_variant": model_variant,
        "model_initialization": model_initialization,
        "feature_protocol": (experiment_metadata or {}).get(
            "feature_protocol", "ones"
        ),
        "input_dim": (experiment_metadata or {}).get("input_dim"),
        "seed": seed,
        "model_seed": seed,
        "sampling_seed": sampling_seed,
        "batch_seed": batch_seed,
        "round": round,
        "d": d,
        "train_nodes": train_node_count,
        "subgraph_size": subgraph_size,
        "subgraph_container_size": len(dataset),
        "occurrence_bound": score_upper_bound,
        "batch_size": batchsize,
        "iterations": n_iteration,
        "learning_rate": lr,
        "learning_rate_schedule": {
            "name": "StepLR",
            "gamma": lr_decay,
            "step_size_updates": lr_decay_steps,
            "final_learning_rate": optimizer.param_groups[0]["lr"],
        },
        "loss_lambda": getattr(loss_function, "C", None),
        "with_dp": with_dp,
        "non_private_clipping": bool(clip_non_private and not with_dp),
        "gradient_reduction": (
            "clipped_per_subgraph_mean" if clip_gradients else "batch_mean"
        ),
        "target_epsilon": target_epsilon if with_dp else None,
        "target_delta": target_delta_value if with_dp else None,
        "privacy_accountant": privacy_accountant if with_dp else None,
        "achieved_epsilon": achieved_epsilon if with_dp else None,
        "best_alpha": best_alpha if with_dp else None,
        "clip_norm": l2_norm_clip if clip_gradients else None,
        "sigma": sigma if with_dp else None,
        "sensitivity_occurrences": sensitivity_multiplier if with_dp else None,
        "gradient_sensitivity": (
            l2_norm_clip * sensitivity_multiplier if with_dp else None
        ),
        "noise_std_on_gradient_sum": (
            l2_norm_clip * sensitivity_multiplier * sigma if with_dp else None
        ),
        "noise_std_after_batch_average": (
            l2_norm_clip * sensitivity_multiplier * sigma / batchsize
            if with_dp else None
        ),
        "optimizer": "SGD",
        "momentum": 0.0,
        "sampling": sampler_metadata or {},
        "experiment": experiment_metadata or {},
        "training_diagnostics": diagnostic_records,
        "model_diagnostics": model_diagnostics,
        "mean_iteration_seconds": float(np.mean(timer_list)),
        "total_training_seconds": float(np.sum(timer_list)),
    }
    model_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Time per iteration: {np.mean(timer_list)}")
    # torch.save(net.state_dict(), f=f"D:/demo/result/d_{d}/kr/size_300_350/K_{K}_r_{r}/num_{num_subgraph}/model/model_{model_name}_d_{d}_seed_{seed}.pt")
    # net.load_state_dict(best_model_wts)
    # torch.save(net.state_dict(), f=f"D:/demo/fastCover-master/output/params/params_model_GRAT3_d_3_seed_42_dp_eps_0.1.pt")
    return avg_train_losses, avg_train_perfs, net

def train_single(net, optimizer, scheduler, loss_function, batch_adj_matrices,
                 batch_graphs, batch_dglgraphs, batch_greedy_perfs,
                 batch_closed_edges, batch_closed_graphs, k_train, iteration,
                 clip_gradients=True):
    """
    单个batch的训练
    """
    loss_list = []

    adj_mats = batch_adj_matrices
    graphs = batch_graphs
    dglgraphs = batch_dglgraphs
    greedy_perfs = batch_greedy_perfs
    closed_graphs = batch_closed_graphs if batch_closed_graphs is not None else None

    net.train()
    if clip_gradients:
        optimizer.zero_accum_grad()
        total_loss = 0
        # optimizer.zero_grad()

        for dglgraph, adj_mat, graph, greedy_perf, closed_graph in zip(dglgraphs, adj_mats, graphs, greedy_perfs, closed_graphs):
            optimizer.zero_microbatch_grad()
            out = net(dglgraph, dglgraph.ndata['feat']).squeeze(1)
            loss = loss_function(out, adj_mat, k_train)
            # total_loss += loss
            loss.backward()
            optimizer.microbatch_step()
            loss_list.append(loss.item())
        optimizer.step()
        scheduler.step()
    else:
        # optimizer.zero_accum_grad()
        total_loss = 0
        optimizer.zero_grad()

        batch_count = len(dglgraphs)
        for dglgraph, adj_mat, graph, greedy_perf, closed_graph in zip(dglgraphs, adj_mats, graphs, greedy_perfs,
                                                                       closed_graphs):
            # optimizer.zero_microbatch_grad()
            out = net(dglgraph, dglgraph.ndata['feat']).squeeze(1)
            loss = loss_function(out, adj_mat, k_train)
            # total_loss += loss
            (loss / batch_count).backward()
            # optimizer.microbatch_step()
            loss_list.append(loss.item())
        # optimizer.step_dp()
        optimizer.step()
        scheduler.step()

    # scheduler.step()
    # total_loss.backward()
    # optimizer.step()
    # optimizer.zero_grad()

    # scheduler.step()


    # epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
    # print(f"Train epsilon: {epsilon:.2f}, best alpha: {best_alpha}")
    #
    # net.eval()
    # perf_list = []
    # for dglgraph, adj_mat, graph, greedy_perf, closed_graph in zip(dglgraphs, adj_mats, graphs, greedy_perfs, closed_graphs):
    #     logits = net(dglgraph, dglgraph.ndata['feat']).squeeze(1)
    #     _, indices = torch.topk(logits, k_train)
    #
    #     if closed_graph is None:
    #         train_perf = get_influence(graph, indices)
    #     else:
    #         train_perf = get_influence(closed_graph, indices)
    #     perf_ratio = train_perf/greedy_perf
    #     perf_list.append(perf_ratio)
    #     # print(f"Train influence: {train_perf}/{greedy_perf}={perf_ratio:.2f}")

    # final_perf = np.average(perf_list)

    return loss_list

def test_single(net, graph_name, graph, dglgraph, ks):
    n_node = graph.vcount()
    n_edge = graph.ecount()
    print(f"Testing on the graph {graph_name}.")
  
    greedy_ts = []  # timing
    greedy_perfs = []
    nn_t0s = []
    nn_t1s = []
    nn_ts = []
    nn_perfs = []
    memories = []

    for k in ks:
        # greedy
        if k <= n_node:
            t0 = time.time()
            _, greedy_perf = greedy(graph, k)
            t1 = time.time()  # t1 - t0: calculate greedy influence

            out = net.grat(dglgraph, dglgraph.ndata['feat']).squeeze(
        1)
            t2 = time.time()  # t2 - t1: nn
            _, nn_seeds = torch.topk(out, k)
            t3 = time.time()  # t3 - t2: get top k seeds.

            nn_perf = get_influence(graph, nn_seeds)  # we move it out
            memory = get_memory()

            greedy_time = t1 - t0
            nn_time_0 = t2 - t1
            nn_time_1 = t3 - t2
            nn_time = nn_time_0 + nn_time_1

            logging.info(
                f"n_node:{n_node}; n_edge:{n_edge}; k:{k}; perf ratio:{nn_perf/greedy_perf:.2f}; time ratio: {nn_time/greedy_time:.2f}; topk ratio: {nn_time_1/nn_time:.2f}"
            )
        
            greedy_ts.append(greedy_time)
            greedy_perfs.append(greedy_perf)
            nn_t0s.append(nn_time_0)
            nn_t1s.append(nn_time_1)
            nn_ts.append(nn_time)
            nn_perfs.append(nn_perf)
            memories.append(memory)
        else:
            greedy_ts.append(-1)
            greedy_perfs.append(-1)
            nn_t0s.append(-1)
            nn_t1s.append(-1)
            nn_ts.append(-1)
            nn_perfs.append(-1)     
            memories.append(-1)       

    result = {
        "graph_name": graph_name,
        "n": n_node,
        "m": n_edge,
        "k": ks,
        "greedy_perf": greedy_perfs,
        "greedy_time": greedy_ts,
        "nn_perf": nn_perfs,
        "nn_time": nn_ts,
        "nn_time_0": nn_t0s,
        "nn_time_1": nn_t1s,
        "memory": memories,
    }

    return result
