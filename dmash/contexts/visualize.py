import json
import glob

import numpy as np

from sklearn.manifold import TSNE

from dmash.contexts.util import compute_context
from dmash.common import plotting


def get_tsne(context, label):
    tsne = TSNE(n_components=2, random_state=0)
    context_low = tsne.fit_transform(context)

    fig = plotting.representation_fig(context_low, label, "TSNE")
    return fig


def prepare_contexts(ds, dce, rce, device, args, training=True, context_info=None):
    dynamics_contexts = []
    reward_contexts = []
    labels = []
    true_contexts = []

    for i in range(ds.num_datasets):
        data = ds.sample(args.batch_size, dataset_index=i)
        dynamics_context, reward_context = compute_context(dce, rce, data, device, args, training)
        dynamics_contexts.append(dynamics_context.detach().cpu().numpy())
        reward_contexts.append(reward_context.detach().cpu().numpy() if args.compute_reward_context else None)
        labels.append(np.ones((dynamics_contexts[0].shape[0], 1), dtype=int) * i)
        if context_info is not None:
            true_contexts.append(np.array([list(context_info[i].values())] * args.batch_size, dtype=np.float32))

    dynamics_context = np.concatenate(dynamics_contexts, axis=0)
    reward_context = np.concatenate(reward_contexts, axis=0) if args.compute_reward_context else None
    label = np.concatenate(labels, axis=0)
    if context_info is not None:
        true_contexts = np.concatenate(true_contexts, axis=0)
    return dynamics_context, reward_context, label, true_contexts


def prepare_runs(env_id, project_id, context_id, method_filter, task_filter):
    runs = []
    metric_dirs = glob.glob(f"/Users/jd/Repos/dmash/dmash/contexts/metrics/{project_id}/*{env_id}*")
    for metric_dir in metric_dirs:
        metrics = []
        filename = f"{metric_dir}/metrics.json"
        with open(filename, 'rb') as f:
            for lines in f:
                metrics.append(json.loads(lines))

        filename = f"{metric_dir}/metadata.json"
        with open(filename, 'rb') as f:
            metadata = json.load(f)

        for task_id, task_name in zip(
            # ["rl_returns_eval/eval_train_return", "rl_returns_eval/eval_in_return", "rl_returns_eval/eval_out_return"],
            # ["Training contexts", "Eval-in contexts", "Eval-out contexts"],
            ["rl_returns_eval/eval_train_return", "rl_returns_eval/eval_in_return", "rl_returns_eval/eval_out_return"],
            ["Training", "Eval-in", "Eval-out"],
        ):
            if metadata["context_id"] != context_id:
                continue
            if metadata["num_train_envs"] != 10:
                continue

            xs = [m["step"] for m in metrics]
            ys = [m[task_id] for m in metrics]

            if not metadata["context_use"]:
                if metadata["context_default"]:
                    method = "unaware, default (*)"
                elif not metadata["context_default"]:
                    method = "unaware, dr"
            elif metadata["context_use"] and not metadata["context_default"]:
                if metadata["context_aware"] and metadata["context_aware_onehot"]:
                    method = "aware, onehot"
                elif metadata["context_aware"] and not metadata["context_aware_onehot"]:
                    method = "aware"
                elif not metadata["context_aware"] and metadata["context_encoder"] in ["Transformer"]:
                    method = "unaware, inferred"
                else:
                    method = "none"
            else:
                method = "none"

            if method not in method_filter:
                continue
            if task_name not in task_filter:
                continue

            run_dict = {
                "method": method,
                "seed": metadata["seed"],
                "task": task_name,
                "xs": xs,
                "ys": ys
            }
            runs.append(run_dict)
    return runs


def save_returns(env_ids, project_id, context_id, method_filter, task_filter, figsize, suffix, xlim=None, ylim=None, titles=None):
    # modified_colors = ["k", "k", "k", "r"]
    # modified_linestyles = ["--", ":", "-", "-"]
    modified_colors = {"aware": "k", "unaware, default (*)": "k", "unaware, dr": "k", "unaware, inferred": "r"}
    modified_linestyles = {"aware": "--", "unaware, default (*)": ":", "unaware, dr": "-", "unaware, inferred": "-"}
    for env_id in env_ids:
        runs = prepare_runs(env_id, project_id, context_id, method_filter, task_filter)
        xlim = (0, max(runs[0]["xs"])) if xlim is None else xlim
        bins = np.linspace(*xlim, 30 + 1, endpoint=True)
        # tasks = ["Training contexts", "Eval-in contexts", "Eval-out contexts"]
        tasks = task_filter

        tensor, tasks, methods, seeds = plotting.tensor(runs, bins, tasks=tasks)

        fig, axes = plotting.plots(len(tasks), cols=len(tasks), size=figsize)

        if titles is not None:
            assert len(titles) == len(tasks)

        for i, task in enumerate(tasks if titles is None else titles):
            ax = axes[i]
            ax.set_title(task)
            # ax.set_xlim(xlim[0], len(runs[0]["xs"]) * (runs[0]["xs"][1] - runs[0]["xs"][0]))
            ax.set_xlim(*xlim)
            ax.xaxis.set_major_formatter(plotting.smart_format)
            if ylim is not None:
                ax.set_ylim(*ylim)

            for j, method in enumerate(methods):
                # Aggregate over seeds.
                mean = np.nanmean(tensor[i, j, :, :], 0)
                std = np.nanstd(tensor[i, j, :, :], 0)
                plotting.curve(
                    ax,
                    bins[1:],
                    mean,
                    low=mean + std / 2,
                    high=mean - std / 2,
                    label=method,
                    order=j,
                    # color=plotting.COLORS[j],
                    linestyle=modified_linestyles[method],
                    color=modified_colors[method]
                )
        plotting.legend(fig, adjust=True, plotpad=0.3)

        plotting.save(fig, f"figures/{project_id}/{env_id}_{suffix}")

