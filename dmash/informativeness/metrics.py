from copy import deepcopy

import numpy as np
import torch

from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from dmash.contexts.util import compute_context
from dmash.informativeness.util import compute_mi_cc
from dmash.disentanglement import dci


def compute_metrics(ds, dce, rce, actor, qf, fm, im, device, args, context_info, training=True, only_some=True):
    actor = deepcopy(actor)
    qf = deepcopy(qf)
    dce = deepcopy(dce)
    rce = deepcopy(rce)
    fm = deepcopy(fm)
    im = deepcopy(im)

    track_dynamics_context = []
    track_true_context = []
    track_qf_main = []
    track_qf_mnet = []
    track_qf_hnet = []
    track_actor_main = []
    track_actor_mnet = []
    track_actor_hnet = []
    track_fm_main = []
    track_fm_mnet = []
    track_fm_hnet = []
    track_im_main = []
    track_im_mnet = []
    track_im_hnet = []

    with torch.no_grad():
        for i in range(ds.num_datasets):
            data = ds.sample(args.batch_size, dataset_index=i)
            dynamics_context, reward_context = compute_context(dce, rce, data, device, args, training)
            track_dynamics_context.append(dynamics_context.detach().cpu().numpy())
            track_true_context.append(np.array([list(context_info[i].values())] * args.batch_size, dtype=np.float32))

            # run models
            actions = torch.tensor(data["actions"]).to(device)[:, -1]
            observations = torch.tensor(data["observations"]).to(device)[:, -1]
            next_observations = torch.tensor(data["next_observations"]).to(device)[:, -1]

            if fm is not None:
                _ = fm(observations, actions, dynamics_context)
            if im is not None:
                _ = im(observations, next_observations, dynamics_context)
            if (
                args.q_context_merge_type == "hypernet_shared" or
                args.policy_context_merge_type == "hypernet_shared"
            ):
                if args.context_mode == "aware":
                    hnet_weights = actor.get_hnet_weights(
                        context=dynamics_context,
                        obs=observations
                    )
                elif fm is not None:
                    hnet_weights = fm.get_hnet_weights(
                        context=dynamics_context,
                        obs=observations
                    )
                elif im is not None:
                    hnet_weights = im.get_hnet_weights(
                        context=dynamics_context,
                        obs=observations
                    )
                else:
                    raise NotImplementedError()
            else:
                hnet_weights = None
            _, _, _ = actor.get_action(observations, dynamics_context, hnet_weights)
            _ = qf(observations, actions, dynamics_context, hnet_weights)

            # track activities
            track_qf_main.append(qf.activation_dict["main"].detach().cpu().numpy())
            track_qf_mnet.append(qf.activation_dict["mnet"].detach().cpu().numpy() if "mnet" in qf.activation_dict.keys() else None)
            track_qf_hnet.append(qf.activation_dict["hnet"].detach().cpu().numpy() if "hnet" in qf.activation_dict.keys() else None)
            track_actor_main.append(actor.activation_dict["main"].detach().cpu().numpy())
            track_actor_mnet.append(actor.activation_dict["mnet"].detach().cpu().numpy() if "mnet" in actor.activation_dict.keys() else None)
            track_actor_hnet.append(actor.activation_dict["hnet"].detach().cpu().numpy() if "hnet" in actor.activation_dict.keys() else None)
            if fm is not None:
                track_fm_main.append(fm.activation_dict["main"].detach().cpu().numpy())
                track_fm_mnet.append(fm.activation_dict["mnet"].detach().cpu().numpy() if "mnet" in fm.activation_dict.keys() else None)
                track_fm_hnet.append(fm.activation_dict["hnet"].detach().cpu().numpy() if "hnet" in fm.activation_dict.keys() else None)
            if im is not None:
                track_im_main.append(im.activation_dict["main"].detach().cpu().numpy())
                track_im_mnet.append(im.activation_dict["mnet"].detach().cpu().numpy() if "mnet" in im.activation_dict.keys() else None)
                track_im_hnet.append(im.activation_dict["hnet"].detach().cpu().numpy() if "hnet" in im.activation_dict.keys() else None)

    track_dynamics_context = np.concatenate(track_dynamics_context, axis=0)
    track_true_context = np.concatenate(track_true_context, axis=0)

    track_qf_main = np.concatenate(track_qf_main, axis=0)
    track_qf_mnet = np.concatenate(track_qf_mnet, axis=0) if not track_qf_mnet[0] is None else None
    track_qf_hnet = np.concatenate(track_qf_hnet, axis=0) if not track_qf_hnet[0] is None else None
    track_actor_main = np.concatenate(track_actor_main, axis=0)
    track_actor_mnet = np.concatenate(track_actor_mnet, axis=0) if not track_actor_mnet[0] is None else None
    track_actor_hnet = np.concatenate(track_actor_hnet, axis=0) if not track_actor_hnet[0] is None else None

    if fm is not None:
        track_fm_main = np.concatenate(track_fm_main, axis=0)
        track_fm_mnet = np.concatenate(track_fm_mnet, axis=0) if not track_fm_mnet[0] is None else None
        track_fm_hnet = np.concatenate(track_fm_hnet, axis=0) if not track_fm_hnet[0] is None else None
    if im is not None:
        track_im_main = np.concatenate(track_im_main, axis=0)
        track_im_mnet = np.concatenate(track_im_mnet, axis=0) if not track_im_mnet[0] is None else None
        track_im_hnet = np.concatenate(track_im_hnet, axis=0) if not track_im_hnet[0] is None else None

    y_train = track_true_context
    r2_informativeness = {}
    mi_informativeness = {}
    mi_af_informativeness = {}
    mi_naf_informativeness = {}
    dci_informativeness = {}
    dci_rf_informativeness = {}
    variability = {}
    context_cosim = {}
    if only_some:
        inputs = [track_dynamics_context, track_actor_hnet, track_fm_hnet]
        input_names = ["context_representation", "actor_hnet", "fm_hnet"]
    else:
        inputs = [
            track_dynamics_context, track_qf_main, track_qf_mnet, track_qf_hnet,
            track_actor_main, track_actor_mnet, track_actor_hnet,
            track_fm_main, track_fm_mnet, track_fm_hnet,
            track_im_main, track_im_mnet, track_im_hnet,
        ]
        input_names = [
            "context_representation", "qf_main", "qf_mnet", "qf_hnet",
            "actor_main", "actor_mnet", "actor_hnet",
            "fm_main", "fm_mnet", "fm_hnet",
            "im_main", "im_mnet", "im_hnet",
        ]
    for x_train, x_name in zip(inputs, input_names):
        if x_train is None or len(x_train) == 0:
            continue
        if "hnet" in x_name:
            pca = PCA(args.helper_hidden_dim, random_state=args.seed).fit(x_train)
            x_train = pca.transform(x_train)

        # variability
        variability[x_name] = x_train.var(axis=0).mean().item()
        # context cosine similarity
        mean_context_representation = x_train.reshape(ds.num_datasets, args.batch_size, -1).mean(1)
        context_cosim[x_name] = cosine_similarity(mean_context_representation).mean()
        # r2
        reg = LinearRegression().fit(x_train, y_train)
        r2_informativeness[x_name] = reg.score(x_train, y_train)
        # mi
        mi_informativeness[x_name] = compute_mi_cc(x_train, y_train)
        if "action_factor" in context_info[0]:
            # index of action_factor
            af_index = np.nonzero(np.array(list(context_info[0].keys())) == "action_factor")[0].item()
            mi_af_informativeness[x_name] = compute_mi_cc(x_train, y_train[:, af_index].reshape(-1, 1))
            mi_naf_informativeness[x_name] = compute_mi_cc(x_train, y_train[:, af_index ^ 1].reshape(-1, 1))
        # dci_i
        _, _, dci_i = dci(y_train, x_train)
        dci_informativeness[x_name] = dci_i.item()
        _, _, dci_i = dci(y_train, x_train, model="random_forest")
        dci_rf_informativeness[x_name] = dci_i.item()

    return variability, context_cosim, r2_informativeness, mi_informativeness, mi_af_informativeness, mi_naf_informativeness, dci_informativeness, dci_rf_informativeness

