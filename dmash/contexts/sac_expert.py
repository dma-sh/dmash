import random
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import gymnasium as gym
import tyro
import wandb

from dmash.contexts.context import setup_context_env
from dmash.contexts.dataset import ReplayBuffer
from dmash.contexts.model import (SoftQNetwork, Actor)
from dmash.contexts.config import Args, modify_groups
from dmash.common.logger import Logger, JSONLOutput, WandBOutput, TerminalOutput


def make_evaluation(args, actor, envs, device):
    # Reset queues
    for _ in range(args.num_eval_episodes):
        obs, _ = envs.reset()
        done = False
        dynamics_context_eval = None
        while not done:
            actions, _, _ = actor.get_action(
                torch.Tensor(obs).to(device), dynamics_context_eval, training=False
            )
            actions = actions.detach().cpu().numpy()

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            obs = next_obs
            done = any(np.logical_or(terminations, truncations))
            if done and "is_success" in infos["episode"]:
                for i in range(envs.num_envs):
                    envs.envs[i].is_success_queue.append(infos["episode"]["is_success"][i])

    # track episodic returns for individual envs and averaged over all envs
    episodic_returns = []
    for i in range(envs.num_envs):
        episodic_return = np.mean(envs.envs[i].return_queue)
        episodic_returns.append(episodic_return)
    return episodic_returns


def train():
    args = tyro.cli(Args)
    args = modify_groups(args)
    # expert case
    args.context_mode = "unaware"
    args.context_encoder_update = ["actor", "critic"]  # to circumvent detach of None

    wandb_id = wandb.util.generate_id()
    run_name = f"sac__{args.env_id.split('/')[-1]}__{args.seed}__{int(time.time())}__{wandb_id}"
    logger_types = []
    if args.wandb:
        logger_types.append(
            WandBOutput(
                project=args.wandb_project_name,
                config=vars(args),
                mode="offline" if args.wandb_offline else "online",
                id=wandb_id
            ),
        )
    if args.verbose:
        logger_types.append(
            TerminalOutput(
                pattern="rl_returns_train/|rl_returns_eval/|fps"
            ),
        )
    if args.write:
        metrics_filename = f"metrics/{args.wandb_project_name}/{run_name}/metrics.json"
        logger_types.append(
            JSONLOutput(filename=metrics_filename, config=vars(args)),
        )
    logger = Logger(0, logger_types)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    _, eval_train_envs, eval_in_envs, eval_out_envs, context_info = setup_context_env(args, run_name)

    rl_returns_eval = {}

    for log_name, train_envs in zip(
        ["eval_train", "eval_in", "eval_out"],
        [eval_train_envs, eval_in_envs, eval_out_envs]
    ):
        for env_i in range(train_envs.num_envs):
            envs = gym.vector.SyncVectorEnv(
                [train_envs.env_fns[env_i]],
                autoreset_mode=gym.vector.AutoresetMode.DISABLED
            )
            actor = Actor(envs, args).to(device)
            qf1 = SoftQNetwork(envs, args).to(device)
            qf2 = SoftQNetwork(envs, args).to(device)
            qf1_target = SoftQNetwork(envs, args).to(device)
            qf2_target = SoftQNetwork(envs, args).to(device)
            qf1_target.load_state_dict(qf1.state_dict())
            qf2_target.load_state_dict(qf2.state_dict())
            q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
            actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

            # Automatic entropy tuning
            if args.autotune:
                target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
                log_alpha = torch.zeros(1, requires_grad=True, device=device)
                alpha = log_alpha.exp().item()
                a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
            else:
                alpha = args.alpha

            rb = ReplayBuffer(
                envs.single_observation_space,
                envs.single_action_space,
                min(args.buffer_size, args.total_timesteps),
            )
            rb.seed(args.seed)

            # TRY NOT TO MODIFY: start the game
            done = True  # used for warm-up
            obs, _ = envs.reset()
            for step in np.arange(-args.learning_starts, args.total_timesteps + 1, dtype=int):
                # ALGO LOGIC: put action logic here
                if step < 0:
                    actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
                else:
                    actions, _, _ = actor.get_action(
                        torch.Tensor(obs).to(device), None, training=True
                    )
                    actions = actions.detach().cpu().numpy()

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, rewards, terminations, truncations, infos = envs.step(actions)

                rb.insert(
                    dict(
                        observations=obs[-1],
                        next_observations=next_obs[-1],
                        actions=actions[-1],
                        rewards=rewards[-1],
                        masks=np.logical_not(terminations)[-1],
                        dones=np.logical_or(terminations, truncations)[-1],
                    ),
                )

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                obs = next_obs
                done = any(np.logical_or(terminations, truncations))
                if done:
                    obs, _ = envs.reset()

                # ALGO LOGIC: training.
                if step >= 0:
                    data = rb.sample(args.batch_size)
                    dynamics_context = None

                    actions = torch.tensor(data["actions"]).to(device)
                    observations = torch.tensor(data["observations"]).to(device)
                    next_observations = torch.tensor(data["next_observations"]).to(device)
                    dones = torch.tensor(data["dones"]).to(device)
                    rewards = torch.tensor(data["rewards"]).to(device)

                    # critic update
                    with torch.no_grad():
                        next_state_actions, next_state_log_pi, _ = actor.get_action(
                            next_observations, dynamics_context, training=True
                        )
                        qf1_next_target = qf1_target(next_observations, next_state_actions, dynamics_context)
                        qf2_next_target = qf2_target(next_observations, next_state_actions, dynamics_context)
                        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                        next_q_value = rewards.flatten() + (~dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

                    qf1_a_values = qf1(observations, actions, dynamics_context).view(-1)
                    qf2_a_values = qf2(observations, actions, dynamics_context).view(-1)
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                    q_optimizer.zero_grad()
                    qf_loss.backward()
                    q_optimizer.step()

                    # actor update
                    if (step == 0) or (step % args.policy_frequency == 0):  # TD 3 Delayed update support
                        for _ in range(
                            args.policy_frequency
                        ):  # compensate for the delay by doing 'actor_update_interval' instead of 1

                            pi, log_pi, _ = actor.get_action(
                                observations, dynamics_context, training=True
                            )
                            qf1_pi = qf1(observations, pi, dynamics_context)
                            qf2_pi = qf2(observations, pi, dynamics_context)
                            min_qf_pi = torch.min(qf1_pi, qf2_pi)
                            actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                            actor_optimizer.zero_grad()
                            actor_loss.backward()
                            actor_optimizer.step()

                            # alpha update
                            if args.autotune:
                                with torch.no_grad():
                                    _, log_pi, _ = actor.get_action(
                                        observations, dynamics_context, training=True
                                    )
                                alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                                a_optimizer.zero_grad()
                                alpha_loss.backward()
                                a_optimizer.step()
                                alpha = log_alpha.exp().item()

                    # target networks update
                    if step % args.target_network_frequency == 0:
                        for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                        for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                            target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

                    # make evaluation and logging
                    if step % 5000 == 0:
                        eval_actor = deepcopy(actor)
                        eval_envs = deepcopy(envs)

                        rl_returns_eval_i = {}
                        eval_episodic_returns = make_evaluation(
                            args,
                            eval_actor,
                            eval_envs,
                            device,
                        )
                        if step not in rl_returns_eval:
                            rl_returns_eval[step] = {}
                        rl_returns_eval[step][f"{log_name}_{env_i}_return"] = eval_episodic_returns[-1].item()

    for step, rl_returns_eval_i in rl_returns_eval.items():
        rl_returns_eval_mean = {}
        for log_name in ["eval_train", "eval_in", "eval_out"]:
            rl_returns_eval_mean[f"{log_name}_return"] = np.array(
                [v for k, v in rl_returns_eval_i.items() if log_name in k]
            ).mean().item()
        logger.add(rl_returns_eval_mean, prefix="rl_returns_eval", step=step)
        logger.add(rl_returns_eval_i, prefix="rl_returns_eval_i", step=step)
        logger.write()

    if args.wandb:
        wandb.run.finish()
    if args.wandb and args.wandb_offline:
        import subprocess
        import glob

        assert wandb_id is not None, "Provide wandb_id."
        offline_wandb_dir = glob.glob(f"./wandb/offline-run*{wandb_id}")
        assert len(offline_wandb_dir) == 1, "Offline wandb id is not unique."
        subprocess.run(["wandb", "sync", "--include-offline", offline_wandb_dir[0]])

    envs.close()


if __name__ == "__main__":
    train()
