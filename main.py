import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.envs import *
from a2c_ppo_acktr.model import Policy, StatePolicy
from a2c_ppo_acktr.storage import RolloutStorage
from evaluation import evaluate

def main():
    args = get_args()

    if args.state:
        assert args.algo == "ppo"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"
    utils.cleanup_log_dir(log_dir)
    utils.cleanup_log_dir(eval_log_dir)

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    if args.load_dir:

        actor_critic, obsrms = torch.load(args.load_dir)
        vec_norm = utils.get_vec_normalize(envs)
        if vec_norm is not None:
            vec_norm.train()
            vec_norm.ob_rms = obsrms
        actor_critic.base.deterministic = args.deterministic
        actor_critic.base.humanoid = args.env_name.startswith("SH")

    else:
        if args.state:
            actor_critic = StatePolicy(
                envs.observation_space.shape,
                envs.action_space,
                base_kwargs={'recurrent': args.recurrent_policy,
                             'deterministic': args.deterministic,
                             'hidden_size': args.code_size,
                             'humanoid': args.env_name.startswith("SH")})
        else:
            actor_critic = Policy(
                envs.observation_space.shape,
                envs.action_space,
                base_kwargs={'recurrent': args.recurrent_policy},)
        actor_critic.to(device)

    if args.state:
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            beta=args.beta,
            beta_end=args.beta_end,
            state=True)
    elif args.algo == 'a2c':
        agent = algo.A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            alpha=args.alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = algo.A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    if args.gail:
        assert len(envs.observation_space.shape) == 1
        discr = gail.Discriminator(
            envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
            device)
        file_name = os.path.join(
            args.gail_experts_dir, "trajs_{}.pt".format(
                args.env_name.split('-')[0].lower()))
        
        expert_dataset = gail.ExpertDataset(
            file_name, num_trajectories=4, subsample_frequency=20)
        drop_last = len(expert_dataset) > args.gail_batch_size
        gail_train_loader = torch.utils.data.DataLoader(
            dataset=expert_dataset,
            batch_size=args.gail_batch_size,
            shuffle=True,
            drop_last=drop_last)


    # A bunch of tensors; circular buffer

    if args.state:

        rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                  envs.observation_space.shape, envs.action_space,
                                  actor_critic.recurrent_hidden_state_size, code_size=args.code_size)

        mis = []

    else:
        rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                  envs.observation_space.shape, envs.action_space,
                                  actor_critic.recurrent_hidden_state_size)

    # Populate the first observation in rollouts
    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    # Rewards is a deque
    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes

    for j in range(num_updates):
        # print(j)
        agent.ratio = j / num_updates
        if args.use_linear_lr_decay:
            # decrease learning rate linearly
            utils.update_linear_schedule(
                agent.optimizer, j, num_updates,
                agent.optimizer.lr if args.algo == "acktr" else args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                if args.state:
                    value, action, action_log_prob, recurrent_hidden_states, eps, code = actor_critic.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

                    # import ipdb; ipdb.set_trace()

                else:
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

            # Obs reward and next obs
            obs, reward, done, infos = envs.step(action)
            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            if args.state:
                rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks, eps=eps, code=code)
            else:
                rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        if args.gail:
            if j >= 10:
                envs.venv.eval()

            gail_epoch = args.gail_epoch
            if j < 10:
                gail_epoch = 100  # Warm up
            for _ in range(gail_epoch):
                discr.update(gail_train_loader, rollouts,
                             utils.get_vec_normalize(envs)._obfilt)

            for step in range(args.num_steps):
                rollouts.rewards[step] = discr.predict_reward(
                    rollouts.obs[step], rollouts.actions[step], args.gamma,
                    rollouts.masks[step])

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)


        value_loss, action_loss, dist_entropy, mi_loss = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + "_" + str(j) + ".pt"))

            torch.save([
                actor_critic,
                getattr(utils.get_vec_normalize(envs), 'ob_rms', None)
            ], os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()

            if args.state:
                print("DKL loss " + str(mi_loss))
                mis.append(mi_loss)

            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

        if (args.eval_interval is not None and len(episode_rewards) > 1
                and j % args.eval_interval == 0):
            ob_rms = utils.get_vec_normalize(envs).ob_rms

            if args.env_name.startswith("SH"):
                masses = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.25, 1.35, 1.45, 1.55]
                damps = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.25, 1.35, 1.45, 1.55]
                means = np.zeros((len(masses), len(damps)))
                stds = np.zeros((len(masses), len(damps)))
                for m_i in range(len(masses)):
                    for d_i in range(len(damps)):
                        m = masses[m_i]
                        d = masses[d_i]
                        u, s = evaluate(actor_critic, ob_rms,
                                        'OracleSHTest' + str(m) + "_" + str(d) + '-v0',
                                        args.seed, args.num_processes, eval_log_dir, device)
                        means[m_i, d_i] = u
                        stds[m_i, d_i] = s
                a, _ = args.load_dir.split(".")
                a = a.split("_")[-1]
                with open("sh_means_" + str(a) + ".npz", "wb") as f:
                    np.save(f, means)
                with open("sh_stds_" + str(a) + ".npz", "wb") as f:
                    np.save(f, stds)
            elif args.env_name.startswith("Oracle"):
                fs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                      19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
                ls = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
                      0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25,
                      1.30, 1.35, 1.40, 1.45, 1.50, 1.55, 1.60, 1.65, 1.70]
                a, _ = args.load_dir.split(".")
                a = a.split("_")[-1]

                means = np.zeros((len(fs), len(ls)))
                stds = np.zeros((len(fs), len(ls)))
                for f_i in range(len(fs)):
                    for l_i in range(len(ls)):
                        f = fs[f_i]
                        l = ls[l_i]
                        u, s = evaluate(actor_critic, ob_rms,
                                        'OracleCartpoleTest' + str(f) + "_" + str(l)  + '-v0',
                                        args.seed,
                                        args.num_processes, eval_log_dir, device)
                        means[f_i, l_i] = u
                        stds[f_i, l_i] = s
                with open("cp_means" + str(a) + ".npz", "wb") as f:
                    np.save(f, means)
                with open("cp_stds" + str(a) + ".npz", "wb") as f:
                    np.save(f, stds)
            elif args.env_name.startswith("HC"):
                ds = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950,
                      1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700, 1750,
                      1800, 1850, 1900, 1950, 2000]
                us = np.zeros_like(ds)
                ss = np.zeros_like(ds)
                for i in range(len(ds)):
                    d = ds[i]
                    u, s = evaluate(actor_critic, ob_rms, "OracleHalfCheetahTest_" + str(d) + "-v0", args.seed,
                                    args.num_processes, eval_log_dir, device)
                    us[i] = u
                    ss[i] = s
                a, _ = args.load_dir.split(".")
                a = a.split("_")[-1]
                with open("hc_means" + str(a) + ".npz", "wb") as f:
                    np.save(f, us)
                with open("hc_stds" + str(a) + ".npz", "wb") as f:
                    np.save(f, ss)
            assert False, "Evaluation Ended"

if __name__ == "__main__":
    main()
