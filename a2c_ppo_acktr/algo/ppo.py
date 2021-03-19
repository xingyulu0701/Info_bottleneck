import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy

class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 beta=0,
                 beta_end=0,
                 state=False):

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        self.beta_beg = beta
        if beta_end == 0:
            beta_end = beta
        self.beta_end = beta_end
        self.beta_ratio = 1 if self.beta_beg == self.beta_end else self.beta_end / self.beta_beg
        self.ratio = 0
        self.state = state

    def update(self, rollouts):
        self.beta = self.beta_beg * (numpy.exp(numpy.log(self.beta_ratio) * self.ratio))
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        value_mi_epoch = 0

        # print(self.beta)
        for e in range(self.ppo_epoch):
            if self.actor_critic.is_recurrent:
                data_generator = rollouts.recurrent_generator(
                    advantages, self.num_mini_batch)
            else:
                data_generator = rollouts.feed_forward_generator(
                    advantages, self.num_mini_batch, state=self.state)

            for sample in data_generator:
                obs_batch, recurrent_hidden_states_batch, actions_batch, \
                   value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
                        adv_targ, eps = sample

                # Reshape to do in a single forward pass for all steps
                if self.state:
                    values, action_log_probs, dist_entropy, kl, _, dist = self.actor_critic.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch, masks_batch,
                        actions_batch, eps)
                else:
                    values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                        obs_batch, recurrent_hidden_states_batch, masks_batch,
                        actions_batch, eps)

                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()

                if self.state:
                    dkl_loss = kl.mean()
                    (value_loss * self.value_loss_coef + action_loss -
                     dist_entropy * self.entropy_coef + self.beta * dkl_loss).backward()
                else:
                    (value_loss * self.value_loss_coef + action_loss -
                     dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_mi_epoch += dkl_loss.item()
                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()


        num_updates = self.ppo_epoch * self.num_mini_batch


        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        value_mi_epoch /= num_updates


        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, value_mi_epoch

# class PPOInfo():
#     def __init__(self,
#                  actor_critic,
#                  clip_param,
#                  ppo_epoch,
#                  num_mini_batch,
#                  value_loss_coef,
#                  entropy_coef,
#                  lr=None,
#                  eps=None,
#                  max_grad_norm=None,
#                  use_clipped_value_loss=True,
#                  beta=0.0001,
#                  beta_end=0,
#                  seperate_gradient=False,
#                  l2=0,
#                  MINE=False,
#                  target=None):
#
#         self.actor_critic = actor_critic
#         self.beta_end = beta_end or beta
#         self.clip_param = clip_param
#         self.ppo_epoch = ppo_epoch
#         self.num_mini_batch = num_mini_batch
#
#         self.value_loss_coef = value_loss_coef
#         self.entropy_coef = entropy_coef
#
#         self.max_grad_norm = max_grad_norm
#         self.use_clipped_value_loss = use_clipped_value_loss
#
#         self.beta_beg = beta
#         self.ratio = 0
#         self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps, weight_decay=l2)
#         self.beta_ratio = 1 if self.beta_beg == self.beta_end else self.beta_end / self.beta_beg
#         self.ratio = 0
#
#         self.separate_gradient = seperate_gradient
#         self.MINE = MINE
#
#         self.target = target
#
#
#     def update(self, rollouts):
#
#         self.beta = self.beta_beg * (numpy.exp(numpy.log(self.beta_ratio) * self.ratio))
#         # print(self.beta)
#         advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
#         advantages = (advantages - advantages.mean()) / (
#             advantages.std() + 1e-5)
#
#         value_loss_epoch = 0
#         action_loss_epoch = 0
#         dist_entropy_epoch = 0
#         value_mi_epoch = 0
#         value_avg_logits_mean = 0
#         value_avg_logits_logstd = 0
#         value_norm_logits_mean = 0
#         value_norm_logits_logstd = 0
#
#         value_grad_norm_logits_logstd = 0
#
#         value_avg_action_mean = None
#         value_avg_action_logstd = None
#
#         for e in range(self.ppo_epoch):
#             if self.actor_critic.is_recurrent:
#                 data_generator = rollouts.recurrent_generator(
#                     advantages, self.num_mini_batch)
#             else:
#                 data_generator = rollouts.feed_forward_generator(
#                     advantages, self.num_mini_batch, state=True)
#
#             for sample in data_generator:
#                 obs_batch, recurrent_hidden_states_batch, actions_batch, \
#                    value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, \
#                         adv_targ, eps = sample
#                 # Reshape to do in a single forward pass for all steps
#                 values, action_log_probs, dist_entropy, kl, _, dist = self.actor_critic.evaluate_actions(
#                 # values, action_log_probs, dist_entropy, kl, _, mean_norm, logstd_norm, avg_logits_mean, avg_logits_logstd, dist = self.actor_critic.evaluate_actions(
#                     obs_batch, recurrent_hidden_states_batch, masks_batch,
#                     actions_batch, eps)
#
#                 ratio = torch.exp(action_log_probs -
#                                   old_action_log_probs_batch)
#                 surr1 = ratio * adv_targ
#                 surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
#                                     1.0 + self.clip_param) * adv_targ
#                 action_loss = -torch.min(surr1, surr2).mean()
#
#                 dkl_loss = kl.mean()
#
#                 if self.use_clipped_value_loss:
#                     value_pred_clipped = value_preds_batch + \
#                         (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
#                     value_losses = (values - return_batch).pow(2)
#                     value_losses_clipped = (
#                         value_pred_clipped - return_batch).pow(2)
#                     value_loss = 0.5 * torch.max(value_losses,
#                                                  value_losses_clipped).mean()
#                 else:
#                     value_loss = 0.5 * (return_batch - values).pow(2).mean()
#
#                 self.optimizer.zero_grad()
#                 if not self.MINE:
#                     (value_loss * self.value_loss_coef + action_loss -
#                      dist_entropy * self.entropy_coef + self.beta * dkl_loss).backward()
#                 else:
#                     (value_loss * self.value_loss_coef + action_loss -
#                      dist_entropy * self.entropy_coef).backward()
#
#                 nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
#                                          self.max_grad_norm)
#                 self.optimizer.step()
#
#                 value_mi_epoch += dkl_loss.item()
#                 value_loss_epoch += value_loss.item()
#                 action_loss_epoch += action_loss.item()
#                 dist_entropy_epoch += dist_entropy.item()
#
#         num_updates = self.ppo_epoch * self.num_mini_batch
#
#         value_loss_epoch /= num_updates
#         action_loss_epoch /= num_updates
#         dist_entropy_epoch /= num_updates
#         value_mi_epoch /= num_updates
#
#
#         return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, value_mi_epoch, value_avg_logits_mean, \
#                value_avg_logits_logstd, value_norm_logits_mean, value_norm_logits_logstd, value_avg_action_mean, \
#                value_avg_action_logstd, value_grad_norm_logits_logstd
