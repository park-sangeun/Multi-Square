from util.model import HighPolicy, LowPolicy, Critic
from alg.bc import Agent as BC_AGENT
from util.replay_buffer import batch_traj_process, HierarchyDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import torch.nn as nn
import json
import torch
import copy
import pandas as pd
import random
import numpy as np
import os
import yaml
from prompt.inst import high_prompt, low_prompt, single_prompt
from util.extract import extract_action_done
from util.replay_buffer import OnlineDataset

class Multi2:
    def __init__(self, args):
        self.args = args
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')
        self.buffer = HierarchyDataset(args)
        base_log = f"{args['log_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}/Offline"
        os.makedirs(base_log, exist_ok=True)
        self.low_writer = SummaryWriter(log_dir=os.path.join(base_log, 'low'))
        self.critic_writer = SummaryWriter(log_dir=os.path.join(base_log, 'critic'))
        self.critic = Critic(args)
        self.critic.train()
        self.high_policy = HighPolicy(args)
        self.high_policy.train()
        self.low_policy = LowPolicy(args)
        self.low_policy.train()
        if hasattr(self.low_policy, 'base'):
            self.low_policy.base.train()
        self.critic_optim = torch.optim.AdamW([p for p in self.critic.parameters() if p.requires_grad], lr=args.get('critic_lr', 0.0001), betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        self.low_optim = torch.optim.AdamW([p for p in self.low_policy.base.parameters() if p.requires_grad], lr=args.get('actor_lr', 1e-05), betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        
        high_path = "/path/to/your/system1_sft_checkpoint"
        
        BC_AGENT.load_high_policy(self, high_path)
        low_path = "/path/to/your/system2_warmup_checkpoint"
        
        BC_AGENT.load_low_policy(self, low_path)
        self.loss_fct = torch.nn.MSELoss()
        self.low_global_step = 0
        self.critic_global_step = 0
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        base_ckpt = f"{args['check_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}"
        self.low_checkpoint_dir = os.path.join(base_ckpt, 'low', timestamp)
        self.critic_checkpoint_dir = os.path.join(base_ckpt, 'critic', timestamp)
        os.makedirs(self.low_checkpoint_dir, exist_ok=True)
        os.makedirs(self.critic_checkpoint_dir, exist_ok=True)
        dev_low = self._input_device(self.low_policy)
        base_dtype = next(self.low_policy.base.parameters()).dtype
        try:
            self.low_policy.base.gradient_checkpointing_enable()
        except Exception:
            pass
        self.log_file = os.path.join(f"{self.args['log_path']}/{self.args['benchmark']}/{self.args['alg_name']}/{self.args['model_name']}", 'validation_seen.log')
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.fixed_dev_seen_subset = None
        val_seed = int(self.args.get('val_seed', 777))
        k_per_task = int(self.args.get('val_k_per_task', 10))
        with open('alg/base_config.yaml') as reader:
            self.config = yaml.safe_load(reader)
    def _append_result_log(self, split, task_id, task_name, variation_id, label, score):
        with open(self.log_file, 'a') as f:
            f.write(f'{split}\t{task_id}\t{task_name}\t{variation_id}\t{label}\t{score}\n')
    def _episode_cleanup(*objs):
        import gc, torch
        for o in objs:
            try:
                del o
            except:
                pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        torch.cuda.synchronize()
    def _get_model(self, obj):
        return getattr(obj, 'base', obj)
    def _device_of(self, module: torch.nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def _micro_bs(self) -> int:
        return int(self.args.get('train_micro_batch_size_per_gpu', self.args.get('batch_size', 1)))
    def _input_device(self, policy) -> torch.device:
        m = self._get_model(policy)
        if hasattr(m, 'get_input_embeddings'):
            return m.get_input_embeddings().weight.device
        try:
            return next(m.parameters()).device
        except StopIteration:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def update_policy(self, batch_data, level='low'):
        micro_batch_size = 8
        total_size = len(batch_data['obs'])
        def expectile_loss(diff: torch.Tensor, expectile: float=0.7):
            weight = torch.where(diff > 0, expectile, 1 - expectile)
            return (weight * diff ** 2).mean()
        if level == 'SFT':
            policy = self.low_policy
            optimizer = self.low_optim
            bc_loss_total = 0.0
            optimizer.zero_grad(set_to_none=True)
            seen = 0
            for start in range(0, total_size, micro_batch_size):
                end = start + micro_batch_size
                batch_slice = {k: v[start:end] for k, v in batch_data.items()}
                tokens = batch_traj_process(batch_slice['subtask'], batch_slice['obs'], batch_slice['action'], policy.tokenizer)
                if torch.cuda.is_available():
                    tokens = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in tokens.items()}
                action_log_probs, action_masks = policy.get_log_prob(tokens)
                max_actions = int(tokens['action_end_mask'].sum(dim=1).max().item())
                valid_action_log_probs = self._extract_valid_action_probs(action_log_probs, action_masks, max_actions)
                bc_loss = -valid_action_log_probs.mean()
                bc_loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                bs = len(batch_slice['obs'])
                bc_loss_total += bc_loss.item() * bs
                seen += bs
            return bc_loss_total / max(1, seen)
        elif level == 'low':
            actor = self.low_policy
            actor_optim = self.low_optim
            critic = self.critic
            critic_optim = self.critic_optim
            q_loss_total = v_loss_total = actor_loss_total = 0.0
            gamma = float(self.args.get('gamma', 0.99))
            actor_optim.zero_grad(set_to_none=True)
            critic_optim.zero_grad(set_to_none=True)
            dev = next(critic.parameters()).device
            critic.value_head.to(dev)
            critic.q_head.to(dev)
            critic.target_value_head.to(dev)
            critic.target_q_head.to(dev)
            seen = 0
            diag_accum = {'adv_mean': 0.0, 'adv_std': 0.0, 'adv_pos_ratio': 0.0, 'lin_mean': 0.0, 'expw_mean': 0.0, 'w_total_mean': 0.0, 'w_total_std': 0.0, 'clip_ratio': 0.0, 'norm_factor': 1.0, 'logp_mean': 0.0}
            diag_count = 0
            for start in range(0, total_size, micro_batch_size):
                end = start + micro_batch_size
                batch_slice = {k: v[start:end] for k, v in batch_data.items()}
                tokens = batch_traj_process(batch_slice['subtask'], batch_slice['obs'], batch_slice['action'], actor.tokenizer)
                if torch.cuda.is_available():
                    tokens = {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in tokens.items()}
                rewards, dones = self.prepare_tensor(batch_slice['reward'], batch_slice['done'])
                rewards = rewards.to(dev)
                dones = dones.to(dev)
                hidden_states, state_end_mask, action_end_mask = actor.get_hidden_states(tokens)
                head_param = next(self.critic.target_value_head.parameters())
                h = hidden_states.to(device=head_param.device, dtype=head_param.dtype)
                with torch.no_grad():
                    target_vs = critic.target_value_head(h).squeeze(-1)
                    target_qsa = critic.target_q_head(h).squeeze(-1)
                    target_vs, _ = self.extract_valid(target_vs, action_end_mask)
                    target_qsa, _ = self.extract_valid(target_qsa, action_end_mask)
                vs, q_sa = critic.forward_hidden(h)
                vs, _ = self.extract_valid(vs, action_end_mask)
                q_sa, _ = self.extract_valid(q_sa, action_end_mask)
                target = rewards[:, :-1] + (1.0 - dones[:, :-1]) * target_vs[:, 1:] * gamma
                q_loss = self.loss_fct(q_sa[:, :-1], target)
                diff = target_qsa[:, :-1] - vs[:, :-1]
                tau = float(self.args.get('weight_tau', 0.7))
                weight = torch.where(diff < 0, 1 - tau, tau)
                v_loss = (weight * diff ** 2).mean()
                critic_loss = q_loss + v_loss
                critic_loss.backward()
                critic_optim.step()
                critic_optim.zero_grad(set_to_none=True)
                critic.soft_update_target_critic(tau=float(self.args.get('tau', 0.01)))
                use_policy_action = bool(self.args.get('use_policy_action', True))
                beta = float(self.args.get('beta', 5.0))
                w_clip = float(self.args.get('awbc_w_clip', 100.0))
                lam = float(self.args.get('lambda_adv', 0.1))
                relu_w = bool(self.args.get('relu_adv_weight', True))
                norm_w = bool(self.args.get('normalize_w', True))
                def tmean(x):
                    return float(x.mean().item())
                def tstd(x):
                    return float(x.std(unbiased=False).item())
                if use_policy_action and hasattr(actor, 'generate_actions'):
                    with torch.no_grad():
                        tokens_pi, action_masks_pi = actor.generate_actions(tokens, max_action_len=int(self.args.get('max_action_len', 32)), strategy=self.args.get('policy_action_strategy', 'greedy'), temperature=float(self.args.get('temp', 1.0)))
                    hidden_states_pi, state_end_mask_pi, action_end_mask_pi = actor.get_hidden_states(tokens_pi)
                    head_param = next(self.critic.q_head.parameters())
                    h_pi = hidden_states_pi.to(device=head_param.device, dtype=head_param.dtype)
                    vs_pi, q_sa_pi = critic.forward_hidden(h_pi)
                    vs_pi, _ = self.extract_valid(vs_pi, action_end_mask_pi)
                    q_sa_pi, _ = self.extract_valid(q_sa_pi, action_end_mask_pi)
                    adv_pi = q_sa_pi[:, :-1] - vs_pi[:, :-1]
                    action_log_probs_pi, action_masks_pi2 = actor.get_log_prob(tokens_pi)
                    valid_logp_pi = self._extract_valid_action_probs(action_log_probs_pi, action_masks_pi2, q_sa_pi.size(1))
                    valid_logp_pi = valid_logp_pi[:, :-1]
                    with torch.no_grad():
                        lin = (torch.relu(adv_pi) if relu_w else adv_pi) * lam
                        raw_expw = torch.exp(beta * adv_pi)
                        clip_mask = raw_expw > w_clip
                        expw = raw_expw.clamp(max=w_clip)
                        w_total = lin + expw
                        norm_factor = 1.0
                        if norm_w:
                            norm_factor = float(w_total.mean().clamp_min(1e-06).item())
                            w_total = w_total / norm_factor
                        w_total = torch.nan_to_num(w_total, nan=0.0, posinf=w_clip, neginf=0.0)
                        diag_accum['adv_mean'] += tmean(adv_pi)
                        diag_accum['adv_std'] += tstd(adv_pi)
                        diag_accum['adv_pos_ratio'] += float((adv_pi > 0).float().mean().item())
                        diag_accum['lin_mean'] += tmean(lin)
                        diag_accum['expw_mean'] += tmean(expw)
                        diag_accum['w_total_mean'] += tmean(w_total)
                        diag_accum['w_total_std'] += tstd(w_total)
                        diag_accum['clip_ratio'] += float(clip_mask.float().mean().item())
                        diag_accum['norm_factor'] += norm_factor
                        diag_accum['logp_mean'] += tmean(valid_logp_pi)
                        diag_count += 1
                    actor_loss = -(w_total.detach() * valid_logp_pi).mean()
                else:
                    action_log_probs, action_masks = actor.get_log_prob(tokens)
                    max_actions = q_sa.size(1)
                    valid_action_log_probs = self._extract_valid_action_probs(action_log_probs, action_masks, max_actions)
                    valid_action_log_probs = valid_action_log_probs[:, :-1]
                    with torch.no_grad():
                        adv = q_sa[:, :-1] - vs[:, :-1]
                        raw_expw = torch.exp(beta * adv)
                        clip_mask = raw_expw > w_clip
                        w = raw_expw.clamp(max=w_clip)
                        diag_accum['adv_mean'] += tmean(adv)
                        diag_accum['adv_std'] += tstd(adv)
                        diag_accum['adv_pos_ratio'] += float((adv > 0).float().mean().item())
                        diag_accum['lin_mean'] += 0.0
                        diag_accum['expw_mean'] += tmean(w)
                        diag_accum['w_total_mean'] += tmean(w)
                        diag_accum['w_total_std'] += tstd(w)
                        diag_accum['clip_ratio'] += float(clip_mask.float().mean().item())
                        diag_accum['norm_factor'] += 1.0
                        diag_accum['logp_mean'] += tmean(valid_action_log_probs)
                        diag_count += 1
                    actor_loss = -(w.detach() * valid_action_log_probs).mean()
                actor_loss.backward()
                actor_optim.step()
                actor_optim.zero_grad(set_to_none=True)
                bs = len(batch_slice['obs'])
                seen += bs
                q_loss_total += q_loss.item() * bs
                v_loss_total += v_loss.item() * bs
                actor_loss_total += actor_loss.item() * bs
            if diag_count > 0:
                avg_diag = {k: v / diag_count for k, v in diag_accum.items()}
            else:
                avg_diag = {k: 0.0 for k in diag_accum}
            return (q_loss_total / max(1, seen), v_loss_total / max(1, seen), actor_loss_total / max(1, seen), avg_diag)
        else:
            raise ValueError(f'Unknown level: {level}')
    def update(self):
        micro_bs = int(self.args.get('train_micro_batch_size_per_gpu', 8))
        dataloader = DataLoader(self.buffer, batch_size=micro_bs, shuffle=True, collate_fn=HierarchyDataset.collate_fn)
        for epoch in range(self.args['epochs']):
            print(f'Epoch: {epoch}')
            for step, batch in enumerate(dataloader, start=self.low_global_step):
                q_loss, v_loss, actor_loss, diag = self.update_policy(batch['low'], level='low')
                self.critic.soft_update_target_critic(tau=self.args['tau'])
                actor_val = float(actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss)
                q_val = float(q_loss.item() if isinstance(q_loss, torch.Tensor) else q_loss)
                v_val = float(v_loss.item() if isinstance(v_loss, torch.Tensor) else v_loss)
                self.low_writer.add_scalar('Loss/low/actor_loss', actor_val, step)
                self.low_writer.add_scalar('Loss/low/critic_loss', q_val, step)
                if step % self.args.get('log_freq', 10) == 0:
                    print(f'Low; step:{step}; actor_loss:{actor_val}; critic_loss:{q_val}; value_loss:{v_val}')
                if step % self.args['eval_freq'] == 0:
                    avg_score = self.evaluate_online(num_episodes=self.args.get('val_max_episodes', 10), dev_or_test='dev')
                    tmp_loader = DataLoader(self.buffer, batch_size=1, shuffle=True, collate_fn=HierarchyDataset.collate_fn)
                    try:
                        val_batch = next(iter(tmp_loader))
                        val_losses = self.compute_val_losses(val_batch)
                      
                    except StopIteration:
                        pass
                if step > 0 and step % self.args['eval_freq'] == 0:
                    self._save_step_policy(self.low_policy, self.low_checkpoint_dir, step)
                    self.save_critic(step, self.critic_checkpoint_dir)
                self.low_global_step = step + 1
            torch.cuda.empty_cache()
        last_step = self.low_global_step
        self._save_step_policy(self.low_policy, self.low_checkpoint_dir, last_step)
        self.save_critic(last_step, self.critic_checkpoint_dir)

    @torch.no_grad()
    def compute_val_losses(self, batch):
        out = {}
        tokens_high = batch_traj_process(batch['high']['task_description'], batch['high']['obs'], batch['high']['subtask'], self.high_policy.tokenizer)
        action_log_probs_h, action_masks_h = self.high_policy.get_log_prob(tokens_high)
        max_actions_h = int(tokens_high['action_end_mask'].sum(dim=1).max().item())
        valid_logp_h = self._extract_valid_action_probs(action_log_probs_h, action_masks_h, max_actions_h)
        out['high_BC_loss_val'] = float((-valid_logp_h.mean()).item())
        tokens_low = batch_traj_process(batch['low']['subtask'], batch['low']['obs'], batch['low']['action'], self.low_policy.tokenizer)
        rewards, dones = self.prepare_tensor(batch['low']['reward'], batch['low']['done'])
        dtype = next(self.critic.parameters()).dtype
        rewards, dones = (rewards.to(dtype), dones.to(dtype))
        hidden_states, state_end_mask, action_end_mask = self.low_policy.get_hidden_states(tokens_low)
        dev = hidden_states.device
        self.critic.value_head.to(dev)
        self.critic.q_head.to(dev)
        self.critic.target_value_head.to(dev)
        self.critic.target_q_head.to(dev)
        h32 = hidden_states.float()
        target_vs = self.critic.target_value_head(h32).squeeze(-1)
        target_qsa = self.critic.target_q_head(h32).squeeze(-1)
        target_vs, _ = self.extract_valid(target_vs, action_end_mask)
        target_qsa, _ = self.extract_valid(target_qsa, action_end_mask)
        vs, q_sa = self.critic.forward_hidden(hidden_states)
        vs, _ = self.extract_valid(vs, action_end_mask)
        q_sa, _ = self.extract_valid(q_sa, action_end_mask)
        gamma = float(self.args['gamma'])
        target = rewards[:, :-1] + (1.0 - dones[:, :-1]) * target_vs[:, 1:] * gamma
        q_loss = self.loss_fct(q_sa[:, :-1], target)
        diff = vs - target_qsa
        tau = float(self.args.get('weight_tau', 0.7))
        weight = torch.where(diff < 0, torch.ones_like(vs) * (1 - tau), torch.ones_like(vs) * tau)
        v_loss = (weight * diff ** 2).mean()
        use_policy_action = bool(self.args.get('use_policy_action', True))
        beta = float(self.args.get('beta', 5.0))
        w_clip = float(self.args.get('awbc_w_clip', 100.0))
        lam = float(self.args.get('lambda_adv', 0.1))
        relu_w = bool(self.args.get('relu_adv_weight', True))
        norm_w = bool(self.args.get('normalize_w', True))
        if use_policy_action and hasattr(self.low_policy, 'generate_actions'):
            tokens_pi, action_masks_pi = self.low_policy.generate_actions(tokens_low, max_action_len=int(self.args.get('max_action_len', 32)), strategy=self.args.get('policy_action_strategy', 'greedy'), temperature=float(self.args.get('temp', 1.0)))
            hidden_states_pi, state_end_mask_pi, action_end_mask_pi = self.low_policy.get_hidden_states(tokens_pi)
            head_param = next(self.critic.q_head.parameters())
            h_pi = hidden_states_pi.to(device=head_param.device, dtype=head_param.dtype)
            vs_pi, q_sa_pi = self.critic.forward_hidden(h_pi)
            vs_pi, _ = self.extract_valid(vs_pi, action_end_mask_pi)
            q_sa_pi, _ = self.extract_valid(q_sa_pi, action_end_mask_pi)
            adv_pi = q_sa_pi[:, :-1] - vs_pi[:, :-1]
            action_log_probs_pi, action_masks_pi2 = self.low_policy.get_log_prob(tokens_pi)
            max_actions_pi = q_sa_pi.size(1)
            valid_logp_pi = self._extract_valid_action_probs(action_log_probs_pi, action_masks_pi2, max_actions_pi).to(q_sa_pi.dtype)
            valid_logp_pi = valid_logp_pi[:, :-1]
            lin = (torch.relu(adv_pi) if relu_w else adv_pi) * lam
            expw = torch.exp(beta * adv_pi).clamp(max=w_clip)
            w_total = lin + expw
            if norm_w:
                w_total = w_total / w_total.mean().clamp_min(1e-06)
            w_total = torch.nan_to_num(w_total, nan=0.0, posinf=w_clip, neginf=0.0)
            actor_loss = -(w_total * valid_logp_pi).mean()
        else:
            action_log_probs_l, action_masks_l = self.low_policy.get_log_prob(tokens_low)
            max_actions_l = q_sa.size(1)
            valid_logp_l = self._extract_valid_action_probs(action_log_probs_l, action_masks_l, max_actions_l).to(q_sa.dtype)
            valid_logp_l = valid_logp_l[:, :-1]
            if self.args.get('use_adv', False):
                adv = q_sa[:, :-1] - vs[:, :-1]
                actor_loss = -(adv * valid_logp_l).mean()
            else:
                actor_loss = -(q_sa[:, :-1] * valid_logp_l).mean()
        out['low_q_loss_val'] = float(q_loss.item())
        out['low_v_loss_val'] = float(v_loss.item())
        out['low_actor_loss_val'] = float(actor_loss.item())
        return out
    def evaluate_online(self, num_episodes=10, dev_or_test='dev'):
        split = 'eval_in_distribution'
        def get_environment(env_type):
            if env_type == 'AlfredTWEnv':
                from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
                return AlfredTWEnv
            elif env_type == 'AlfredThorEnv':
                from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv
                return AlfredThorEnv
            elif env_type == 'AlfredHybrid':
                from alfworld.agents.environment.alfred_hybrid import AlfredHybrid
                return AlfredHybrid
            else:
                raise NotImplementedError(f'Environment {env_type} is not implemented.')
        env_cls = get_environment(self.config['env']['type'])
        self.eval_env = env_cls(self.config, train_eval=split)
        self.eval_env = self.eval_env.init_env(batch_size=1)
        total_scores = []
        failure = 0
        total_task = 0
        for ep in range(num_episodes):
            print(f'EP: {ep}')
            obs, info = self.eval_env.reset()
            score = self.eval_policy()
            if score == 0:
                failure += 1
                total_task += 1
            else:
                total_scores.append(score)
                total_task += 1
            print(f'[Episode {ep + 1}], Score: {score}')
        if total_scores:
            avg_score = sum(total_scores) / len(total_scores)
            mean = np.mean(total_scores)
            std = np.std(total_scores)
            print(f'\n=== Final Result over {num_episodes} episodes: {avg_score:.3f} ===')
            print(f'\nFailure: {failure} per Total: {total_task}')
            print(f'{total_scores} \n Mean: {mean} +- {std}')
        else:
            avg_score = 0
            print('No valid tasks/variations evaluated.')
        return avg_score
    def eval_policy(self):
        episode_steps = 0
        obs, info = self.eval_env.reset()
        obs_text = obs[0] if isinstance(obs, (list, tuple)) else obs
        task_description = None
        for line in obs_text.split('\n'):
            if 'Your task is to:' in line:
                task_description = line.replace('Your task is to:', 'Your task is to').strip()
                break
        if task_description is None:
            task_description = 'Your task is unknown.'
        lines = obs_text.split('\n')
        lines = [l.strip() for l in lines if 'Welcome to TextWorld' not in l and l.strip() != '']
        lines = [l for l in lines if not l.lower().startswith('your task is to')]
        obs = ' '.join(lines).strip()
        initial_room_obs = obs
        high_traj_token = self.high_policy.tokenizer(high_prompt + ' Task Description:\n' + task_description, return_tensors='pt')
        traj_subtask, traj_group_action = ([], [])
        group_action = []
        episode_done = False
        total_reward = 0.0
        with torch.inference_mode():
            while not episode_done:
                state = f'Group action: {group_action}. Current observation: {obs}'
      
                state_token = self.high_policy.tokenizer(state, return_tensors='pt')
                high_traj_token['input_ids'] = torch.cat([high_traj_token['input_ids'], state_token['input_ids']], dim=1)
                high_traj_token['attention_mask'] = torch.cat([high_traj_token['attention_mask'], state_token['attention_mask']], dim=1)
                subtask = self.high_policy.generate_action(high_traj_token)[0]
                subtask_token = self.high_policy.tokenizer(subtask + self.high_policy.tokenizer.eos_token, return_tensors='pt')

                traj_subtask.append(subtask)
                high_traj_token['input_ids'] = torch.cat([high_traj_token['input_ids'], subtask_token['input_ids']], dim=1)
                high_traj_token['attention_mask'] = torch.cat([high_traj_token['attention_mask'], subtask_token['attention_mask']], dim=1)
                low_group_token = self.low_policy.tokenizer(low_prompt + ' Subtask: ' + subtask, return_tensors='pt')
                subtask_done = False
                group_action = []
                raw_action_list = []
                low_iter = 0
                is_first_low_step = True
                while not subtask_done:
                    episode_steps += 1
                    low_iter += 1
                    if initial_room_obs:
                        room_phrase = 'you are in the middle of a room'
                        if is_first_low_step and room_phrase not in obs.lower():
                            obs_for_model = f'{initial_room_obs} {obs}'
                        else:
                            obs_for_model = obs
                    else:
                        obs_for_model = obs
         
                    obs_token = self.low_policy.tokenizer('Obs: ' + obs_for_model, return_tensors='pt')
                    is_first_low_step = False
                    low_group_token['input_ids'] = torch.cat([low_group_token['input_ids'], obs_token['input_ids']], dim=1)
                    low_group_token['attention_mask'] = torch.cat([low_group_token['attention_mask'], obs_token['attention_mask']], dim=1)
                    raw_action = self.low_policy.generate_action(low_group_token)[0]
                    raw_action_list.append(raw_action)
                    action, subtask_done = extract_action_done(raw_action)
         
                    group_action.append(action)
                    if isinstance(action, str):
                        action = [action]
                    elif action is None:
                        action = ['look around']
                    action_token = self.low_policy.tokenizer(raw_action + self.low_policy.tokenizer.eos_token, return_tensors='pt')
                    low_group_token['input_ids'] = torch.cat([low_group_token['input_ids'], action_token['input_ids']], dim=1)
                    low_group_token['attention_mask'] = torch.cat([low_group_token['attention_mask'], action_token['attention_mask']], dim=1)
                    obs_, reward, step_done, info = self.eval_env.step(action)
                    obs_text = obs_[0] if isinstance(obs_, (list, tuple)) else obs_
                    obs = obs_text

                    if episode_steps == self.args['env_step_limit']:
                        episode_done = True
                        break
                    if low_iter >= self.args['env_step_limit']:
               
                        episode_done = True
                        break
                    if True in step_done if isinstance(step_done, (list, tuple)) else step_done:
                        episode_done = True
                        break
                traj_group_action.append(group_action)
        won_value = info.get('won', False)
        if isinstance(won_value, (list, tuple)):
            won_value = won_value[0] if len(won_value) > 0 else False
        final_score = 1.0 if won_value else 0.0
        self._episode_cleanup(high_traj_token, low_group_token, state_token if 'state_token' in locals() else None)
        return final_score
    def extract_valid(self, value, valid_mark):
        batch_size = value.size(0)
        max_valid_len = valid_mark.sum(dim=1).max().item()
        valid_value = torch.zeros(batch_size, max_valid_len, device=value.device)
        mask = torch.zeros(batch_size, max_valid_len, device=value.device)
        for i in range(batch_size):
            valid_idx = torch.where(valid_mark[i] == 1)[0]
            valid_len = valid_idx.size(0)
            valid_value[i, :valid_len] = value[i][valid_idx]
            mask[i, :valid_len] = 1
        return (valid_value, mask)
    @staticmethod
    def _extract_valid_action_probs(log_probs, masks, max_action_nums: int):
        B = log_probs.size(0)
        device = log_probs.device
        out = torch.zeros(B, max_action_nums, device=device)
        for i in range(B):
            pos = torch.where(masks[i] == 1)[0]
            groups = []
            cur = []
            for p in pos:
                if not cur or p == cur[-1] + 1:
                    cur.append(p)
                else:
                    groups.append(cur)
                    cur = [p]
            if cur:
                groups.append(cur)
            for j, g in enumerate(groups):
                out[i, j] = log_probs[i, g].mean()
        return out
    @staticmethod
    def _save_policy(policy, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        policy.base.save_pretrained(save_dir)
        policy.tokenizer.save_pretrained(save_dir)
    def prepare_tensor(self, rewards, dones):
        device = self._input_device(self.low_policy)
        reward_list = [torch.as_tensor(seq, dtype=torch.float32, device=device) for seq in rewards]
        done_list = [torch.as_tensor(seq, dtype=torch.float32, device=device) for seq in dones]
        reward_tensor = pad_sequence(reward_list, batch_first=True, padding_value=0.0)
        done_tensor = pad_sequence(done_list, batch_first=True, padding_value=0.0)
        return (reward_tensor, done_tensor)
    @staticmethod
    def _save_step_policy(policy, save_dir, step_val=None):
        if step_val is not None:
            try:
                step_str = str(int(step_val))
            except Exception:
                step_str = str(step_val)
            out_dir = os.path.join(save_dir, step_str)
        else:
            out_dir = save_dir
        os.makedirs(out_dir, exist_ok=True)
        policy.base.save_pretrained(out_dir)
        policy.tokenizer.save_pretrained(out_dir)
        return out_dir
    def save_critic(self, step, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(self.critic.state_dict(), f'{checkpoint_dir}/critic.pt')