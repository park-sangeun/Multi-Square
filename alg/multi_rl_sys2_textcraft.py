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

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

        self.buffer = HierarchyDataset(args)
        base_log = f"{args['log_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}/Offline"
        os.makedirs(base_log, exist_ok=True)

        self.low_writer    = SummaryWriter(log_dir=os.path.join(base_log, "low"))
        self.critic_writer = SummaryWriter(log_dir=os.path.join(base_log, "critic"))

        self.critic      = Critic(args);      self.critic.train()
        self.low_policy  = LowPolicy(args);   self.low_policy.train()
        if hasattr(self.low_policy, "base"):
            self.low_policy.base.train()
        self.critic_optim = torch.optim.AdamW(
            [p for p in self.critic.parameters() if p.requires_grad],
            lr=args.get("critic_lr", 1e-4), betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
        )

        self.low_optim = torch.optim.AdamW(
            [p for p in self.low_policy.base.parameters() if p.requires_grad],
            lr=args.get("actor_lr", 1e-5), betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
        )
      
        low_path = "/path/to/your/system2_warmup_checkpoint"
        BC_AGENT.load_low_policy(self, low_path)

        self.loss_fct = torch.nn.MSELoss()

        self.low_global_step  = 0
        self.critic_global_step = 0

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

        base_ckpt = f"{args['check_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}"
        self.low_checkpoint_dir    = os.path.join(base_ckpt, "low",    timestamp)
        self.critic_checkpoint_dir = os.path.join(base_ckpt, "critic", timestamp)
        os.makedirs(self.low_checkpoint_dir, exist_ok=True)
        os.makedirs(self.critic_checkpoint_dir, exist_ok=True)

        dev_low = self._input_device(self.low_policy) 

        base_dtype = next(self.low_policy.base.parameters()).dtype

        try:
            self.low_policy.base.gradient_checkpointing_enable()
        except Exception:
            pass
        
        
        self.log_file = os.path.join(
            f"{self.args['log_path']}/{self.args['benchmark']}/{self.args['alg_name']}/{self.args['model_name']}",
            "validation_seen.log"
        )
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

        self.fixed_dev_seen_subset = None 
        val_seed = int(self.args.get("val_seed", 777))
        k_per_task = int(self.args.get("val_k_per_task", 10))

        with open('alg/base_config.yaml') as reader:
            self.config = yaml.safe_load(reader)
       

    def _append_result_log(self, split, task_id, task_name, variation_id, label, score):
        with open(self.log_file, "a") as f:
            f.write(f"{split}\t{task_id}\t{task_name}\t{variation_id}\t{label}\t{score}\n")

    def _episode_cleanup(*objs):
        import gc, torch
        for o in objs:
            try: del o
            except: pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def _get_model(self, obj):
        return getattr(obj, "base", obj)

    def _device_of(self, module: torch.nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _micro_bs(self) -> int:
        return int(self.args.get("train_micro_batch_size_per_gpu",
                                self.args.get("batch_size", 1)))

    def _input_device(self, policy) -> torch.device:
        m = self._get_model(policy) 
        if hasattr(m, "get_input_embeddings"):
            return m.get_input_embeddings().weight.device
        try:
            return next(m.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    def update_policy(self, batch_data, level="low"):
        micro_batch_size = 8 
        total_size = len(batch_data['obs'])
        def expectile_loss(diff: torch.Tensor, expectile: float = 0.7):
            weight = torch.where(diff > 0, expectile, 1 - expectile)
            return (weight * (diff ** 2)).mean()
        
        if level == "SFT":
            policy = self.low_policy
            optimizer = self.low_optim
            bc_loss_total = 0.0

            optimizer.zero_grad(set_to_none=True)
            seen = 0
            for start in range(0, total_size, micro_batch_size):
                end = start + micro_batch_size
                batch_slice = {k: v[start:end] for k, v in batch_data.items()}

                tokens = batch_traj_process(
                    batch_slice["subtask"],
                    batch_slice["obs"],
                    batch_slice["action"],
                    policy.tokenizer
                )
                if torch.cuda.is_available():
                    tokens = {
                        k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
                        for k, v in tokens.items()
                    }
                action_log_probs, action_masks = policy.get_log_prob(tokens)
                max_actions = int(tokens['action_end_mask'].sum(dim=1).max().item())
                valid_action_log_probs = self._extract_valid_action_probs(
                    action_log_probs, action_masks, max_actions
                )
                bc_loss = -valid_action_log_probs.mean()

                bc_loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                bs = len(batch_slice['obs'])
                bc_loss_total += bc_loss.item() * bs
                seen += bs

            return bc_loss_total / max(1, seen)


        elif level == "low":
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

            diag_accum = {
                'adv_mean': 0.0, 'adv_std': 0.0, 'adv_pos_ratio': 0.0,
                'lin_mean': 0.0, 'expw_mean': 0.0,
                'w_total_mean': 0.0, 'w_total_std': 0.0,
                'clip_ratio': 0.0, 'norm_factor': 1.0,
                'logp_mean': 0.0
            }
            diag_count = 0

            for start in range(0, total_size, micro_batch_size):
                end = start + micro_batch_size
                batch_slice = {k: v[start:end] for k, v in batch_data.items()}

                tokens = batch_traj_process(
                    batch_slice['subtask'],
                    batch_slice['obs'],
                    batch_slice['action'],
                    actor.tokenizer
                )
                if torch.cuda.is_available():
                    tokens = {
                        k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
                        for k, v in tokens.items()
                    }
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
                weight = torch.where(diff < 0, (1 - tau), tau)
                v_loss = (weight * (diff ** 2)).mean()

                critic_loss = q_loss + v_loss
                critic_loss.backward()
                critic_optim.step()
                critic_optim.zero_grad(set_to_none=True)

                critic.soft_update_target_critic(tau=float(self.args.get('tau', 0.01)))
                use_policy_action = bool(self.args.get('use_policy_action', True)) 
                beta   = float(self.args.get('beta', 5.0))
                w_clip = float(self.args.get('awbc_w_clip', 100.0))
                lam    = float(self.args.get('lambda_adv', 0.1))
                relu_w = bool(self.args.get('relu_adv_weight', True))
                norm_w = bool(self.args.get('normalize_w', True))

                def tmean(x): return float(x.mean().item())
                def tstd(x):  return float(x.std(unbiased=False).item())

                if use_policy_action and hasattr(actor, 'generate_actions'):
       
                    with torch.no_grad():
                        tokens_pi, action_masks_pi = actor.generate_actions(
                            tokens,
                            max_action_len=int(self.args.get('max_action_len', 32)),
                            strategy=self.args.get('policy_action_strategy', 'greedy'),
                            temperature=float(self.args.get('temp', 1.0))
                        )

                    hidden_states_pi, state_end_mask_pi, action_end_mask_pi = actor.get_hidden_states(tokens_pi)
                    head_param = next(self.critic.q_head.parameters())
                    h_pi = hidden_states_pi.to(device=head_param.device, dtype=head_param.dtype)
                    vs_pi, q_sa_pi = critic.forward_hidden(h_pi)    
                    vs_pi, _   = self.extract_valid(vs_pi,   action_end_mask_pi)
                    q_sa_pi, _ = self.extract_valid(q_sa_pi, action_end_mask_pi)
                    adv_pi = q_sa_pi[:, :-1] - vs_pi[:, :-1]     
                    action_log_probs_pi, action_masks_pi2 = actor.get_log_prob(tokens_pi)
                    valid_logp_pi = self._extract_valid_action_probs(
                        action_log_probs_pi, action_masks_pi2, q_sa_pi.size(1)
                    )
                    valid_logp_pi = valid_logp_pi[:, :-1]
                    with torch.no_grad():
                        lin  = (torch.relu(adv_pi) if relu_w else adv_pi) * lam
                        raw_expw = torch.exp(beta * adv_pi)
                        clip_mask = raw_expw > w_clip
                        expw = raw_expw.clamp(max=w_clip)
                        w_total = lin + expw
                        norm_factor = 1.0
                        if norm_w:
                            norm_factor = float(w_total.mean().clamp_min(1e-6).item())
                            w_total = w_total / norm_factor
                        w_total = torch.nan_to_num(w_total, nan=0.0, posinf=w_clip, neginf=0.0)

                        diag_accum['adv_mean']      += tmean(adv_pi)
                        diag_accum['adv_std']       += tstd(adv_pi)
                        diag_accum['adv_pos_ratio'] += float((adv_pi > 0).float().mean().item())
                        diag_accum['lin_mean']      += tmean(lin)
                        diag_accum['expw_mean']     += tmean(expw)
                        diag_accum['w_total_mean']  += tmean(w_total)
                        diag_accum['w_total_std']   += tstd(w_total)
                        diag_accum['clip_ratio']    += float(clip_mask.float().mean().item())
                        diag_accum['norm_factor']   += norm_factor
                        diag_accum['logp_mean']     += tmean(valid_logp_pi)

                        diag_count += 1

                    actor_loss = -(w_total.detach() * valid_logp_pi).mean()

                else:
                    action_log_probs, action_masks = actor.get_log_prob(tokens)
                    max_actions = q_sa.size(1)
                    valid_action_log_probs = self._extract_valid_action_probs(
                        action_log_probs, action_masks, max_actions
                    ) 
                    valid_action_log_probs = valid_action_log_probs[:, :-1]

                    with torch.no_grad():
                        adv = q_sa[:, :-1] - vs[:, :-1]
                        raw_expw = torch.exp(beta * adv)
                        clip_mask = raw_expw > w_clip
                        w = raw_expw.clamp(max=w_clip)
                        diag_accum['adv_mean']      += tmean(adv)
                        diag_accum['adv_std']       += tstd(adv)
                        diag_accum['adv_pos_ratio'] += float((adv > 0).float().mean().item())
                        diag_accum['lin_mean']      += 0.0
                        diag_accum['expw_mean']     += tmean(w)
                        diag_accum['w_total_mean']  += tmean(w)
                        diag_accum['w_total_std']   += tstd(w)
                        diag_accum['clip_ratio']    += float(clip_mask.float().mean().item())
                        diag_accum['norm_factor']   += 1.0
                        diag_accum['logp_mean']     += tmean(valid_action_log_probs)

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
                avg_diag = {k: (v / diag_count) for k, v in diag_accum.items()}
            else:
                avg_diag = {k: 0.0 for k in diag_accum}

            return (
                q_loss_total / max(1, seen),
                v_loss_total / max(1, seen),
                actor_loss_total / max(1, seen),
                avg_diag,
            )

        else:
            raise ValueError(f"Unknown level: {level}")
            
         

    def update(self):
        micro_bs  = int(self.args.get("train_micro_batch_size_per_gpu", 8))
        
        dataloader = DataLoader(
            self.buffer,
            batch_size=micro_bs,
            shuffle=True,
            collate_fn=HierarchyDataset.collate_fn
        )
       
        for epoch in range(self.args['epochs']):
            print(f"Epoch: {epoch}")
            for step, batch in enumerate(dataloader, start=self.low_global_step):
                q_loss, v_loss, actor_loss, diag = self.update_policy(batch['low'], level="low")
                self.critic.soft_update_target_critic(tau=self.args['tau'])
                actor_val = float(actor_loss.item() if isinstance(actor_loss, torch.Tensor) else actor_loss)
                q_val     = float(q_loss.item()    if isinstance(q_loss,    torch.Tensor) else q_loss)
                v_val     = float(v_loss.item()    if isinstance(v_loss,    torch.Tensor) else v_loss)
                self.low_writer.add_scalar('Loss/low/actor_loss',  actor_val, step)
                self.low_writer.add_scalar('Loss/low/critic_loss', q_val,     step)

                if step > 0 and step % self.args['eval_freq'] == 0:
                    self._save_step_policy(self.low_policy, self.low_checkpoint_dir, step)
                    self.save_critic(step, self.critic_checkpoint_dir)
                self.low_global_step = step + 1 

            torch.cuda.empty_cache()

        last_step = self.low_global_step
        self._save_step_policy(self.low_policy, self.low_checkpoint_dir, last_step)
        self.save_critic(last_step, self.critic_checkpoint_dir)

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

        return valid_value, mask

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
        print(f"Saved model at {save_dir}")

    
    def prepare_tensor(self, rewards, dones):
        device = self._input_device(self.low_policy)

        reward_list = [torch.as_tensor(seq, dtype=torch.float32, device=device)
                    for seq in rewards]
        done_list   = [torch.as_tensor(seq, dtype=torch.float32, device=device)
                    for seq in dones]

        reward_tensor = pad_sequence(reward_list, batch_first=True, padding_value=0.0)
        done_tensor   = pad_sequence(done_list,   batch_first=True, padding_value=0.0)
        return reward_tensor, done_tensor

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

        print(f"Saved model at {out_dir}")
        return out_dir
    

    def save_critic(self, step, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(self.critic.state_dict(), f"{checkpoint_dir}/critic.pt")
        print(f"Saved critic at {checkpoint_dir}/critic.pt")
