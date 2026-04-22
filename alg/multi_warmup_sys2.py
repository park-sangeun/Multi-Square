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
from prompt.inst import high_prompt, low_prompt, single_prompt

from util.extract import extract_action_done
from scienceworld import ScienceWorldEnv
from util.replay_buffer import OnlineDataset
            
class Multi2:
    def __init__(self, args):
        self.args = args

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
        self.buffer = HierarchyDataset(args)
        base_log = f"{args['log_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}/Offline"
        os.makedirs(base_log, exist_ok=True)

        self.warmup_writer = SummaryWriter(log_dir=os.path.join(base_log, "warmup"))
        self.low_policy  = LowPolicy(args);   self.low_policy.train()
        if hasattr(self.low_policy, "base"):
            self.low_policy.base.train()
        self.low_optim_sft = torch.optim.AdamW(
            [p for p in self.low_policy.base.parameters() if p.requires_grad],
            lr=self.args.get("lr_warmup", 5e-5),
            betas=(0.9, 0.999), eps=1e-8, weight_decay=self.args.get("wd_sft", 1e-2)
        )
        self.loss_fct = torch.nn.MSELoss()
        self.warmup_global_step = 0
        self.low_global_step  = 0

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

        base_ckpt = f"{args['check_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}/{args['lr_warmup']}"
        self.warmup_checkpoint_dir    = os.path.join(base_ckpt, "warmup",    timestamp)
        os.makedirs(self.warmup_checkpoint_dir, exist_ok=True)
        try:
            self.low_policy.base.gradient_checkpointing_enable()
        except Exception:
            pass
        
        self.eval_env = ScienceWorldEnv("", envStepLimit=self.args['env_step_limit'])
        self.log_file = os.path.join(
            f"{self.args['log_path']}/{self.args['benchmark']}/{self.args['alg_name']}/{self.args['model_name']}",
            "validation_seen.log"
        )
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.fixed_dev_seen_subset = None 

        val_seed = int(self.args.get("val_seed", 777))
        k_per_task = int(self.args.get("val_k_per_task", 10))
       
        ann_path = self.args.get("dev_annotation_path", "eval_results/eval_variations_dev_annotated.json")

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
       
    def _build_fixed_dev_seen_subset(self, annotation_path: str, k_per_task: int = 10, seed: int = 777):
        import json
        rng = random.Random(seed)

        ann_map = {}
        if annotation_path and os.path.exists(annotation_path):
            with open(annotation_path, "r") as f:
                ann_rows = json.load(f)
            for row in ann_rows:
                t_id = row["task_id"]
                v_id = row["variation_id"]
                label = row.get("is_seen", "Unknown") or "Unknown"
                ann_map[(t_id, v_id)] = label
        else:
            self.fixed_dev_seen_subset = {}
            return

        task_names = [
            "boil", "change-the-state-of-matter-of", "chemistry-mix",
            "chemistry-mix-paint-secondary-color", "chemistry-mix-paint-tertiary-color",
            "find-animal", "find-living-thing", "find-non-living-thing",
            "find-plant", "freeze", "grow-fruit", "grow-plant",
            "identify-life-stages-1", "identify-life-stages-2",
            "inclined-plane-determine-angle", "inclined-plane-friction-named-surfaces",
            "inclined-plane-friction-unnamed-surfaces", "lifespan-longest-lived",
            "lifespan-longest-lived-then-shortest-lived", "lifespan-shortest-lived",
            "measure-melting-point-known-substance", "measure-melting-point-unknown-substance",
            "melt", "mendelian-genetics-known-plant", "mendelian-genetics-unknown-plant",
            "power-component", "power-component-renewable-vs-nonrenewable-energy",
            "test-conductivity", "test-conductivity-of-unknown-substances", "use-thermometer"
        ]
        self.task_names = task_names

        fixed = {}
        for task_id, task_name in enumerate(task_names):
            try:
                self.eval_env.load(task_name)
                var_ids = list(self.eval_env.getVariationsDev() or [])
            except Exception:
                var_ids = []

            if not var_ids:
                fixed[task_id] = []
                continue

            seen_ids = [vid for vid in var_ids if ann_map.get((task_id, vid), "Unknown") == "Seen"]

            if not seen_ids:
                fixed[task_id] = []
                continue
            seen_ids = sorted(seen_ids)

            if len(seen_ids) > k_per_task:
                seen_ids = sorted(rng.sample(seen_ids, k_per_task))
            else:
                pass

            fixed[task_id] = seen_ids

        self.fixed_dev_seen_subset = fixed
        print("[VAL] Built fixed dev/Seen subset with seed=", seed)

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
        max_len = int(self.args.get("max_seq_len", 1000))
        def expectile_loss(diff: torch.Tensor, expectile: float = 0.7):
            weight = torch.where(diff > 0, expectile, 1 - expectile)
            return (weight * (diff ** 2)).mean()
        
        if level == "SFT":
            policy = self.low_policy
            optimizer = self.low_optim_sft
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
                if "attention_mask" in tokens:
                    attn_mask = tokens["attention_mask"]
                else:
                    pad_id = policy.tokenizer.pad_token_id
                    attn_mask = (tokens["input_ids"] != pad_id).long()

                lengths = attn_mask.sum(dim=1)

                keep_mask = lengths <= max_len

                if keep_mask.sum() == 0:
                    continue

                def _filter_tensor(t):
                    if not torch.is_tensor(t):
                        return t
                    if t.size(0) != keep_mask.size(0):
                        return t
                    return t[keep_mask]

                tokens = {k: _filter_tensor(v) for k, v in tokens.items()}
                effective_bs = int(keep_mask.sum().item())
                action_log_probs, action_masks = policy.get_log_prob(tokens)
                if 'action_end_mask' in tokens:
                    max_actions = int(tokens['action_end_mask'].sum(dim=1).max().item())
                else:
                    max_actions = action_masks.size(1)

                if max_actions == 0:
                    continue

                valid_action_log_probs = self._extract_valid_action_probs(
                    action_log_probs, action_masks, max_actions
                )
                bc_loss = -valid_action_log_probs.mean()

                bc_loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                bc_loss_total += bc_loss.item() * effective_bs
                seen += effective_bs

            return bc_loss_total / max(1, seen)

        else:
            raise ValueError(f"Unknown level: {level}")
            
         

    def update(self):
        self.env = ScienceWorldEnv("", envStepLimit=self.args['env_step_limit'])
        self.task_names = self.env.getTaskNames()

        micro_bs  = int(self.args.get("train_micro_batch_size_per_gpu", 8))
        
        dataloader = DataLoader(
            self.buffer,
            batch_size=micro_bs,
            shuffle=True,
            collate_fn=HierarchyDataset.collate_fn
        )
        start_step = int(self.warmup_global_step)
        for i, batch in enumerate(dataloader, start=start_step):
            warmup_loss = self.update_policy(batch['low'], level="SFT")
            warmup_val = float(warmup_loss.item() if isinstance(warmup_loss, torch.Tensor) else warmup_loss)

            self.warmup_writer.add_scalar('Loss/Warmup/SFT_loss', warmup_val, i)
        self.warmup_global_step = i + 1
        self._save_step_policy(self.low_policy, self.warmup_checkpoint_dir)
        

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
    def _save_policy(policy, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        policy.base.save_pretrained(save_dir)
        policy.tokenizer.save_pretrained(save_dir)
        print(f"Saved model at {save_dir}")

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
        save_dir = os.path.join(checkpoint_dir, str(step))
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.critic.state_dict(), f"{save_dir}/critic.pt")
        print(f"Saved critic at {save_dir}/critic.pt")
