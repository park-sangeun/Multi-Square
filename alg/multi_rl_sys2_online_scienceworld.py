from __future__ import annotations

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
import os
import json
import copy
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence

from util.model import HighPolicy, LowPolicy, Critic
from util.replay_buffer import batch_traj_process
from scienceworld import ScienceWorldEnv

try:
    from util.replay_buffer import OnlineDataset
except Exception:
    OnlineDataset = None

from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Episode:
    task_description: str
    obs: List[str]
    subtask: List[str]
    action: List[str]
    reward: List[float]
    done: List[float]
    task_name: Optional[str] = None
    task_id: Optional[int] = None
    variation_id: Optional[int] = None
    label: Optional[str] = None
    split: Optional[str] = None
    token_len: Optional[int] = None

class SimpleOnlineBuffer:
    def __init__(self, capacity_episodes: int = 2000, seed: int = 0):
        self.capacity = int(capacity_episodes)
        self.rng = random.Random(seed)
        self.episodes: List[Episode] = []
    def __len__(self) -> int:
        return len(self.episodes)
    def append_episode(self, ep: Episode):
        self.episodes.append(ep)
        if len(self.episodes) > self.capacity:
            extra = len(self.episodes) - self.capacity
            if extra > 0:
                self.episodes = self.episodes[extra:]
    def sample_batch(self, batch_size: int) -> Dict[str, List[Any]]:
        assert len(self.episodes) > 0, "Online buffer is empty."
        bs = min(int(batch_size), len(self.episodes))
        eps = self.rng.sample(self.episodes, bs)
        return {
            "task_description": [e.task_description for e in eps],
            "obs": [e.obs for e in eps],
            "subtask": [e.subtask for e in eps],
            "action": [e.action for e in eps],
            "reward": [e.reward for e in eps],
            "done": [e.done for e in eps],
        }

class Multi2:
    def __init__(self, args: Dict[str, Any]):
        self.args = args
        self.global_rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', 0)))
        self.local_rank  = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size  = int(os.environ.get('WORLD_SIZE', 1))

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

        self.low_policy = LowPolicy(args)
        self.low_policy.train()
        self.critic = Critic(args)
        self.critic.train()

        self.high_policy = HighPolicy(args);  self.high_policy.train()
        try:
            _dev = self._input_device(self.low_policy)
        except Exception:
            _dev = next(self.low_policy.base.parameters()).device

        self.critic.to(_dev)


        for _name in ['value_head','q_head','target_value_head','target_q_head']:
            if hasattr(self.critic, _name):
                getattr(self.critic, _name).to(_dev)

        self.critic_optim = torch.optim.AdamW(
            [p for p in self.critic.parameters() if p.requires_grad],
            lr=args.get("critic_lr", 1e-4), betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
        )
        self.low_optim = torch.optim.AdamW(
            [p for p in self.low_policy.base.parameters() if p.requires_grad],
            lr=args.get("actor_lr", 1e-5), betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
        )

        # Path to the trained System 1 (SFT) checkpoint
        high_path = "/path/to/your/system1_sft_checkpoint"

        # Path to the trained System 2 warmup checkpoint
        low_path = "/path/to/your/system2_offline-rl-trained_policy_checkpoint"

        BC_AGENT.load_high_policy(self, high_path)
        self.high_policy.eval()
        BC_AGENT.load_low_policy(self, low_path)

        def _count_trainable(m):
            n = sum(p.numel() for p in m.parameters() if p.requires_grad)
            a = sum(p.numel() for p in m.parameters())
            return n, a

        from datetime import datetime
        self.run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")

        base_log = f"{args['log_path']}/{args['benchmark']}/multi_Online/{args['model_name']}/{self.run_timestamp}"

        os.makedirs(base_log, exist_ok=True)
        self.writer_actor = SummaryWriter(log_dir=os.path.join(base_log, "actor"))
        self.writer_critic = SummaryWriter(log_dir=os.path.join(base_log, "critic"))
        self.env = ScienceWorldEnv("", envStepLimit=int(args.get("env_step_limit", 50)))

        cap = int(self.args.get("online_capacity_episodes", 2000))
        seed = int(self.args.get("seed", 0))

        self.offline_buffer = SimpleOnlineBuffer(
            capacity_episodes=int(self.args.get("offline_capacity_episodes", cap)),
            seed=seed + 123
        )
        self.online_buffer = SimpleOnlineBuffer(
            capacity_episodes=cap,
            seed=seed
        )
        self._use_project_online_buffer = False
        self.seed_online_buffer_from_offline()
        self.global_step = 0

        dev_low = self._input_device(self.low_policy)
        base_dtype = next(self.low_policy.base.parameters()).dtype
        try:
            self.low_policy.base.gradient_checkpointing_enable()
        except Exception:
            pass
        try:
            self.critic.base.gradient_checkpointing_enable()
        except Exception:
            pass

        try:
            if hasattr(self.low_policy, "base") and hasattr(self.low_policy.base, "config"):
                self.low_policy.base.config.use_cache = False
            if hasattr(self.critic, "base") and hasattr(self.critic.base, "config"):
                self.critic.base.config.use_cache = False
            if hasattr(self.high_policy, "base") and hasattr(self.high_policy.base, "config"):
                self.high_policy.base.config.use_cache = False
        except Exception:
            pass

        self._use_amp = bool(self.args.get("use_amp", True))
        self._amp_dtype = torch.bfloat16 if self.args.get("amp_dtype", "bf16") == "bf16" else torch.float16
        self._scaler = torch.cuda.amp.GradScaler(enabled=(self._use_amp and self._amp_dtype == torch.float16))

        self.task_names = [
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

    def _estimate_low_traj_token_len(
        self,
        subtask_prompt: str,
        obs: List[str],
        actions: List[str],
        limit: int,
    ) -> int:
        tok = self.low_policy.tokenizer
        eos = tok.eos_token or ""
        n = 0
        n += len(tok.encode(str(subtask_prompt), add_special_tokens=True))
        if n > limit:
            return n
        T = min(len(obs), len(actions))
        for t in range(T):
            n += len(tok.encode("Obs: " + str(obs[t]), add_special_tokens=False))
            if n > limit:
                return n
            n += len(tok.encode(str(actions[t]) + eos, add_special_tokens=False))
            if n > limit:
                return n
        return n

    def _input_device(self, module) -> torch.device:
        try:
            if hasattr(module, "get_input_embeddings"):
                return module.get_input_embeddings().weight.device
            if hasattr(module, "base") and hasattr(module.base, "get_input_embeddings"):
                return module.base.get_input_embeddings().weight.device
            return next(module.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _move_high_policy(self, device: torch.device | str):
        if not hasattr(self, "high_policy") or self.high_policy is None:
            return
        try:
            self.high_policy.to(device)
        except Exception:

            if hasattr(self.high_policy, "base"):
                self.high_policy.base.to(device)

    def _pad_or_trunc_bn(self, x: torch.Tensor, N: int, pad_value: float) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"_pad_or_trunc_bn expects 2D tensor, got shape={tuple(x.shape)}")
        B, L = x.shape
        if L == N:
            return x
        if L > N:
            return x[:, :N]

        pad = x.new_full((B, N - L), float(pad_value))
        return torch.cat([x, pad], dim=1)

    def _prepare_tensor_bn(self, seqs: List[List[float]], pad_value: float) -> torch.Tensor:
        device = self._input_device(self.low_policy)
        t_list = [torch.as_tensor(s, dtype=torch.float32, device=device) for s in seqs]
        return pad_sequence(t_list, batch_first=True, padding_value=float(pad_value))

    @torch.no_grad()
    def rollout_one_episode(
        self,
        task_id: int,
        variation_id: int,
        label: str,
        max_steps: int | None = None,
        debug: bool = False,
    ) -> list[Episode]:
        if max_steps is None:
            max_steps = int(self.args.get("env_step_limit", 100))
        high_device = self._input_device(self.high_policy)
        low_device  = self._input_device(self.low_policy)
        task_name = self.task_names[int(task_id)]
        self.env.load(task_name, int(variation_id))
        task_desc = str(self.env.taskdescription())
        task_description = (high_prompt + " " + task_desc).strip()
        obs, _ = self.env.reset()
        done = False
        episode_steps = 0
        high_traj = self.high_policy.tokenizer(task_description, return_tensors="pt")
        high_traj = {k: (v.to(high_device) if torch.is_tensor(v) else v) for k, v in high_traj.items()}
        low_episodes: list[Episode] = []
        group_action: list[str] = []
        def _truncate(traj, max_len):
            for k in ["input_ids","attention_mask"]:
                if traj[k].size(1) > max_len:
                    traj[k] = traj[k][:, -max_len:]
            return traj
        max_rollout_ctx = int(self.args.get("rollout_max_ctx", 1024))

        while (not done) and (episode_steps < max_steps):
            state = f"Group action: {group_action}. Current observation: {obs}"
            state_tok = self.high_policy.tokenizer(state, return_tensors="pt")
            state_tok = {k: (v.to(high_device) if torch.is_tensor(v) else v) for k, v in state_tok.items()}
            high_traj["input_ids"] = torch.cat([high_traj["input_ids"], state_tok["input_ids"]], dim=1)
            high_traj["attention_mask"] = torch.cat([high_traj["attention_mask"], state_tok["attention_mask"]], dim=1)
            subtask = self.high_policy.generate_action(high_traj)[0]
            subtask = str(subtask)
            subtask_tok = self.high_policy.tokenizer(subtask + self.high_policy.tokenizer.eos_token, return_tensors="pt")
            subtask_tok = {k: (v.to(high_device) if torch.is_tensor(v) else v) for k, v in subtask_tok.items()}
            high_traj["input_ids"] = torch.cat([high_traj["input_ids"], subtask_tok["input_ids"]], dim=1)
            high_traj["attention_mask"] = torch.cat([high_traj["attention_mask"], subtask_tok["attention_mask"]], dim=1)
            low_prompt_str = f"{low_prompt} Subtask: {subtask}"
            low_traj = self.low_policy.tokenizer(low_prompt_str, return_tensors="pt")
            low_traj = {k: (v.to(low_device) if torch.is_tensor(v) else v) for k, v in low_traj.items()}
            cur_obs: list[str] = []
            cur_act: list[str] = []
            cur_rew: list[float] = []
            cur_done: list[float] = []

            subtask_done = False
            group_action = []

            while (not subtask_done) and (not done) and (episode_steps < max_steps):
                episode_steps += 1
                obs_text = str(obs)
                cur_obs.append(obs_text)
                obs_tok = self.low_policy.tokenizer("Obs: " + obs_text, return_tensors="pt")
                obs_tok = {k: (v.to(low_device) if torch.is_tensor(v) else v) for k, v in obs_tok.items()}
                low_traj["input_ids"] = torch.cat([low_traj["input_ids"], obs_tok["input_ids"]], dim=1)
                low_traj["attention_mask"] = torch.cat([low_traj["attention_mask"], obs_tok["attention_mask"]], dim=1)

                raw_action = self.low_policy.generate_action(low_traj)[0]
                raw_action = str(raw_action)
                action, subtask_done = extract_action_done(raw_action)
                action = str(action) if action is not None else "look"
                group_action.append(action)

                act_tok = self.low_policy.tokenizer(raw_action + self.low_policy.tokenizer.eos_token, return_tensors="pt")
                act_tok = {k: (v.to(low_device) if torch.is_tensor(v) else v) for k, v in act_tok.items()}
                low_traj["input_ids"] = torch.cat([low_traj["input_ids"], act_tok["input_ids"]], dim=1)
                low_traj["attention_mask"] = torch.cat([low_traj["attention_mask"], act_tok["attention_mask"]], dim=1)

                obs, reward, done, info = self.env.step(action)
                r = float(reward) / 100.0
                cur_act.append(raw_action)
                cur_rew.append(r)
                cur_done.append(1.0 if bool(done) else 0.0)
            T = min(len(cur_obs), len(cur_act), len(cur_rew), len(cur_done))
            cur_obs, cur_act, cur_rew, cur_done = cur_obs[:T], cur_act[:T], cur_rew[:T], cur_done[:T]

            max_train_tokens = int(self.args.get("max_train_tokens", 3000))

            tok_len = self._estimate_low_traj_token_len(
                subtask_prompt=low_prompt_str,
                obs=cur_obs,
                actions=cur_act,
                limit=max_train_tokens + 1,
            )
            if tok_len is not None and tok_len > max_train_tokens:
                continue
            if T > 0:
                low_episodes.append(
                    Episode(
                        task_description=task_description,
                        obs=cur_obs,
                        subtask=low_prompt_str,
                        action=cur_act,
                        reward=cur_rew,
                        done=cur_done,
                        task_name=task_name,
                        task_id=int(task_id),
                        variation_id=int(variation_id),
                        label=str(label),
                        split=str(self.args.get("online_split", "dev")),
                    )
                )
            if debug and self.global_rank == 0:
                print(f"[DEBUG][rollout] task={task_name} var={variation_id} label={label} | "
                    f"subtask='{subtask[:60]}' | steps(subtask)={T} | episode_steps={episode_steps} done={done}")
        return low_episodes

    def seed_online_buffer_from_offline(self):
        path = self.args.get("seed_offline_low_path",
                            f"./dataset/{self.args.get('benchmark')}/low_data/expert.json")\
            or self.args.get("offline_low_path", None)\
            or self.args.get("expert_low_path", None)
        if path is None or (not os.path.exists(path)):
            if self.global_rank == 0:
                print(f"[seed_offline] skip (path not set or not found): {path}")
            return

        with open(path, "r") as f:
            data = json.load(f)

        required = ["subtask", "obs", "action", "reward", "done"]
        if isinstance(data, dict) and all(k in data for k in required) and all(isinstance(data[k], list) for k in required):
            n = min(len(data[k]) for k in required)
            optional_keys = [k for k in data.keys() if k not in required]
            data_rows = []
            for i in range(n):
                row = {k: data[k][i] for k in required}
                for k in optional_keys:

                    try:
                        row[k] = data[k][i]
                    except Exception:
                        pass
                data_rows.append(row)
            data = data_rows

        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                data = data["data"]
            elif "episodes" in data and isinstance(data["episodes"], list):
                data = data["episodes"]
            else:
                raise ValueError(f"[seed_offline] unsupported dict format keys={list(data.keys())[:20]}")
        if not isinstance(data, list):
            raise ValueError(f"[seed_offline] expected list, got {type(data)}")
        num_items = self.args.get("seed_offline_num_items", None)
        frac = float(self.args.get("seed_offline_frac", 0.0))
        success_only = bool(self.args.get("seed_offline_success_only", False))
        do_shuffle = bool(self.args.get("seed_offline_shuffle", True))

        rng = random.Random(int(self.args.get("seed", 0)) + 12345)
        if do_shuffle:
            rng.shuffle(data)

        if num_items is None and frac > 0:
            num_items = int(len(data) * frac)
        if num_items is None:
            num_items = min(len(data), int(self.args.get("online_capacity_episodes", 2000)))

        max_tok = int(self.args.get("max_train_tokens", 3000))

        pushed = 0
        skipped_long = 0
        skipped_fail = 0

        for row in data[:num_items]:
            if not isinstance(row, dict):
                continue

            obs = row.get("obs", None)
            action = row.get("action", None)
            reward = row.get("reward", None)
            done = row.get("done", None)
            if (not obs) or (not action) or (reward is None) or (done is None):
                continue
            T = min(len(obs), len(action), len(reward), len(done))
            if len(obs) == len(action) + 1:

                obs = obs[:-1]
            T = min(len(obs), len(action), len(reward), len(done))
            obs, action, reward, done = obs[:T], action[:T], reward[:T], done[:T]

            subtask_prompt = row.get("subtask", "")
            if isinstance(subtask_prompt, list):
                subtask_prompt = " ".join([str(x) for x in subtask_prompt])
            subtask_prompt = str(subtask_prompt)
            if success_only:
                try:
                    if float(np.max(np.array(reward))) <= 0.0:
                        skipped_fail += 1
                        continue
                except Exception:
                    pass
            try:
                tlen = self._estimate_low_traj_token_len(
                    subtask_prompt=subtask_prompt,
                    obs=list(obs),
                    actions=list(action),
                    limit=max_tok + 1,
                )
            except Exception:
                tlen = None

            if tlen is not None and int(tlen) > max_tok:
                skipped_long += 1
                continue
            ep = Episode(
                task_description=str(row.get("task_description", "")),
                obs=list(obs),
                subtask=subtask_prompt,
                action=list(action),
                reward=[float(x) for x in reward],
                done=[float(x) for x in done],
                task_name=row.get("task_name", None),
                task_id=row.get("task_id", None),
                variation_id=row.get("variation_id", None),
                label=row.get("label", None),
                split=row.get("split", None),
                token_len=int(tlen) if tlen is not None else None,
            )

            if getattr(self, "_use_project_online_buffer", False):
                self.online_buffer.add_episode(ep.__dict__)
            else:
                self.offline_buffer.append_episode(ep)

            pushed += 1

        if self.global_rank == 0:
            print(f"[seed_offline] loaded={len(data)} requested={num_items} pushed={pushed} "
                f"skipped_long={skipped_long} skipped_fail={skipped_fail} buffer_size={len(self.online_buffer)}")

    def collect_online_data(self) -> int:
        push_mode = str(self.args.get("online_push_mode", "downsample_fail"))
        min_return = float(self.args.get("online_min_return", 0.0))
        keep_fail_ratio = float(self.args.get("online_keep_fail_ratio", 0.1))

        offload_high = bool(self.args.get("offload_high_policy", True))


        if offload_high:
            high_target = self._input_device(self.low_policy)
            self._move_high_policy(high_target)
            self.high_policy.eval()
        try:
            num_eps = int(self.args.get("online_episodes_per_epoch", 10))
            split = self.args.get("online_split", "dev")
            ann_path = self.args.get("online_annotation_path", None)
            label_filter = self.args.get("online_label_filter", None)
            k_per_task = int(self.args.get("online_k_per_task", 50))
            max_steps = int(self.args.get("env_step_limit", 100))
            pool = self.build_variation_pool(
                split="train",
                annotation_path=[
                    "./eval_results/eval_variations_dev_annotated.json",
                    "./eval_results/eval_variations_test_annotated.json",
                ],
                k_per_task=10,
            )
            if not pool:
                if self.global_rank == 0:
                    print(f"[collect_online_data] empty pool (split={split}, label_filter={label_filter})")
                return 0
            rng = random.Random(int(self.args.get("seed", 0)) + int(self.global_step))
            rng.shuffle(pool)

            collected_env_eps = 0
            skipped_long = 0
            pushed_items = 0

            def _episode_stats_from_low_eps(low_eps):

                if not low_eps:
                    return {
                        "num_low": 0,
                        "sum_return": 0.0,
                        "mean_return_per_low": 0.0,
                        "success_low": 0,
                        "success_low_ratio": 0.0,
                        "max_step_reward": 0.0,
                    }
                low_returns = []
                success_low = 0
                max_step_reward = 0.0
                for ep in low_eps:
                    r = float(np.sum(ep.reward)) if ep.reward is not None else 0.0
                    low_returns.append(r)
                    if r > 0.0:
                        success_low += 1
                    if ep.reward:
                        max_step_reward = max(max_step_reward, float(np.max(ep.reward)))
                sum_return = float(np.sum(low_returns))
                mean_return = float(np.mean(low_returns)) if len(low_returns) > 0 else 0.0
                ratio = float(success_low / max(1, len(low_eps)))

                return {
                    "num_low": int(len(low_eps)),
                    "sum_return": sum_return,
                    "mean_return_per_low": mean_return,
                    "success_low": int(success_low),
                    "success_low_ratio": ratio,
                    "max_step_reward": float(max_step_reward),
                }
            for item in pool:
                if collected_env_eps >= num_eps:
                    break
                try:
                    task_id, task_name, var_id, label = item
                except Exception:
                    task_id, var_id, label = item
                    task_name = self.task_names[int(task_id)]

                debug = (self.global_rank == 0 and collected_env_eps < 2)

                low_eps = self.rollout_one_episode(
                    task_id=int(task_id),
                    variation_id=int(var_id),
                    label=str(label),
                    max_steps=max_steps,
                    debug=debug,
                )
                if not low_eps:
                    if self.global_rank == 0 and debug:
                        print("[collect_online_data] rollout produced 0 low trajectories; skip")
                    collected_env_eps += 1
                    continue
                stats = _episode_stats_from_low_eps(low_eps)

                if self.global_rank == 0:
                    print(
                        f"[collect_online_data][perf] env_ep#{collected_env_eps} "
                        f"low_trajs={stats['num_low']} "
                        f"sumR={stats['sum_return']:.3f} "
                        f"meanR/low={stats['mean_return_per_low']:.3f} "
                        f"low_success={stats['success_low']}/{stats['num_low']}({stats['success_low_ratio']*100:.1f}%) "
                        f"max_step_r={stats['max_step_reward']:.3f}"
                    )
                pushed_this_env = 0
                pushed_success = 0
                pushed_fail = 0

                for ep in low_eps:
                    max_ctx = int(self.args.get("max_ctx", self.args.get("max_length", 2048)))
                    max_tok = int(self.args.get("max_train_tokens", 3000))
                    max_ctx = min(max_ctx, max_tok)

                    if (not ep.obs) or (not ep.action):
                        continue
                    ep_return = float(np.sum(ep.reward)) if ep.reward is not None else 0.0
                    is_success = (ep_return > min_return)
                    if push_mode == "success_only":
                        if not is_success:
                            continue
                    elif push_mode == "downsample_fail":
                        if (not is_success) and (rng.random() > keep_fail_ratio):
                            continue
                    tlen = self._estimate_low_traj_token_len(
                        subtask_prompt=ep.subtask,
                        obs=ep.obs,
                        actions=ep.action,
                        limit=max_tok + 1,
                    )
                    ep.token_len = int(tlen)
                    if tlen > max_tok:
                        skipped_long += 1
                        continue


                    try:
                        ep.return_sum = ep_return
                        ep.is_success = bool(is_success)
                        ep.source = "online"
                    except Exception:
                        pass

                    if getattr(self, "_use_project_online_buffer", False):
                        d = ep.__dict__
                        d["return_sum"] = ep_return
                        d["is_success"] = bool(is_success)
                        d["source"] = "online"
                        self.online_buffer.add_episode(d)
                    else:
                        self.online_buffer.append_episode(ep)
                    pushed_items += 1
                    pushed_this_env += 1
                    if is_success:
                        pushed_success += 1
                    else:
                        pushed_fail += 1

                if self.global_rank == 0 and debug:
                    print(f"[collect_online_data] env_ep#{collected_env_eps} -> pushed low_trajs={len(low_eps)} "
                        f"(buffer_size={len(self.online_buffer)})")
                collected_env_eps += 1
            if self.global_rank == 0:
                print(f"[collect_online_data] collected_env_eps={collected_env_eps} pushed_items={pushed_items} "
                    f"buffer_size={len(self.online_buffer)}")
            return collected_env_eps

        finally:
            if offload_high:
                self._move_high_policy("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    def build_variation_pool(
        self,
        split="train",
        annotation_path=None,
        label_filter=None,
        task_filter=None,
        k_per_task=10,
    ):
        task_names = getattr(self, "task_names", None)
        if not task_names:
            task_names = [
                "boil","change-the-state-of-matter-of","chemistry-mix",
                "chemistry-mix-paint-secondary-color","chemistry-mix-paint-tertiary-color",
                "find-animal","find-living-thing","find-non-living-thing",
                "find-plant","freeze","grow-fruit","grow-plant",
                "identify-life-stages-1","identify-life-stages-2",
                "inclined-plane-determine-angle","inclined-plane-friction-named-surfaces",
                "inclined-plane-friction-unnamed-surfaces","lifespan-longest-lived",
                "lifespan-longest-lived-then-shortest-lived","lifespan-shortest-lived",
                "measure-melting-point-known-substance","measure-melting-point-unknown-substance",
                "melt","mendelian-genetics-known-plant","mendelian-genetics-unknown-plant",
                "power-component","power-component-renewable-vs-nonrenewable-energy",
                "test-conductivity","test-conductivity-of-unknown-substances","use-thermometer"
            ]
        ann_map = {}
        ann_paths = []
        if isinstance(annotation_path, (list, tuple)):
            ann_paths = [p for p in annotation_path if p]
        elif isinstance(annotation_path, str) and annotation_path:
            ann_paths = [annotation_path]
        for ap in ann_paths:
            if not os.path.exists(ap):
                continue
            with open(ap, "r") as f:
                ann_rows = json.load(f)

            for row in ann_rows:
                is_seen = row.get("is_seen", None)
                if isinstance(is_seen, bool):
                    label = "Seen" if is_seen else "Unseen"
                else:
                    label = row.get("label", "Unknown")
                ann_map[(int(row["task_id"]), int(row["variation_id"]))] = label
        pool = []
        for task_id, task_name in enumerate(task_names):
            if task_filter is not None and task_id not in task_filter:
                continue

            self.env.load(task_name)
            if split == "dev":
                var_ids = list(self.env.getVariationsDev() or [])
            elif split == "test":
                var_ids = list(self.env.getVariationsTest() or [])
            else:
                var_ids = list(self.env.getVariationsTrain() or [])
            if not var_ids:
                continue
            sel = sorted(var_ids)[:min(k_per_task, len(var_ids))]
            for v in sel:
                label = ann_map.get((task_id, v), "Unknown")
                if label_filter is not None and label != label_filter:
                    continue
                pool.append((task_id, task_name, v, label))
        return pool

    def _get_ref_low_policy(self):
        if getattr(self, "offline_low_policy", None) is None:
            ref_args = dict(self.args)
            ref_args["device_map"] = {"": "cpu"}
            ref = LowPolicy(ref_args)
            ref.eval()
            for p in ref.parameters():
                p.requires_grad_(False)
            sd = {k: v.detach().cpu() for k, v in self.low_policy.base.state_dict().items()}
            ref.base.load_state_dict(sd, strict=False)
            self.offline_low_policy = ref
        return self.offline_low_policy

    def update_awac_on_batch(self, batch_data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        actor = self.low_policy
        critic = self.critic
        actor_optim = self.low_optim
        critic_optim = self.critic_optim
        dev = next(critic.parameters()).device
        for name in ["value_head", "q_head", "target_value_head", "target_q_head"]:
            if hasattr(critic, name):
                try:
                    getattr(critic, name).to(dev)
                except Exception:
                    pass
        actor_dev = self._input_device(actor)
        gamma = float(self.args.get("gamma", self.args.get("gama", 0.99)))
        B = len(batch_data["subtask"])
        micro_bs = int(self.args.get("micro_batch_size", self.args.get("awac_micro_batch_size", B)))
        micro_bs = max(1, min(micro_bs, B))
        eta = float(self.args.get("kl_coef", self.args.get("kl_eta", self.args.get("eta", 0.01))))
        ref = (
            getattr(self, "offline_low_policy", None)
            or getattr(self, "ref_low_policy", None)
            or getattr(self, "low_policy_ref", None)
            or getattr(self, "low_policy_offline", None)
        )
        if eta > 0.0 and ref is None:
            ref = self._get_ref_low_policy()
            self.ref_low_policy = ref

        self.ref_low_policy = ref
        if self.ref_low_policy is not None:
            self.ref_low_policy.eval()
            for p in self.ref_low_policy.parameters():
                p.requires_grad_(False)
        assert ref is None or ref is not self.low_policy, "ref_low_policy is the SAME object as actor!"
        q_loss_sum = 0.0
        actor_loss_sum = 0.0
        kl_sum = 0.0
        w_sum = 0.0
        adv_sum = 0.0
        count = 0
        empty_cache_each_step = bool(self.args.get("empty_cache_each_step", False))

        for start in range(0, B, micro_bs):
            end = min(B, start + micro_bs)

            batch_slice = {k: v[start:end] for k, v in batch_data.items()}


            keep_idx = [i for i, a in enumerate(batch_slice["action"]) if (a is not None and len(a) > 0)]
            if len(keep_idx) == 0:
                continue

            if len(keep_idx) != len(batch_slice["action"]):
                batch_slice = {k: [v[i] for i in keep_idx] for k, v in batch_slice.items()}

            b2 = len(batch_slice["subtask"])

            max_ctx = int(self.args.get("max_ctx", self.args.get("max_length", 2048)))
            max_tok = int(self.args.get("max_train_tokens", 3000))
            max_ctx = min(max_ctx, max_tok)
            tokens = batch_traj_process(
                batch_slice["subtask"],
                batch_slice["obs"],
                batch_slice["action"],
                actor.tokenizer,

                max_length=max_ctx)
            tokens = {k: (v.to(actor_dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in tokens.items()}
            rewards = self._prepare_tensor_bn(batch_slice["reward"], pad_value=0.0).to(dev)
            dones   = self._prepare_tensor_bn(batch_slice["done"],   pad_value=1.0).to(dev)
            with torch.no_grad():
                hidden_states, _, action_end_mask_actor = actor.get_hidden_states(tokens)
            valid_counts = action_end_mask_actor.sum(dim=1)
            keep = valid_counts > 0
            if keep.sum().item() == 0:
                continue
            if keep.sum().item() < keep.numel():
                idx = torch.nonzero(keep, as_tuple=False).squeeze(-1)
                for k, v in list(tokens.items()):
                    if torch.is_tensor(v) and v.size(0) == keep.size(0):
                        tokens[k] = v.index_select(0, idx)
                hidden_states = hidden_states.index_select(0, idx)
                action_end_mask_actor = action_end_mask_actor.index_select(0, idx)
                rewards = rewards.index_select(0, idx)
                dones   = dones.index_select(0, idx)
            try:
                head_param = next(critic.parameters())
                if hasattr(critic, "q_head"):
                    head_param = next(critic.q_head.parameters())
            except Exception:
                head_param = next(critic.parameters())

            h = hidden_states.to(device=head_param.device, dtype=head_param.dtype, non_blocking=True)
            a_mask = action_end_mask_actor.to(device=head_param.device, non_blocking=True)
            with torch.no_grad():
                q_tgt_all = critic.target_q_head(h).squeeze(-1)
                q_tgt, q_mask = self.extract_valid(q_tgt_all, a_mask)
            q_sa_all = critic.q_head(h).squeeze(-1)
            q_sa, q_mask_sa = self.extract_valid(q_sa_all, a_mask)
            if q_sa.size(1) == 0 or q_mask_sa.sum().item() == 0:
                continue

            q_mask = q_mask_sa

            N = q_sa.size(1)
            rewards = self._pad_or_trunc_bn(rewards, N, pad_value=0.0).to(q_sa.device, non_blocking=True)
            dones   = self._pad_or_trunc_bn(dones,   N, pad_value=1.0).to(q_sa.device, non_blocking=True)
            q_tgt   = q_tgt.to(q_sa.device, non_blocking=True)
            q_mask  = self._pad_or_trunc_bn(q_mask,  N, pad_value=0.0).to(q_sa.device, non_blocking=True).to(torch.float32)
            next_q = torch.zeros_like(q_tgt)
            if N > 1:
                next_q[:, :-1] = q_tgt[:, 1:]

            target = rewards + (1.0 - dones) * next_q * gamma
            td_err2 = (q_sa.float() - target.float()).pow(2)

            q_loss = (td_err2 * q_mask).sum() / q_mask.sum().clamp_min(1.0)
            critic_optim.zero_grad(set_to_none=True)
            if not q_loss.requires_grad:
                continue
            q_loss.backward()
            critic_optim.step()
            critic.soft_update_target_critic(tau=float(self.args.get("tau", 0.01)))
            q_tgt_small = q_tgt.detach()
            del hidden_states, h, a_mask, q_tgt_all, q_sa_all, q_sa, q_mask, q_mask_sa, rewards, dones, next_q, target, td_err2
            if empty_cache_each_step and torch.cuda.is_available():
                torch.cuda.empty_cache()
            logp_on, masks_on = actor.get_log_prob(tokens)
            action_counts = action_end_mask_actor.sum(dim=1).to(dtype=torch.long)
            max_actions = int(action_counts.max().item()) if action_counts.numel() > 0 else 0

            logp_on_valid = self._extract_valid_action_probs(
                logp_on, masks_on, max_actions
            ).to(torch.float32)

            act_mask = (
                torch.arange(max_actions, device=logp_on_valid.device)[None, :]
                < action_counts.to(logp_on_valid.device)[:, None]
            ).to(torch.float32)
            denom_total = act_mask.sum().clamp_min(1.0)


            with torch.no_grad():
                q_for_adv = q_tgt_small.to(logp_on_valid.device, non_blocking=True)
                if q_for_adv.size(1) != max_actions:
                    q_for_adv = self._pad_or_trunc_bn(q_for_adv, max_actions, pad_value=0.0)

                denom_traj = act_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                v = (q_for_adv * act_mask).sum(dim=1, keepdim=True) / denom_traj
                adv = (q_for_adv - v).to(torch.float32)
                alpha = float(self.args.get("awac_alpha", self.args.get("awac_lambda", 1.0)))
                w = torch.exp(torch.clamp(adv / max(alpha, 1e-6), min=-20.0, max=20.0))
                if bool(self.args.get("normalize_awac_weights", True)):
                    w_mean_traj = (w * act_mask).sum(dim=1, keepdim=True) / denom_traj
                    w = w / w_mean_traj.clamp_min(1e-6)
                w_clip = float(self.args.get("awac_weight_clip", 20.0))
                w = torch.clamp(w, max=w_clip) * act_mask
            logp_off_valid = None
            kl = torch.zeros((), device=logp_on_valid.device, dtype=torch.float32)
            if eta > 0.0 and ref is not None:
                ref_dev = next(ref.parameters()).device
                need_keys = ("input_ids", "attention_mask")
                tokens_ref = {}
                for k in need_keys:
                    if k in tokens:
                        v = tokens[k]
                        tokens_ref[k] = v.detach().to(ref_dev, non_blocking=False)
                for k in ("action_end_mask", "state_end_mask"):
                    if k in tokens:
                        tokens_ref[k] = tokens[k].detach().to(ref_dev, non_blocking=False)
                with torch.inference_mode():
                    logp_off, masks_off = ref.get_log_prob(tokens_ref)
                    logp_off_valid = self._extract_valid_action_probs(
                        logp_off, masks_off, max_actions
                    ).to(dtype=torch.float32)
                del logp_off, masks_off, tokens_ref
                if torch.cuda.is_available() and str(ref_dev).startswith("cuda"):
                    torch.cuda.empty_cache()

                logp_off_valid = logp_off_valid.to(logp_on_valid.device, non_blocking=False)

                kl_per = (logp_on_valid - logp_off_valid) * act_mask
                kl = kl_per.sum() / denom_total
                bc = (-(w * logp_on_valid).sum() / denom_total)
                actor_loss = bc + eta * kl
            else:
                kl = torch.zeros((), device=logp_on_valid.device, dtype=torch.float32)
                actor_loss = (-(w * logp_on_valid).sum() / denom_total)
            actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_optim.step()
            b_eff = len(batch_slice["subtask"])
            q_loss_sum     += float(q_loss.detach().item()) * b_eff
            actor_loss_sum += float(actor_loss.detach().item()) * b_eff
            kl_sum         += float(kl.detach().item()) * b_eff
            count          += b_eff
            w_sum  += float(((w.detach() * act_mask).sum() / denom_total).item()) * b_eff
            adv_sum+= float(((adv.detach() * act_mask).sum() / denom_total).item()) * b_eff
            del tokens, action_end_mask_actor, q_tgt_small, logp_on, masks_on, logp_on_valid, action_counts, act_mask, denom_total, w, adv, kl
            if empty_cache_each_step and torch.cuda.is_available():
                torch.cuda.empty_cache()
        denom = max(1, count)
        diag = {
            "q_loss": q_loss_sum / denom,
            "actor_loss": actor_loss_sum / denom,
            "kl_mean": kl_sum / denom,
            "w_mean": w_sum / denom,
            "adv_mean": adv_sum / denom,
            "eta": float(eta),
            "micro_batch_size": int(micro_bs),
        }
        return torch.tensor(diag["q_loss"], device=dev), torch.tensor(diag["actor_loss"], device=dev), diag

    def update(self):
        def _cuda_mem(msg=""):
            if not torch.cuda.is_available():
                return
            torch.cuda.synchronize()
            a = torch.cuda.memory_allocated() / 2**30
            r = torch.cuda.memory_reserved() / 2**30
            p = torch.cuda.max_memory_allocated() / 2**30
            print(f"[mem] {msg} alloc={a:.2f}G reserved={r:.2f}G peak={p:.2f}G")

        epochs = int(self.args.get("epochs", 1))
        batch_size = int(self.args.get("online_batch_episodes", 1))
        updates_per_epoch = int(self.args.get("updates_per_epoch", 50))

        save_every = int(self.args.get("save_every_epochs", 10))
        for epoch in range(epochs):
            collected = self.collect_online_data()
            print(f"[epoch {epoch}] collected episodes = {collected} | buffer size = {len(self.online_buffer)}")

            if len(self.online_buffer) == 0:
                continue
            for u in range(updates_per_epoch):

                if self._use_project_online_buffer:
                    batch = self.online_buffer.sample_batch(batch_size)
                else:
                    batch = self.online_buffer.sample_batch(batch_size)
                torch.cuda.reset_peak_memory_stats()
                q_loss, a_loss, diag = self.update_awac_on_batch(batch)

                if (self.global_step % int(self.args.get("log_freq", 10))) == 0:
                    print(
                        f"[epoch {epoch} | upd {u+1}/{updates_per_epoch} | step {self.global_step}] "
                        f"critic_loss={float(q_loss):.6f} actor_loss={float(a_loss):.6f} "
                        f"kl={diag.get('kl_mean', 0.0):.6f} w={diag.get('w_mean', 0.0):.3f} adv={diag.get('adv_mean', 0.0):.3f}"
                    )

                    self.writer_critic.add_scalar("loss/critic", float(q_loss), self.global_step)
                    self.writer_actor.add_scalar("loss/actor", float(a_loss), self.global_step)
                self.global_step += 1
            if ((epoch + 1) % save_every) == 0:
                self.save(tag=f"epoch{epoch+1}")
            torch.cuda.empty_cache()
        self.save(tag=f"epoch{epoch+1}")

    import torch
    from torch.nn.utils.rnn import pad_sequence

    def extract_valid(self, value: torch.Tensor, valid_mark: torch.Tensor):
        B, T = value.shape
        outs = []
        masks = []
        max_len = 0

        for i in range(B):
            idx = torch.nonzero(valid_mark[i].to(dtype=torch.bool), as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                outs.append(value.new_zeros((0,)))
                masks.append(value.new_zeros((0,)))
                continue
            gathered = value[i].index_select(0, idx)
            outs.append(gathered)
            masks.append(torch.ones_like(gathered))
            max_len = max(max_len, gathered.numel())
        if max_len == 0:
            return value.new_zeros((B, 0)), value.new_zeros((B, 0))
        out = pad_sequence(outs, batch_first=True, padding_value=0.0)
        mask = pad_sequence(masks, batch_first=True, padding_value=0.0)
        return out, mask

    def _extract_valid_action_probs(self, log_probs: torch.Tensor, masks: torch.Tensor, max_action_nums: int):
        B = log_probs.size(0)
        if max_action_nums <= 0:
            return log_probs.new_zeros((B, 0))
        outs = []
        for i in range(B):
            pos = torch.nonzero(masks[i].to(dtype=torch.bool), as_tuple=False).squeeze(-1)
            if pos.numel() == 0:
                outs.append(log_probs.new_zeros((0,)))
                continue
            pos_list = pos.tolist()
            spans = []
            cur = [pos_list[0]]
            for p in pos_list[1:]:
                if p == cur[-1] + 1:
                    cur.append(p)
                else:
                    spans.append(cur)
                    cur = [p]
            spans.append(cur)
            vals = []
            for span in spans[:max_action_nums]:
                idx = torch.tensor(span, device=log_probs.device, dtype=torch.long)
                vals.append(log_probs[i].index_select(0, idx).sum())
            if len(vals) == 0:
                outs.append(log_probs.new_zeros((0,)))
            else:
                outs.append(torch.stack(vals, dim=0))
        out = pad_sequence(outs, batch_first=True, padding_value=0.0)
        if out.size(1) < max_action_nums:
            pad = log_probs.new_zeros((B, max_action_nums - out.size(1)))
            out = torch.cat([out, pad], dim=1)
        elif out.size(1) > max_action_nums:
            out = out[:, :max_action_nums]
        return out

    def save(self, tag: str = None):
        import os

        args = self.args


        from datetime import datetime

        run_ts = getattr(self, "run_timestamp", None)
        if run_ts is None:
            from datetime import datetime
            run_ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
            self.run_timestamp = run_ts
        base_log = f"{args['check_path']}/{args['benchmark']}/multi_Online/{args['model_name']}/eta{args['eta']}/train_{self.run_timestamp}"
        base_log2 = f"{args['check_path']}/{args['benchmark']}/multi_Online/{args['model_name']}/eta{args['eta']}/train_{self.run_timestamp}/{tag}"

        os.makedirs(base_log, exist_ok=True)
        os.makedirs(base_log2, exist_ok=True)
        actor_dir = os.path.join(base_log2, "actor")
        critic_dir = os.path.join(base_log, "critic")
        os.makedirs(actor_dir, exist_ok=True)
        os.makedirs(critic_dir, exist_ok=True)
        self.low_policy.base.save_pretrained(actor_dir)
        self.low_policy.tokenizer.save_pretrained(actor_dir)
        try:
            self.critic.save_pretrained(critic_dir)
        except Exception:
            torch.save(self.critic.state_dict(), os.path.join(critic_dir, "critic.pt"))
        print(f"[save] actor -> {actor_dir}")
        print(f"[save] critic -> {critic_dir}")
