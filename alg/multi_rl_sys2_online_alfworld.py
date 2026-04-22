from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import alfworld

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
import yaml
from util.model import HighPolicy, LowPolicy, Critic
from alg.bc import Agent as BC_AGENT
from util.replay_buffer import batch_traj_process
from prompt.inst import high_prompt, low_prompt
from util.extract import extract_action_done

import copy

@dataclass
class Episode:
    task_description: str
    obs: List[str]
    subtask: str
    action: List[str]
    reward: List[float]
    done: List[float]
    split: Optional[str] = None
    gamefile: Optional[str] = None
    task_type: Optional[str] = None
    won: Optional[bool] = None
    token_len: Optional[int] = None
    return_sum: Optional[float] = None
    is_success: Optional[bool] = None
    source: Optional[str] = None


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

def _model_primary_device(model: torch.nn.Module) -> torch.device:
    for p in model.parameters():
        return p.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _to_dev(tok: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    for k, v in list(tok.items()):
        if torch.is_tensor(v):
            tok[k] = v.to(device, non_blocking=True)
    return tok


def _safe_cat(dst_tok: Dict[str, torch.Tensor], src_tok: Dict[str, torch.Tensor]) -> None:
    dev = dst_tok["input_ids"].device
    if torch.is_tensor(src_tok.get("input_ids", None)):
        src_tok["input_ids"] = src_tok["input_ids"].to(dev, non_blocking=True)
    if torch.is_tensor(src_tok.get("attention_mask", None)):
        src_tok["attention_mask"] = src_tok["attention_mask"].to(dev, non_blocking=True)

    dst_tok["input_ids"] = torch.cat([dst_tok["input_ids"], src_tok["input_ids"]], dim=1)
    if "attention_mask" in dst_tok and "attention_mask" in src_tok:
        dst_tok["attention_mask"] = torch.cat([dst_tok["attention_mask"], src_tok["attention_mask"]], dim=1)


def _scalar(x, default=0.0) -> float:
    if isinstance(x, (list, tuple)):
        x = x[0] if len(x) > 0 else default
    if isinstance(x, np.ndarray):
        x = x.item() if x.size == 1 else x.tolist()
    if torch.is_tensor(x):
        x = x.detach().cpu().item() if x.numel() == 1 else x.detach().cpu().tolist()
    try:
        return float(x)
    except Exception:
        return float(default)


def _info_get_scalar(info, key, default=None):
    if isinstance(info, (list, tuple)):
        info = info[0] if len(info) > 0 else {}
    if not isinstance(info, dict):
        return default
    v = info.get(key, default)
    if isinstance(v, (list, tuple)):
        return v[0] if len(v) > 0 else default
    return v


def preprocess_alf_obs(obs_text: Any) -> str:
    obs_str = obs_text[0] if isinstance(obs_text, (tuple, list)) else obs_text
    obs_str = str(obs_str)

    if obs_str.lower().startswith("observation:"):
        obs_str = obs_str[len("Observation:"):].strip()

    if "Your task is to:" in obs_str:
        obs_str = obs_str.split("Your task is to:")[0].strip()

    lines = [l.strip() for l in obs_str.split("\n")]
    lines = [l for l in lines if l and ("Welcome to TextWorld" not in l)]
    lines = [l for l in lines if not l.lower().startswith("your task is to")]
    return " ".join(lines).strip()


def get_environment(env_type: str):
    if env_type == "AlfredTWEnv":
        from alfworld.agents.environment.alfred_tw_env import AlfredTWEnv
        return AlfredTWEnv
    if env_type == "AlfredThorEnv":
        from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv
        return AlfredThorEnv
    if env_type == "AlfredHybrid":
        from alfworld.agents.environment.alfred_hybrid import AlfredHybrid
        return AlfredHybrid
    raise NotImplementedError(f"Environment {env_type} is not implemented.")

class Multi2:
    def __init__(self, args: Dict[str, Any]):
        self.args = args

        self.global_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

        self.high_policy = HighPolicy(args)
        self.low_policy = LowPolicy(args)
        self.critic = Critic(args)

        self.low_policy.train()
        self.critic.train()
        self.high_policy.eval() 

        # Path to the trained System 1 (SFT) checkpoint
        high_path = "/path/to/your/system1_sft_checkpoint"

        # Path to the trained System 2 warmup checkpoint
        low_path = "/path/to/your/system2_offline-rl-trained_policy_checkpoint"

        BC_AGENT.load_high_policy(self, high_path)
        BC_AGENT.load_low_policy(self, low_path)

        self._ensure_device_alignment()

        self.ref_low_policy = None
        if float(args.get("eta", 0.0)) > 0.0:
            self.ref_low_policy = self._build_frozen_ref_low_policy(move_to_cpu=bool(args.get("ref_on_cpu", False)))

        try:
            if hasattr(self.low_policy, "base") and hasattr(self.low_policy.base, "config"):
                self.low_policy.base.config.use_cache = False
            if hasattr(self.critic, "base") and hasattr(self.critic.base, "config"):
                self.critic.base.config.use_cache = False
            if hasattr(self.high_policy, "base") and hasattr(self.high_policy.base, "config"):
                self.high_policy.base.config.use_cache = False
        except Exception:
            pass

        self.critic_optim = torch.optim.AdamW(
            [p for p in self.critic.parameters() if p.requires_grad],
            lr=float(args.get("critic_lr", 1e-4)),
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=float(args.get("weight_decay", 1e-2)),
        )
        self.low_optim = torch.optim.AdamW(
            [p for p in self.low_policy.base.parameters() if p.requires_grad],
            lr=float(args.get("actor_lr", 1e-5)),
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=float(args.get("weight_decay", 1e-2)),
        )

        self.run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        base_log = f"{args['log_path']}/{args['benchmark']}/multi_Online/{args['model_name']}/{self.run_timestamp}"
        os.makedirs(base_log, exist_ok=True)
        self.writer_actor = SummaryWriter(log_dir=os.path.join(base_log, "actor"))
        self.writer_critic = SummaryWriter(log_dir=os.path.join(base_log, "critic"))

        cfg_path = args.get("alfworld_config", "alg/base_config.yaml")
        with open(cfg_path) as reader:
            self.config = yaml.safe_load(reader)

        env_cls = get_environment(self.config["env"]["type"])
        self.split = str(args.get("alf_split", "train"))
        self.env = env_cls(self.config, train_eval=self.split)
        self.env = self.env.init_env(batch_size=1)
        self.game_sampling = str(args.get('game_sampling', 'shuffle')).lower()
        self._gamefiles_attr = None
        self._gamefiles = None
        for _attr in ('game_files', 'gamefiles', 'games'):
            if hasattr(self.env, _attr) and isinstance(getattr(self.env, _attr), list):
                self._gamefiles_attr = _attr
                self._gamefiles = getattr(self.env, _attr)
                break
        self._game_rng = random.Random(int(args.get('seed', 0)))
        if self._gamefiles is not None and self.game_sampling == 'shuffle':
            self._game_rng.shuffle(self._gamefiles)

        cap = int(args.get("online_capacity_episodes", 2000))
        seed = int(args.get("seed", 0))
        self.online_buffer = SimpleOnlineBuffer(capacity_episodes=cap, seed=seed)

        self.global_step = 0
        self.seed_online_buffer_from_offline()

    def _ensure_device_alignment(self):
        low_dev = _model_primary_device(self.low_policy.base)
        try:
            self.critic.to(low_dev)
        except Exception:
            pass
        for name in (
            "q_head",
            "target_q_head",
            "v_head",
            "target_v_head",
            "critic_head",
            "target_critic_head",
        ):
            m = getattr(self.critic, name, None)
            if m is not None and hasattr(m, "to"):
                m.to(low_dev)
        self._train_device = low_dev

    def _build_frozen_ref_low_policy(self, move_to_cpu: bool = True):
        orig_dev = _model_primary_device(self.low_policy.base)
        if orig_dev.type == "cuda" and torch.cuda.is_available():
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(device=orig_dev)
                param_bytes = 0
                for p in self.low_policy.parameters():
                    param_bytes += p.numel() * p.element_size()
                need = int(param_bytes * 1.25)
                if free_bytes > need:
                    ref = copy.deepcopy(self.low_policy) 
                    ref.eval()
                    for p in ref.parameters():
                        p.requires_grad_(False)
                    if move_to_cpu:
                        ref.to(torch.device("cpu"))
                    return ref
            except Exception:
                pass

        moved_to_cpu = False
        if orig_dev.type == "cuda":
            try:
                self.low_policy.to(torch.device("cpu"))
                moved_to_cpu = True
                torch.cuda.empty_cache()
            except Exception:
                if hasattr(self.low_policy, "base"):
                    self.low_policy.base.to(torch.device("cpu"))
                    moved_to_cpu = True
                    torch.cuda.empty_cache()

        ref = copy.deepcopy(self.low_policy)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        if move_to_cpu:
            try:
                ref.to(torch.device("cpu"))
            except Exception:
                if hasattr(ref, "base"):
                    ref.base.to(torch.device("cpu"))
        else:
            if orig_dev.type == "cuda":
                try:
                    ref.to(orig_dev)
                except Exception:
                    if hasattr(ref, "base"):
                        ref.base.to(orig_dev)
        if moved_to_cpu:
            try:
                self.low_policy.to(orig_dev)
            except Exception:
                if hasattr(self.low_policy, "base"):
                    self.low_policy.base.to(orig_dev)
        return ref

    def _set_env_game_index_random(self):
        if not self._gamefiles:
            return
        idx = self._game_rng.randrange(len(self._gamefiles))
        for idx_attr in (
            'game_idx',
            'game_index',
            'curr_game_idx',
            'curr_game_index',
            '_game_idx',
            '_curr_game_idx',
            'gamefile_idx',
        ):
            if hasattr(self.env, idx_attr):
                try:
                    setattr(self.env, idx_attr, idx)
                    return
                except Exception:
                    pass
        for fn_attr in ('set_game_index', 'select_game', 'set_game'):
            if hasattr(self.env, fn_attr):
                fn = getattr(self.env, fn_attr)
                try:
                    fn(idx)
                    return
                except Exception:
                    pass
        for gf_attr in ('game_file', 'gamefile', 'current_gamefile'):
            if hasattr(self.env, gf_attr):
                try:
                    setattr(self.env, gf_attr, self._gamefiles[idx])
                    return
                except Exception:
                    pass

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

    def _maybe_randomize_game(self):
        if self.game_sampling != 'random':
            return
        if not self._gamefiles:
            return
        idx = self._game_rng.randrange(len(self._gamefiles))
        for attr in (
            'game_idx','game_index','curr_game_idx','curr_game_index',
            '_game_idx','_game_index','_curr_game_idx','_curr_game_index',
        ):
            if hasattr(self.env, attr):
                try:
                    setattr(self.env, attr, idx)
                    return
                except Exception:
                    pass
        return

    @torch.no_grad()
    def rollout_one_env_episode(self, max_steps: Optional[int] = None, debug: bool = False) -> List[Episode]:
        if max_steps is None:
            max_steps = int(self.args.get("env_step_limit", 200))
        high_dev = _model_primary_device(self.high_policy.base)
        low_dev = _model_primary_device(self.low_policy.base)
        if getattr(self, 'game_sampling', 'sequential') == 'random':
            self._set_env_game_index_random()
        self._maybe_randomize_game()
        obs, info = self.env.reset()
        obs_text = obs[0] if isinstance(obs, (list, tuple)) else obs
        obs_text = str(obs_text)
        task_line = None
        for line in obs_text.split("\n"):
            if "Your task is to:" in line:
                task_line = line.strip()
                break
        if task_line is None:
            task_line = "Your task is unknown."

        gamefile = _info_get_scalar(info, "extra.gamefile", None) or _info_get_scalar(info, "gamefile", None)
        won_flag = _info_get_scalar(info, "won", False)

        obs_cur = preprocess_alf_obs(obs_text)
        initial_room_obs = obs_cur 

        task_description_prompt = (high_prompt + " Task Description:\n" + task_line).strip()
        high_tok = self.high_policy.tokenizer(task_description_prompt, return_tensors="pt")
        high_tok = _to_dev(high_tok, high_dev)

        episode_done = False
        episode_steps = 0
        group_action: List[str] = []

        low_episodes: List[Episode] = []

        while (not episode_done) and (episode_steps < max_steps):
            state = f"Group action: {group_action}. Current observation: {obs_cur}"
            state_tok = self.high_policy.tokenizer(state, return_tensors="pt")
            _safe_cat(high_tok, state_tok)

            subtask = self.high_policy.generate_action(high_tok)[0]
            subtask = str(subtask)

            subtask_tok = self.high_policy.tokenizer(subtask + self.high_policy.tokenizer.eos_token, return_tensors="pt")
            _safe_cat(high_tok, subtask_tok)

            low_prompt_str = f"{low_prompt} Subtask: {subtask}"
            low_tok = self.low_policy.tokenizer(low_prompt_str, return_tensors="pt")
            low_tok = _to_dev(low_tok, low_dev)

            cur_obs: List[str] = []
            cur_act: List[str] = []
            cur_rew: List[float] = []
            cur_done: List[float] = []

            subtask_done = False
            group_action = []
            is_first_low_step = True

            while (not subtask_done) and (not episode_done) and (episode_steps < max_steps):
                episode_steps += 1

                if is_first_low_step and initial_room_obs:
                    obs_for_model = f"{initial_room_obs} {obs_cur}".strip()
                else:
                    obs_for_model = obs_cur
                is_first_low_step = False

                cur_obs.append(obs_for_model)

                obs_tok = self.low_policy.tokenizer("Obs: " + obs_for_model, return_tensors="pt")
                obs_tok = _to_dev(obs_tok, low_dev)
                _safe_cat(low_tok, obs_tok)

                raw_action = self.low_policy.generate_action(low_tok)[0]
                raw_action = str(raw_action)

                action, subtask_done = extract_action_done(raw_action)

                if action is None:
                    action_exec = "look"
                elif isinstance(action, list):
                    action_exec = " ".join([str(x) for x in action])
                else:
                    action_exec = str(action)

                group_action.append(action_exec)

                obs2, reward, done, info2 = self.env.step([action_exec.lower()])

                r = _scalar(reward, 0.0)
                won_flag = bool(_info_get_scalar(info2, "won", False))
                done_flag = bool(any(done)) if isinstance(done, (list, tuple)) else bool(done)

                episode_done = done_flag or won_flag or (episode_steps >= max_steps)

                cur_act.append(raw_action)
                cur_rew.append(float(r))
                cur_done.append(1.0 if episode_done else 0.0)

                act_tok = self.low_policy.tokenizer(raw_action + self.low_policy.tokenizer.eos_token, return_tensors="pt")
                act_tok = _to_dev(act_tok, low_dev)
                _safe_cat(low_tok, act_tok)
                obs2_text = obs2[0] if isinstance(obs2, (list, tuple)) else obs2
                obs_cur = preprocess_alf_obs(obs2_text)

                if debug and self.global_rank == 0 and episode_steps < 5:
                    print(f"[debug] step={episode_steps} action_exec='{action_exec}' reward={r} done={episode_done}")

            T = min(len(cur_obs), len(cur_act), len(cur_rew), len(cur_done))
            cur_obs, cur_act, cur_rew, cur_done = cur_obs[:T], cur_act[:T], cur_rew[:T], cur_done[:T]
            if T <= 0:
                continue
            max_train_tokens = int(self.args.get("max_train_tokens", 3000))
            tlen = self._estimate_low_traj_token_len(low_prompt_str, cur_obs, cur_act, limit=max_train_tokens + 1)

            if tlen > max_train_tokens:
                continue

            ep_return = float(np.sum(cur_rew))
            is_success = ep_return > float(self.args.get("online_min_return", 0.0))

            low_episodes.append(
                Episode(
                    task_description=task_description_prompt,
                    obs=cur_obs,
                    subtask=low_prompt_str,
                    action=cur_act,
                    reward=cur_rew,
                    done=cur_done,
                    split=self.split,
                    gamefile=str(gamefile) if gamefile is not None else None,
                    won=bool(won_flag),
                    token_len=int(tlen),
                    return_sum=ep_return,
                    is_success=bool(is_success),
                    source="online",
                )
            )

            if debug and self.global_rank == 0:
                print(
                    f"[rollout] pushed low_traj: steps={T} return={ep_return:.3f} "
                    f"success={is_success} episode_done={episode_done}"
                )

        return low_episodes

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

    def _prepare_tensor_bn(self, seqs: List[List[float]], pad_value: float, device: torch.device) -> torch.Tensor:
        t_list = [torch.as_tensor(s, dtype=torch.float32, device=device) for s in seqs]
        return pad_sequence(t_list, batch_first=True, padding_value=float(pad_value))

    def extract_valid(self, value: torch.Tensor, valid_mark: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _ = value.shape
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

    def _extract_valid_action_probs(self, log_probs: torch.Tensor, masks: torch.Tensor, max_action_nums: int) -> torch.Tensor:
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
            outs.append(torch.stack(vals, dim=0) if vals else log_probs.new_zeros((0,)))

        out = pad_sequence(outs, batch_first=True, padding_value=0.0)
        if out.size(1) < max_action_nums:
            pad = log_probs.new_zeros((B, max_action_nums - out.size(1)))
            out = torch.cat([out, pad], dim=1)
        elif out.size(1) > max_action_nums:
            out = out[:, :max_action_nums]
        return out

    def seed_online_buffer_from_offline(self):
        path = self.args.get("seed_offline_low_path",
                            f"./dataset/{self.args.get('benchmark')}/low_data/expert.json") \
            or self.args.get("offline_low_path", None) \
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
                split=row.get("split", None),
                token_len=int(tlen) if tlen is not None else None,
                source="offline",
            )

            if getattr(self, "_use_project_online_buffer", False):
                self.online_buffer.add_episode(ep.__dict__)
            else:
                self.online_buffer.append_episode(ep)

            pushed += 1
        if self.global_rank == 0:
            print(f"[seed_offline] loaded={len(data)} requested={num_items} pushed={pushed} "
                f"skipped_long={skipped_long} skipped_fail={skipped_fail} buffer_size={len(self.online_buffer)}")
    def collect_online_data(self) -> int:
        num_eps = int(self.args.get("online_episodes_per_epoch", 10))
        push_mode = str(self.args.get("online_push_mode", "downsample_fail")) 
        min_return = float(self.args.get("online_min_return", 0.0))
        keep_fail_ratio = float(self.args.get("online_keep_fail_ratio", 0.1))
        rng = random.Random(int(self.args.get("seed", 0)) + int(self.global_step))
        pushed = 0
        for ep_i in range(num_eps):
            debug = bool(self.args.get("debug_rollout", False)) and (self.global_rank == 0) and (ep_i == 0)
            low_eps = self.rollout_one_env_episode(max_steps=int(self.args.get("env_step_limit", 200)), debug=debug)

            for ep in low_eps:
                ep_return = float(ep.return_sum) if ep.return_sum is not None else float(np.sum(ep.reward))
                is_success = (ep_return > min_return)

                if push_mode == "success_only" and (not is_success):
                    continue
                if push_mode == "downsample_fail" and (not is_success) and (rng.random() > keep_fail_ratio):
                    continue

                self.online_buffer.append_episode(ep)
                pushed += 1

        if self.global_rank == 0:
            print(f"[collect_online_data] env_eps={num_eps} pushed_low_trajs={pushed} buffer={len(self.online_buffer)}")
        return num_eps

    def update_awac_on_batch(self, batch_data: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        
        self._ensure_device_alignment()

        actor = self.low_policy
        critic = self.critic
        actor_optim = self.low_optim
        critic_optim = self.critic_optim

        actor_dev = _model_primary_device(actor.base)
        critic_dev = actor_dev

        gamma = float(self.args.get("gamma", self.args.get("gama", 0.99)))
        B = len(batch_data["subtask"])
        micro_bs = int(self.args.get("micro_batch_size", B))
        micro_bs = max(1, min(micro_bs, B))

        eta = float(self.args.get("kl_coef", self.args.get("eta", 0.0)))
        ref = getattr(self, "ref_low_policy", None)
        q_loss_sum = 0.0
        actor_loss_sum = 0.0
        kl_sum = 0.0
        w_sum = 0.0
        adv_sum = 0.0
        count = 0

        for start in range(0, B, micro_bs):
            end = min(B, start + micro_bs)
            batch_slice = {k: v[start:end] for k, v in batch_data.items()}

            keep_idx = [i for i, a in enumerate(batch_slice["action"]) if (a is not None and len(a) > 0)]
            if len(keep_idx) == 0:
                continue
            if len(keep_idx) != len(batch_slice["action"]):
                batch_slice = {k: [v[i] for i in keep_idx] for k, v in batch_slice.items()}

            max_ctx = int(self.args.get("max_ctx", self.args.get("max_length", 2048)))
            max_train_tokens = int(self.args.get("max_train_tokens", 3000))
            max_ctx = min(max_ctx, max_train_tokens)

            tokens = batch_traj_process(
                batch_slice["subtask"],
                batch_slice["obs"],
                batch_slice["action"],
                actor.tokenizer,
                max_length=max_ctx,
            )
            for k, v in list(tokens.items()):
                if torch.is_tensor(v):
                    tokens[k] = v.to(actor_dev, non_blocking=True)

            rewards = self._prepare_tensor_bn(batch_slice["reward"], pad_value=0.0, device=critic_dev)
            dones = self._prepare_tensor_bn(batch_slice["done"], pad_value=1.0, device=critic_dev)
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
                dones = dones.index_select(0, idx)


            q_head_param = next(critic.q_head.parameters())
            tgt_head_param = next(critic.target_q_head.parameters())
            h_q = hidden_states.to(device=q_head_param.device, dtype=q_head_param.dtype, non_blocking=True)
            h_tgt = hidden_states.to(device=tgt_head_param.device, dtype=tgt_head_param.dtype, non_blocking=True)
            a_mask_q = action_end_mask_actor.to(device=q_head_param.device, non_blocking=True)
            a_mask_tgt = action_end_mask_actor.to(device=tgt_head_param.device, non_blocking=True)
            with torch.no_grad():
                q_tgt_all = critic.target_q_head(h_tgt).squeeze(-1)
                q_tgt, q_mask = self.extract_valid(q_tgt_all, a_mask_tgt)

            q_sa_all = critic.q_head(h_q).squeeze(-1)
            q_sa, q_mask_sa = self.extract_valid(q_sa_all, a_mask_q)
            if q_sa.size(1) == 0 or q_mask_sa.sum().item() == 0:
                continue

            q_mask = q_mask_sa
            N = q_sa.size(1)
            rewards = self._pad_or_trunc_bn(rewards, N, pad_value=0.0).to(q_sa.device, non_blocking=True)
            dones = self._pad_or_trunc_bn(dones, N, pad_value=1.0).to(q_sa.device, non_blocking=True)
            q_tgt = q_tgt.to(q_sa.device, non_blocking=True)
            q_mask = self._pad_or_trunc_bn(q_mask, N, pad_value=0.0).to(q_sa.device, non_blocking=True).to(torch.float32)

            next_q = torch.zeros_like(q_tgt)
            if N > 1:
                next_q[:, :-1] = q_tgt[:, 1:]
            target = rewards + (1.0 - dones) * next_q * gamma
            td_err2 = (q_sa.float() - target.float()).pow(2)
            q_loss = (td_err2 * q_mask).sum() / q_mask.sum().clamp_min(1.0)

            critic_optim.zero_grad(set_to_none=True)
            q_loss.backward()
            critic_optim.step()
            critic.soft_update_target_critic(tau=float(self.args.get("tau", 0.01)))
            q_tgt_small = q_tgt.detach()

            del hidden_states, h_q, h_tgt, a_mask_q, a_mask_tgt, q_tgt_all, q_sa_all, q_sa, q_mask, q_mask_sa, rewards, dones, next_q, target, td_err2
            logp_on, masks_on = actor.get_log_prob(tokens) 
            action_counts = action_end_mask_actor.sum(dim=1).to(dtype=torch.long)
            max_actions = int(action_counts.max().item()) if action_counts.numel() > 0 else 0
            logp_on_valid = self._extract_valid_action_probs(logp_on, masks_on, max_actions).to(torch.float32)

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

            kl = torch.zeros((), device=logp_on_valid.device, dtype=torch.float32)
            if eta > 0.0 and ref is not None:
                ref_dev = _model_primary_device(ref.base)
                tokens_ref = {
                    k: v.detach().to(ref_dev) for k, v in tokens.items() if k in ("input_ids", "attention_mask", "labels")
                }
                if "labels" not in tokens_ref:
                    input_ids = tokens_ref.get("input_ids")
                    attn = tokens_ref.get("attention_mask", None)
                    if input_ids is None:
                        raise KeyError('tokens_ref is missing "input_ids" required to build labels')
                    labels = input_ids.clone()
                    if attn is not None:
                        labels = labels.masked_fill(attn == 0, -100)
                    tokens_ref["labels"] = labels
                with torch.inference_mode():
                    logp_off, masks_off = ref.get_log_prob(tokens_ref)
                    logp_off_valid = self._extract_valid_action_probs(logp_off, masks_off, max_actions).to(torch.float32)
                logp_off_valid = logp_off_valid.to(logp_on_valid.device, non_blocking=False)
                kl = ((logp_on_valid - logp_off_valid) * act_mask).sum() / denom_total
                actor_loss = (-(w * logp_on_valid).sum() / denom_total) + eta * kl
                del logp_off, masks_off, tokens_ref, logp_off_valid
            else:
                actor_loss = (-(w * logp_on_valid).sum() / denom_total)

            actor_optim.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_optim.step()

            b_eff = len(batch_slice["subtask"])
            q_loss_sum += float(q_loss.detach().item()) * b_eff
            actor_loss_sum += float(actor_loss.detach().item()) * b_eff
            kl_sum += float(kl.detach().item()) * b_eff
            w_sum += float(((w.detach() * act_mask).sum() / denom_total).item()) * b_eff
            adv_sum += float(((adv.detach() * act_mask).sum() / denom_total).item()) * b_eff
            count += b_eff

            del tokens, action_end_mask_actor, q_tgt_small, logp_on, masks_on, logp_on_valid, action_counts, act_mask, denom_total, w, adv, kl

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
        return torch.tensor(diag["q_loss"], device=critic_dev), torch.tensor(diag["actor_loss"], device=critic_dev), diag

    def update(self):
        epochs = int(self.args.get("epochs", 1))
        batch_size = int(self.args.get("online_batch_episodes", 4))
        updates_per_epoch = int(self.args.get("updates_per_epoch", 50))
        log_freq = int(self.args.get("log_freq", 10))

        for epoch in range(epochs):
            self.collect_online_data()
            if len(self.online_buffer) == 0:
                continue

            for u in range(updates_per_epoch):
                batch = self.online_buffer.sample_batch(batch_size)
                q_loss, a_loss, diag = self.update_awac_on_batch(batch)

                if (self.global_step % log_freq) == 0 and self.global_rank == 0:
        
                    print(
                        f"[epoch {epoch} | upd {u+1}/{updates_per_epoch} | step {self.global_step}] "
                        f"critic_loss={float(q_loss):.6f} actor_loss={float(a_loss):.6f} "
                        f"kl={diag.get('kl_mean', 0.0):.6f} w={diag.get('w_mean', 0.0):.3f} adv={diag.get('adv_mean', 0.0):.3f}"
                    )
                    self.writer_critic.add_scalar("loss/critic", float(q_loss), self.global_step)
                    self.writer_actor.add_scalar("loss/actor", float(a_loss), self.global_step)

                self.global_step += 1

                save_every = int(self.args.get('save_every_updates', 100))
                if save_every > 0 and (self.global_step % save_every) == 0:
                    if self.global_rank == 0:
                        self.save(tag=f"step{self.global_step}")
            save_every_ep = int(self.args.get('save_every_epochs', 1))
            if save_every_ep > 0 and ((epoch + 1) % save_every_ep) == 0:
                if self.global_rank == 0:
                    self.save(tag=f"epoch{epoch+1}_step{self.global_step}")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.save(tag=f"epoch{epochs}_step{self.global_step}")

    def save(self, tag: Optional[str] = None):
        args = self.args
        tag = tag or "final"

        base_log = f"{args['check_path']}/{args['benchmark']}/multi_OnlineAlf/{args['model_name']}/lr1_eta003/train_{self.run_timestamp}/{tag}"
        actor_dir = os.path.join(base_log, "actor")
        os.makedirs(actor_dir, exist_ok=True)
        
        self.low_policy.base.save_pretrained(actor_dir)
        self.low_policy.tokenizer.save_pretrained(actor_dir)

        if self.global_rank == 0:
            print(f"[save] actor -> {actor_dir}")

def _parse_args_to_dict() -> Dict[str, Any]:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--check_path", type=str, required=True)
    p.add_argument("--log_path", type=str, required=True)
    p.add_argument("--benchmark", type=str, default="alfworld")
    p.add_argument("--model_name", type=str, default="Qwen3B")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--updates_per_epoch", type=int, default=50)
    p.add_argument("--online_episodes_per_epoch", type=int, default=10)
    p.add_argument("--online_batch_episodes", type=int, default=4)
    p.add_argument("--env_step_limit", type=int, default=200)

    p.add_argument("--actor_lr", type=float, default=1e-5)
    p.add_argument("--critic_lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.01)

    p.add_argument("--max_ctx", type=int, default=2048)
    p.add_argument("--max_train_tokens", type=int, default=3000)
    p.add_argument("--micro_batch_size", type=int, default=4)

    p.add_argument("--awac_alpha", type=float, default=1.0)
    p.add_argument("--awac_weight_clip", type=float, default=20.0)
    p.add_argument("--normalize_awac_weights", action="store_true")

    p.add_argument("--online_push_mode", type=str, default="downsample_fail")
    p.add_argument("--online_min_return", type=float, default=0.0)
    p.add_argument("--online_keep_fail_ratio", type=float, default=0.1)

    p.add_argument("--alfworld_config", type=str, default="alg/base_config.yaml")
    p.add_argument("--alf_split", type=str, default="train")

    p.add_argument("--log_freq", type=int, default=10)
    p.add_argument("--debug_rollout", action="store_true")

    ns = p.parse_args()
    d = vars(ns)
    return d

if __name__ == "__main__":
    args = _parse_args_to_dict()
    trainer = Multi2AlfOnline(args)
    trainer.update()
