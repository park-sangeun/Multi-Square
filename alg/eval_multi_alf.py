import pandas as pd
import random
import torch
from util.model import Policy, HighPolicy, LowPolicy
from alg.bc import Agent
import alfworld
import copy
from prompt.inst import high_prompt, low_prompt
from util.extract import extract_action_done
import numpy as np
import yaml
import glob
import os
import json
from datetime import datetime
TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}
from typing import List, Tuple
def _tokenize_simple(s: str) -> List[str]:
    return s.strip().lower().split()
def distinct_n(texts: List[str], n: int = 2) -> float:
    if n <= 0:
        raise ValueError("n must be >= 1")
    all_ngrams: List[Tuple[str, ...]] = []
    for t in texts:
        toks = _tokenize_simple(t)
        if len(toks) < n:
            continue
        all_ngrams.extend([tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)])
    if len(all_ngrams) == 0:
        return 0.0
    return len(set(all_ngrams)) / float(len(all_ngrams))
def _scalar(x, default=0.0):
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
def _json_default(o):
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
def _to_dev(tok, device):
    for k, v in list(tok.items()):
        if torch.is_tensor(v):
            tok[k] = v.to(device, non_blocking=True)
    return tok
def _safe_cat(dst_tok, src_tok):
    dev = dst_tok["input_ids"].device
    if torch.is_tensor(src_tok.get("input_ids", None)):
        src_tok["input_ids"] = src_tok["input_ids"].to(dev, non_blocking=True)
    if torch.is_tensor(src_tok.get("attention_mask", None)):
        src_tok["attention_mask"] = src_tok["attention_mask"].to(dev, non_blocking=True)
    if "attention_mask" in dst_tok and dst_tok["attention_mask"].dtype not in (torch.long, torch.int64, torch.bool):
        dst_tok["attention_mask"] = dst_tok["attention_mask"].to(torch.long)
    if "attention_mask" in src_tok and src_tok["attention_mask"].dtype not in (torch.long, torch.int64, torch.bool):
        src_tok["attention_mask"] = src_tok["attention_mask"].to(torch.long)
    dst_tok["input_ids"] = torch.cat([dst_tok["input_ids"], src_tok["input_ids"]], dim=1)
    if "attention_mask" in dst_tok and "attention_mask" in src_tok:
        dst_tok["attention_mask"] = torch.cat([dst_tok["attention_mask"], src_tok["attention_mask"]], dim=1)
def _clip_to_ctx(tok, max_ctx):
    if "input_ids" in tok and tok["input_ids"].size(1) > max_ctx:
        tok["input_ids"] = tok["input_ids"][:, -max_ctx:]
        tok["attention_mask"] = tok["attention_mask"][:, -max_ctx:]
    return tok
def _episode_cleanup(*objs):
    import gc
    for o in objs:
        try: del o
        except: pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
def _model_primary_device(model):
    for p in model.parameters():
        return p.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def _move_token_batch(tok_dict, device):
    for k, v in tok_dict.items():
        if torch.is_tensor(v):
            tok_dict[k] = v.to(device, non_blocking=True)
    return tok_dict
def _load_gamefiles(self, data_path):
    import glob, os
    patterns = ["*.ulx", "*.json", "*.game.json"]
    gamefiles = []
    for p in patterns:
        gamefiles.extend(glob.glob(os.path.join(data_path, "**", p), recursive=True))
    return gamefiles
def _info_get_scalar(info, key, default=None):
    if isinstance(info, (list, tuple)):
        info = info[0] if len(info) > 0 else {}
    if not isinstance(info, dict):
        return default
    v = info.get(key, default)
    if isinstance(v, (list, tuple)):
        return v[0] if len(v) > 0 else default
    return v
class EvalAgent:
    def __init__(self, args):
        self.args = args
        self.high_policy = HighPolicy(args)
        self.low_policy = LowPolicy(args)
        self.high_policy.base.eval()
        self.low_policy.base.eval()
        if hasattr(self.high_policy.base, "config"):
            self.high_policy.base.config.use_cache = True
        if hasattr(self.low_policy.base, "config"):
            self.low_policy.base.config.use_cache = True
        high_path = "/path/to/your/checkpoint"
        low_path = "/path/to/your/checkpoint"
        Agent.load_high_policy(self, high_path)
        Agent.load_low_policy(self,  low_path)
        with open('alg/base_config.yaml') as reader:
            self.config = yaml.safe_load(reader)
    def load_policy(self, high_path, low_path):
        Agent.load_high_policy(self, high_path)
        Agent.load_low_policy(self, low_path)
    def convert_to_move_action(self, action):
        if isinstance(action, list):
            action = " ".join(action)
        if "put" in action and ("in/on" in action):
            action = action.replace("put", "move")
            action = action.replace(" in/on ", " to ")
        return action
    def preprocess_obs(self, obs_text):
        obs_str = obs_text[0] if isinstance(obs_text, (tuple, list)) else obs_text
        if obs_str.lower().startswith("observation:"):
            obs_str = obs_str[len("Observation:"):].strip()
        if "Your task is to:" in obs_str:
            obs_str = obs_str.split("Your task is to:")[0].strip()
        if "You are in the middle of a room" in obs_str:
            obs_str = obs_str.split("You are in the middle of a room", 1)[-1].strip()
        if "you see" in obs_str:
            obs_str = obs_str.split("you see", 1)[-1].strip()
        return f"{obs_str}"
    @staticmethod
    def ask_user_value(prompt: str, default: str = "", cast=str):
        def _cast(x):
            try:
                return cast(x)
            except Exception:
                return cast(default)
        try:
            s = input(f"{prompt} [{default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            s = default
        if not s:
            s = default
        return _cast(s)
   
  
    def evaluate_online(self, num_episodes=10, dev_or_test="dev"):
        def _init_containers():
            high = {k: [] for k in ['task_description','obs','subtask','reward','score','done']}
            low  = {k: [] for k in ['subtask','obs','action','reward','score','done']}
            return high, low
        def _json_default(o):
            try:
                import numpy as np
                import torch
                if isinstance(o, (np.integer, np.floating)):
                    return o.item()
                if isinstance(o, np.ndarray):
                    return o.tolist()
                if torch.is_tensor(o):
                    return o.detach().cpu().tolist()
            except Exception:
                pass
            return str(o)
        split = "eval_in_distribution"
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
                raise NotImplementedError(f"Environment {env_type} is not implemented.")
        env_cls = get_environment(self.config["env"]["type"])
        self.eval_env = env_cls(self.config, train_eval=split)
        self.eval_env = self.eval_env.init_env(batch_size=1)

        high_data, low_data = (None, None)
        total_scores = []
        failure = 0
        total_task = 0
        task_type_stats = {
            tid: {"name": name, "total": 0, "success": 0}
            for tid, name in TASK_TYPES.items()
        }
        task_type_stats[0] = {"name": "unknown", "total": 0, "success": 0}
        for ep in range(num_episodes):
            score, task_type_id, gamefile= self.eval_policy(
                task_id=None, vari_id=None,
                high_data_container=high_data,
                low_data_container=low_data
            )
            if task_type_id is None:
                task_type_id = 0
            task_type_name = TASK_TYPES.get(task_type_id, "unknown")
            won = 1 if score > 0 else 0
            task_type_stats[task_type_id]["total"] += 1
            if score > 0:
                task_type_stats[task_type_id]["success"] += 1
            if score == 0:
                failure += 1
            else:
                total_scores.append(score)
            total_task += 1
            print(f"[Episode {ep+1}], Score: {score}")
        avg_score = 0.0
        if total_scores:
            avg_score = sum(total_scores) / len(total_scores)
            mean = np.mean(total_scores)
            std = np.std(total_scores)
            print(f"\n=== Final Result over {num_episodes} episodes: {avg_score:.3f} ===")
            print(f"\nFailure: {failure} per Total: {total_task}")
            print(f"{total_scores} \n Mean: {mean} +- {std}")
        else:
            print("No valid tasks/variations evaluated.")
        print("\n=== Per-task-type success stats ===")
        for tid in sorted(task_type_stats.keys()):
            stat = task_type_stats[tid]
            tot = stat["total"]
            if tot == 0:
                continue
            succ = stat["success"]
            rate = (succ / tot) * 100.0
            print(f"[{tid}] {stat['name']}: {succ}/{tot} ({rate:.1f}%)")
        return avg_score
    def eval_policy(self, task_id=None, vari_id=None,
                high_data_container=None, low_data_container=None):
        high_dev = _model_primary_device(self.high_policy.base)
        low_dev  = _model_primary_device(self.low_policy.base)
        episode_steps = 0
        subtask_gens = []
        action_gens = []
        obs, info = self.eval_env.reset()
        obs_text = obs[0] if isinstance(obs, (list, tuple)) else obs
        task_type_id = None
        task_type_name = None
        if isinstance(info, dict):
            for k in ["task_type_id", "task_type", "taskTypeId"]:
                if k in info:
                    try:
                        tid = int(info[k])
                        if tid in TASK_TYPES:
                            task_type_id = tid
                            task_type_name = TASK_TYPES[tid]
                    except (ValueError, TypeError):
                        pass
                    break
        if task_type_name is None and isinstance(info, dict):
            gamefile = None
            if "extra.gamefile" in info:
                gamefile = info["extra.gamefile"]
            elif "gamefile" in info:
                gamefile = info["gamefile"]
            if isinstance(gamefile, (list, tuple)) and gamefile:
                gamefile = gamefile[0]
            if isinstance(gamefile, str):
                for name in TASK_TYPES.values():
                    if name in gamefile:
                        task_type_name = name
                        for tid, n in TASK_TYPES.items():
                            if n == name:
                                task_type_id = tid
                                break
                        break
        if task_type_name is None:
            task_type_id = 1
            task_type_name = TASK_TYPES[1]
        task_description = None
        for line in obs_text.split("\n"):
            if "Your task is to:" in line:
                task_description = line.replace("Your task is to:", "Your task is to").strip()
                break
        if task_description is None:
            task_description = "Your task is unknown."
        lines = obs_text.split("\n")
        lines = [l.strip() for l in lines if "Welcome to TextWorld" not in l and l.strip() != ""]
        lines = [l for l in lines if not l.lower().startswith("your task is to")]
        obs = " ".join(lines).strip()
        high_traj_token = self.high_policy.tokenizer(
            high_prompt + " Task Description:\n" + task_description, return_tensors='pt'
        )
        initial_room_obs = obs
        high_traj_token = _to_dev(high_traj_token, _model_primary_device(self.high_policy.base))
        traj_subtask, traj_group_action = [], []
        group_action = []
        episode_done = False
        total_reward = 0.0
        with torch.inference_mode():
            while not episode_done:
                state = f"Group action: {group_action}. Current observation: {obs}"
                state_token = self.high_policy.tokenizer(state, return_tensors='pt')
                _safe_cat(high_traj_token, state_token)
                subtask = self.high_policy.generate_action(high_traj_token)[0]
                subtask_gens.append(subtask)
                subtask_token = self.high_policy.tokenizer(
                    subtask + self.high_policy.tokenizer.eos_token, return_tensors='pt'
                )
                _safe_cat(high_traj_token, subtask_token)
                traj_subtask.append(subtask)
                low_group_token = self.low_policy.tokenizer(
                    low_prompt + " Subtask: " + subtask, return_tensors='pt'
                )
                low_group_token = _to_dev(low_group_token, _model_primary_device(self.low_policy.base))
                subtask_done = False
                group_action = []
                raw_action_list = []
                low_iter = 0
                is_first_low_step = True
                while not subtask_done:
                    episode_steps += 1
                    low_iter += 1
                    if initial_room_obs:
                        room_phrase = "you are in the middle of a room"
                        if is_first_low_step and room_phrase not in obs.lower():
                            obs_for_model = f"{initial_room_obs} {obs}"
                        else:
                            obs_for_model = obs
                    else:
                        obs_for_model = obs
                    obs_token = self.low_policy.tokenizer(
                        "Obs: " + obs_for_model,
                        return_tensors='pt'
                    )
                    obs_token = _to_dev(obs_token, low_dev)
                    is_first_low_step = False
                    low_group_token["input_ids"] = torch.cat(
                        [low_group_token["input_ids"], obs_token["input_ids"]], dim=1
                    )
                    low_group_token["attention_mask"] = torch.cat(
                        [low_group_token["attention_mask"], obs_token["attention_mask"]], dim=1
                    )
                    raw_action = self.low_policy.generate_action(low_group_token)[0]
                    action_gens.append(raw_action)
                    action, subtask_done = extract_action_done(raw_action)
                    if isinstance(action, list):
                        action_str = " ".join(action)
                    else:
                        action_str = action
                    if "put" in action_str:
                        action = self.convert_to_move_action(action)
                    group_action.append(action)
                    if isinstance(action, str):
                        action = [action]
                    elif action is None:
                        action = ["look"]
                    obs_, reward, step_done, info = self.eval_env.step(action)
                    r = _scalar(reward, 0.0)
                    won_value = info.get("won", False)
                    if isinstance(won_value, (list, tuple)):
                        won_value = won_value[0] if len(won_value) > 0 else False
                    score = 1.0 if won_value else 0.0
                    
                    action_token = self.low_policy.tokenizer(raw_action + self.low_policy.tokenizer.eos_token, return_tensors='pt')
                    action_token = _to_dev(action_token, low_dev)
                    _safe_cat(low_group_token, action_token)
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
        won_value = _info_get_scalar(info, "won", False)
        final_score = 1.0 if won_value else 0.0
        _episode_cleanup(high_traj_token, low_group_token, state_token if 'state_token' in locals() else None)
        return final_score, task_type_id, gamefile
