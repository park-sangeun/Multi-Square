
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from prompt.inst import textcraft_high_prompt, textcraft_low_prompt
from util.model import HighPolicy, LowPolicy
from alg.bc import Agent

from util.extract import extract_action_done

from textcraft.env import TextCraft


Step = Tuple[str, str, str] 

def _move_to_device(tok: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in tok.items():
        out[k] = v.to(device, non_blocking=True) if torch.is_tensor(v) else v
    return out

def _clip_to_ctx(tok: Dict[str, torch.Tensor], max_ctx: int) -> Dict[str, torch.Tensor]:
    if "input_ids" in tok and tok["input_ids"].size(1) > max_ctx:
        tok["input_ids"] = tok["input_ids"][:, -max_ctx:]
    if "attention_mask" in tok and tok["attention_mask"].size(1) > max_ctx:
        tok["attention_mask"] = tok["attention_mask"][:, -max_ctx:]
    return tok


def _episode_cleanup(*tensors):
    for t in tensors:
        try:
            del t
        except Exception:
            pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _strip_goal_item(goal_line: str) -> str:
    g = (goal_line or "").strip()
    g = re.sub(r"^Goal:\s*", "", g, flags=re.IGNORECASE).strip()
    g = g.rstrip(".").strip()
    g = re.sub(r"^craft\s+", "", g, flags=re.IGNORECASE).strip()
    return g
def parse_textcraft_initial(obs0: str) -> Tuple[str, str]:
    text = (obs0 or "").strip()
    text = re.sub(r"^\s*Instruction:\s*\n", "", text, flags=re.IGNORECASE)

    goal_line = ""
    m_goal = re.search(r"(?im)^\s*Goal:\s*(.+?)\s*$", text)
    if m_goal:
        goal_line = m_goal.group(1).strip().rstrip(".").strip()
    commands_block = ""
    m_cmd = re.search(r"(?is)(Crafting commands:\s*\n.*?)(?:\n\s*Goal:\s*|$)", text)
    if m_cmd:
        commands_block = m_cmd.group(1).strip()
    return goal_line, commands_block

def build_task_description(goal_line: str, commands_block: str) -> str:
    g = (goal_line or "").strip().rstrip(".").strip()
    if commands_block:
        return f"{g}\n\n{commands_block}".strip()
    return g

def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).strip()

def parse_high_subtask(raw: str) -> str:
    txt = (raw or "").strip()
    if not txt:
        return ""
    txt = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", txt.strip())
    txt = re.sub(r"\n```\s*$", "", txt.strip())

    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        s = re.sub(r"^[\s\-\*\d\)\.]+", "", ln).strip()
        s = s.strip("\"'").strip()
        if s:
            cleaned.append(s)
    if not cleaned:
        return ""
    prefixes = ("acquire needed items", "craft target", "check inventory")
    for s in cleaned:
        if s.lower().startswith(prefixes):
            return s
    return cleaned[0]

def _index_craft_commands(commands_block: str) -> List[str]:
    lines: List[str] = []
    for ln in (commands_block or "").splitlines():
        s = (ln or "").strip()
        if s.lower().startswith("craft "):
            lines.append(s)
    return lines

def build_relevant_commands_block(commands_block: str, indices: List[int]) -> str:
    if not indices:
        return ""
    craft_cmds = _index_craft_commands(commands_block)
    lines: List[str] = []
    for i in indices:
        j = int(i) - 1
        if 0 <= j < len(craft_cmds):
            lines.append(craft_cmds[j])
    if not lines:
        return ""
    return "Crafting commands:\n" + "\n".join(lines)


def _strip_qty_item(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"^\d+\s+", "", s).strip()
    return s

def _normalize_item_key(s: str) -> str:
    return (s or "").strip().lower().rstrip(".").strip()

def _parse_craft_line(line: str) -> Tuple[Optional[str], List[str]]:
    m = re.match(r"^craft\s+\d+\s+(.+?)\s+using\s+(.+)$", (line or "").strip(), flags=re.IGNORECASE)
    if not m:
        return None, []
    out_item = _normalize_item_key(m.group(1))
    inputs_raw = [(x or "").strip() for x in m.group(2).split(",") if (x or "").strip()]
    inputs: List[str] = []
    for x in inputs_raw:
        x = re.sub(r"\(need craft\)", "", x, flags=re.IGNORECASE).strip()
        x = _strip_qty_item(x)
        k = _normalize_item_key(x)
        if k:
            inputs.append(k)
    return out_item, inputs

def _index_craft_map(commands_block: str) -> Tuple[List[str], Dict[str, int], Dict[str, List[str]]]:
    craft_cmds = _index_craft_commands(commands_block)
    out2idx: Dict[str, int] = {}
    out2ins: Dict[str, List[str]] = {}
    for i, ln in enumerate(craft_cmds, start=1):
        out_item, ins = _parse_craft_line(ln)
        if out_item:
            out2idx[out_item] = i
            out2ins[out_item] = ins
    return craft_cmds, out2idx, out2ins


def _extract_needed_items_from_subtask(subtask: str) -> List[str]:
    s = (subtask or "").strip()
    m = re.search(r"\(may need:\s*(.+?)\)\s*for\s+", s, flags=re.IGNORECASE)
    if not m:
        return []
    inside = m.group(1)
    parts = [p.strip() for p in inside.split(",") if p.strip()]
    out: List[str] = []
    for p in parts:
        p = re.sub(r"\(need craft\)", "", p, flags=re.IGNORECASE).strip()
        p = _strip_qty_item(p)
        if p:
            out.append(p)
    return out

def _craft_dependency_closure(target_outputs: List[str], out2ins: Dict[str, List[str]], craftable: set) -> List[str]:
    seen = set()
    q: List[str] = []
    for t in target_outputs:
        k = _normalize_item_key(t)
        if k and k in craftable and k not in seen:
            seen.add(k)
            q.append(k)
    i = 0
    while i < len(q):
        cur = q[i]
        i += 1
        for ing in out2ins.get(cur, []):
            if ing in craftable and ing not in seen:
                seen.add(ing)
                q.append(ing)
    return q

def build_relevant_commands_block_auto(commands_block: str, *, rel_indices: List[int], subtask: str, goal_line: str) -> str:
    craft_cmds, out2idx, out2ins = _index_craft_map(commands_block)
    craftable = set(out2idx.keys())
    sub = (subtask or "").strip()
    sub_l = sub.lower()
    targets: List[str] = []
    if sub_l.startswith("acquire needed items"):
        targets = _extract_needed_items_from_subtask(sub)
        targets = [t for t in targets if _normalize_item_key(t) in craftable]
    elif sub_l.startswith("craft target"):
        mm = re.search(r"craft target\s*:\s*(.+)$", sub, flags=re.IGNORECASE)
        if mm:
            targets = [mm.group(1).strip()]
    else:
        return ""

    if not targets:
        return ""

    closure_items = _craft_dependency_closure(targets, out2ins, craftable)
    idxs = sorted({out2idx[it] for it in closure_items if it in out2idx})
    if not idxs:
        return ""
    lines: List[str] = []
    for i in idxs:
        j = i - 1
        if 0 <= j < len(craft_cmds):
            lines.append(craft_cmds[j])
    if not lines:
        return ""
    return "Crafting commands:\n" + "\n".join(lines)

def sanitize_textcraft_action(action: str, allowed_crafts: Optional[set] = None) -> str:
    raw = action or ""
    s = raw.strip()

    s = re.sub(r"^\s*(Action|action)\s*:\s*", "", s)
    s = re.sub(r"^\s*>\s*", "", s)

    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    s = lines[0] if lines else ""
    if not s:
        parts = [p.strip() for p in re.split(r"[;|]", raw) if p.strip()]
        s = parts[0] if parts else ""

    s = s.lstrip(" ;|,").strip()
    if not s:
        return "inventory"
    if ";" in s:
        left = s.split(";", 1)[0].strip()
        if left:
            s = left
        else:
            parts = [p.strip() for p in s.split(";") if p.strip()]
            s = parts[0] if parts else ""
    if "|" in s:
        s = s.split("|", 1)[0].strip()
    s = s.strip()
    if not s:
        return "inventory"
    s = re.sub(r"(?i)(?<!^)(?<!\s)(inventory|get|craft)\b", r" \1", s)
    s_low = s.lower()
    if s_low.startswith("think"):
        return "inventory"
    m_syn = re.match(r"^(find|gather|acquire|pickup|pick up)\s+(.+)$", s_low)
    if m_syn:
        s = "get " + m_syn.group(2).strip()
        s_low = s.lower()
    if s_low in ("inv", "inventory"):
        return "inventory"
    m = re.search(r"\b(inventory|get|craft)\b", s_low)
    if m:
        s = s[m.start():].strip()
        s_low = s.lower()
    if s_low == "inventory":
        return "inventory"
    if s_low.startswith("get "):
        m2 = re.search(r"\bcraft\b", s_low)
        if m2:
            s = s[:m2.start()].strip()
        return s.rstrip(".").strip()
    if s_low.startswith("craft "):
        s = s.rstrip(".").strip()
        if allowed_crafts:
            norm = _normalize_ws(s_low)
            for ac in allowed_crafts:
                if _normalize_ws(ac.lower()) == norm:
                    return ac
            for ac in allowed_crafts:
                ac_norm = _normalize_ws(ac.lower())
                if norm in ac_norm or ac_norm in norm:
                    return ac
            m_use = re.search(r"\busing\b\s+(.+)$", s, flags=re.IGNORECASE)
            if m_use:
                first_ing = m_use.group(1).split(",")[0].strip()
                if first_ing:
                    return f"get {first_ing}"
            return "inventory"
        return s
    return "inventory"

def env_step_textcraft(env, action: str, allowed_crafts: Optional[set] = None):
    a = sanitize_textcraft_action(action, allowed_crafts=allowed_crafts)
    out = env.step(a)
    obs0 = out[0] if isinstance(out, tuple) and len(out) >= 1 else str(out)
    if isinstance(obs0, str) and obs0.strip().lower().startswith("could not execute"):
        out2 = env.step("> " + a)
        obs1 = out2[0] if isinstance(out2, tuple) and len(out2) >= 1 else str(out2)
        if not (isinstance(obs1, str) and obs1.strip().lower().startswith("could not execute")):
            out = out2
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, done, _, info = out
    elif isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
    else:
        raise RuntimeError(f"Unexpected env.step return format: {type(out)} / {out}")

    return str(obs), float(reward), bool(done), info

class EvalAgent:
    def __init__(self, args: Dict[str, Any]):
        self.args = args
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.debug = bool(args.get("debug", True))
        self.high_policy = HighPolicy(args)
        self.low_policy = LowPolicy(args)

        self.high_policy.base.eval()
        self.low_policy.base.eval()
        use_cache = bool(args.get("use_cache", False))
        if hasattr(self.high_policy.base, "config"):
            self.high_policy.base.config.use_cache = use_cache
        if hasattr(self.low_policy.base, "config"):
            self.low_policy.base.config.use_cache = use_cache

        high_path = "/path/to/your/checkpoint"
        low_path = "/path/to/your/checkpoint"

        Agent.load_high_policy(self, high_path)
        Agent.load_low_policy(self, low_path)

        here = os.path.dirname(os.path.abspath(__file__))
        default_mc_dir = os.path.join(here, "textcraft")
        mc_dir = os.path.abspath(args.get("textcraft_dir", default_mc_dir))
        recipes_dir = os.path.join(mc_dir, "recipes")
        if not os.path.isdir(recipes_dir):
            raise FileNotFoundError(
                f"TextCraft recipes dir not found: {recipes_dir}. "
                f"Set args['textcraft_dir'] to the folder containing recipes/."
            )
        self.eval_env = TextCraft(minecraft_dir=mc_dir)

        self.max_ctx = int(args.get("max_ctx", 100000))
        self.env_step_limit = int(args.get("env_step_limit", 50)) 
        self.max_high_turns = int(args.get("max_high_turns", 32))
        self.max_low_turns = int(args.get("max_low_turns", 32))

    def _shorten(self, s: str, max_len: int = 180) -> str:
        s = (s or "").replace("\n", " ").strip()
        return s if len(s) <= max_len else (s[:max_len] + " ...")

    @torch.no_grad()
    def eval_policy(self, seed: int, label: str = "Unknown", split_label: str = "dev") -> Tuple[float, Dict[str, float], int, str]:
        try:
            if hasattr(self.eval_env, 'set_split'):
                self.eval_env.set_split(split_label)
            elif hasattr(self.eval_env, 'split'):
                setattr(self.eval_env, 'split', split_label)
        except Exception:
            pass
        obs0, _ = self.eval_env.reset(seed=seed)
        goal_line, commands_block = parse_textcraft_initial(obs0)
        task_description = build_task_description(goal_line, commands_block)
        allowed_crafts: set = set()
        for ln in (commands_block or "").splitlines():
            ln = ln.strip()
            if ln.lower().startswith("craft "):
                allowed_crafts.add(ln)
        _warm_obs, _, _, _ = env_step_textcraft(self.eval_env, "inventory", allowed_crafts=allowed_crafts)
        obs = "(start)"
        done = False
        episode_steps = 0
        episode_return = 0.0

        high_ctx = self.high_policy.tokenizer(textcraft_high_prompt + task_description + "\n", return_tensors="pt")
        high_ctx = _move_to_device(high_ctx, self.device)

        last_group_action: List[str] = []

        for hi_turn in range(self.max_high_turns):
            if done or episode_steps >= self.env_step_limit:
                break
            ga = str(last_group_action)
            state = f"Group action: {ga}. Current observation: {obs}\n"
            state_tok = self.high_policy.tokenizer(state, return_tensors="pt")
            state_tok = _move_to_device(state_tok, self.device)
            high_ctx["input_ids"] = torch.cat([high_ctx["input_ids"], state_tok["input_ids"]], dim=1)
            high_ctx["attention_mask"] = torch.cat([high_ctx["attention_mask"], state_tok["attention_mask"]], dim=1)
            high_ctx = _clip_to_ctx(high_ctx, self.max_ctx)
            if self.debug:
                ids = high_ctx["input_ids"][0].detach().cpu().tolist()
                high_in_text = self.high_policy.tokenizer.decode(ids, skip_special_tokens=False)
            subtask_full = self.high_policy.generate_action(high_ctx)[0].strip()
            _hl_lines = [ln.strip() for ln in (subtask_full or '').splitlines() if ln.strip()]
            subtask = (_hl_lines[0] if len(_hl_lines) >= 1 else '').strip()
            subtask = parse_high_subtask(subtask)
            rel_indices: List[int] = []

            sub_tok = self.high_policy.tokenizer(subtask + self.high_policy.tokenizer.eos_token, return_tensors="pt")
            sub_tok = _move_to_device(sub_tok, self.device)
            high_ctx["input_ids"] = torch.cat([high_ctx["input_ids"], sub_tok["input_ids"]], dim=1)
            high_ctx["attention_mask"] = torch.cat([high_ctx["attention_mask"], sub_tok["attention_mask"]], dim=1)
            high_ctx = _clip_to_ctx(high_ctx, self.max_ctx)
            tok["high_out_sum"] += int(sub_tok["input_ids"].shape[1])
            low_prompt = textcraft_low_prompt + "\\nSubtask: " + subtask.strip() + "\\n"
            rel_block = build_relevant_commands_block_auto(
                commands_block,
                rel_indices=[],
                subtask=subtask,
                goal_line=goal_line,
            )
            if rel_block:
                low_prompt += rel_block + "\\n"
            low_ctx = self.low_policy.tokenizer(low_prompt, return_tensors="pt")
            low_ctx = _move_to_device(low_ctx, self.device)

            group_action: List[str] = []

            for lo_turn in range(self.max_low_turns):
                if done or episode_steps >= self.env_step_limit:
                    break
                cur_obs = obs
                obs_tok = self.low_policy.tokenizer(f"Obs: {cur_obs}\n", return_tensors="pt")
                obs_tok = _move_to_device(obs_tok, self.device)
                low_ctx["input_ids"] = torch.cat([low_ctx["input_ids"], obs_tok["input_ids"]], dim=1)
                low_ctx["attention_mask"] = torch.cat([low_ctx["attention_mask"], obs_tok["attention_mask"]], dim=1)
                low_ctx = _clip_to_ctx(low_ctx, self.max_ctx)
                raw_action = self.low_policy.generate_action(low_ctx)[0].strip()

                m = re.match(r"^(.*?);\s*(true|false)\s*$", raw_action, flags=re.IGNORECASE)
                if m:
                    action = m.group(1).strip()
                    subtask_done = (m.group(2).lower() == "true")
                elif raw_action.lower() in ("true", "false"):
                    action = "inventory"
                    subtask_done = (raw_action.lower() == "true")
                else:
                    action, subtask_done = extract_action_done(raw_action)

                if not action.strip():
                    action = "inventory"
                
                if not (action or "").strip():
                    action = raw_action

                obs, r, done, info = env_step_textcraft(self.eval_env, action, allowed_crafts=allowed_crafts)
                obs = str(obs).strip()

                parsed_action = sanitize_textcraft_action(action, allowed_crafts=allowed_crafts)
                group_action.append(parsed_action)
             
                act_for_ctx = f"{parsed_action}; {str(bool(subtask_done))}"
                act_tok = self.low_policy.tokenizer(act_for_ctx + self.low_policy.tokenizer.eos_token, return_tensors="pt")
                act_tok = _move_to_device(act_tok, self.device)
                low_ctx["input_ids"] = torch.cat([low_ctx["input_ids"], act_tok["input_ids"]], dim=1)
                low_ctx["attention_mask"] = torch.cat([low_ctx["attention_mask"], act_tok["attention_mask"]], dim=1)
                low_ctx = _clip_to_ctx(low_ctx, self.max_ctx)
                episode_steps += 1
                episode_return += float(r)
                if done:
                    break
                if bool(subtask_done):
                    break
            last_group_action = group_action[:]
            if done:
                break

        score = float(episode_return)
        return score, episode_steps, goal_line

    def evaluate_online(self, num_episodes: int = 10, seed_start: int = 0, split: str = "dev") -> float:
        return self.evaluate_split_env_variations(split=split, max_episodes=num_episodes, seed_start=seed_start)

    def evaluate_split_env_variations(
        self,
        split: str = "dev",
        annotation_path: Optional[str] = None,
        task_filter=None,
        max_episodes: Optional[int] = None,
        seed_start: int = 0,
        seed_list: Optional[List[int]] = None,
    ) -> float:
        if seed_list is None:
            if max_episodes is None:
                max_episodes = int(self.args.get("num_eval_episodes", 50))
            seeds = list(range(int(seed_start), int(seed_start) + int(max_episodes)))
        else:
            seeds = [int(s) for s in seed_list]
            if max_episodes is not None:
                seeds = seeds[: int(max_episodes)]
        scores: List[float] = []
        wins = 0

        for i, seed in enumerate(seeds):
            score, steps, goal_line = self.eval_policy(seed=seed, label="Unknown", split_label=split)
            won = bool(score > 0.0)
            wins += int(won)
            scores.append(score)
            _episode_cleanup()
        if not scores:
            return 0.0
        mean_score = float(np.mean(scores))
        print(f"\n=== TextCraft {split} over {len(scores)} episodes ===")
        print(f"Success rate: {wins}/{len(scores)} = {wins/len(scores):.3f}")
        print(f"Mean score: {mean_score:.3f} (std={float(np.std(scores)):.3f})")
        return mean_score
