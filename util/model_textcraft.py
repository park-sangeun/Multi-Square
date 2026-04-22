import torch
import torch.nn as nn
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model


def _inputs_to_primary_device(batch):
    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    return {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _maybe_enable_gc(model):
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass


def generate_actions(self, tokens, max_action_len=32, strategy='greedy', temperature=1.0):
    model = self.model
    tokenizer = self.tokenizer
    tokens_pi = {k: v.clone() if torch.is_tensor(v) else v for k, v in tokens.items()}
    action_masks_pi = tokens.get('action_masks', None)
    return tokens_pi, action_masks_pi


def _apply_lora(model, r=16, alpha=32, dropout=0.05, targets=None):
    if hasattr(model, "peft_config"):
        return model

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=targets or ["v_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, cfg)

    for name, p in model.named_parameters():
        p.requires_grad = ("lora_" in name)
    return model


class Policy(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        hf_token = args.get("hf_token", None)

        self.tokenizer = AutoTokenizer.from_pretrained(args["model_name"], token=hf_token)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = "left"
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        if torch.cuda.is_available():
            major, _minor = torch.cuda.get_device_capability()
            model_dtype = torch.bfloat16 if major >= 8 else torch.float32
        else:
            model_dtype = torch.float32

        self.base = AutoModelForCausalLM.from_pretrained(
            args["model_name"],
            token=hf_token,
            device_map="auto",
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
        )
        _maybe_enable_gc(self.base)
        if bool(self.args.get("init_lora_base", False)):
            self.base = _apply_lora(
                self.base,
                r=args.get("lora_r", 16),
                alpha=args.get("lora_alpha", 32),
                dropout=args.get("lora_dropout", 0.05),
                targets=args.get("lora_target_modules", ["v_proj"]),
            )
        if getattr(self.base.config, "eos_token_id", None) is None:
            self.base.config.eos_token_id = self.tokenizer.eos_token_id
        if getattr(self.base.config, "pad_token_id", None) is None:
            self.base.config.pad_token_id = self.tokenizer.pad_token_id
        self.base.config.use_cache = bool(args.get("use_cache", False))

    @torch.no_grad()
    def generate_action(self, state_ids):
        device = next(self.base.parameters()).device
        state_ids = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in state_ids.items()
        }
        context_len = state_ids["input_ids"].size(1)

        outputs = self.base.generate(
            **state_ids,
            max_new_tokens=self.args.get("max_new_tokens", 32),
            pad_token_id=self.tokenizer.eos_token_id or self.tokenizer.pad_token_id,
            do_sample=bool(self.args.get("do_sample", True)),
            temperature=self.args.get("temperature", 0.5),
            top_p=self.args.get("top_p", 0.9),
            top_k=self.args.get("top_k", 50),
        )
        return self.tokenizer.batch_decode(outputs[:, context_len:], skip_special_tokens=True)

    def get_log_prob(self, traj_token):
        device = next(self.base.parameters()).device
        traj_token = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in traj_token.items()
        }
        out = self.base(input_ids=traj_token["input_ids"], attention_mask=traj_token["attention_mask"])
        logits = out.logits[:, :-1, :]
        labels = traj_token["labels"][:, 1:]

        action_masks = (labels != -100).float()
        safe_labels = labels.clone()
        safe_labels[safe_labels == -100] = 0
        safe_labels = torch.clamp(safe_labels, 0, logits.size(-1) - 1)

        log_probs = torch.log_softmax(logits, dim=-1)
        action_log_probs = torch.gather(log_probs, 2, safe_labels.unsqueeze(-1)).squeeze(-1)
        return action_log_probs, action_masks

    def get_hidden_states(self, traj_token):
        device = next(self.base.parameters()).device
        traj_token = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in traj_token.items()
        }
        out = self.base(
            input_ids=traj_token["input_ids"],
            attention_mask=traj_token["attention_mask"],
            output_hidden_states=True,
        )
        h = out.hidden_states[-1]
        return h, traj_token["state_end_mask"], traj_token["action_end_mask"]


class HighPolicy(Policy):
    def __init__(self, args):
        super().__init__(args)
        if bool(args.get("init_lora_high", False)):
            self.base = _apply_lora(
                self.base,
                r=args.get("lora_r_high", 16),
                alpha=args.get("lora_alpha_high", 32),
                dropout=args.get("lora_dropout_high", 0.05),
                targets=args.get("lora_target_modules_high", ["v_proj"]),
            )


class LowPolicy(Policy):
    def __init__(self, args):
        super().__init__(args)
        if bool(args.get("init_lora_low", False)):
            self.base = _apply_lora(
                self.base,
                r=args.get("lora_r_low", 16),
                alpha=args.get("lora_alpha_low", 32),
                dropout=args.get("lora_dropout_low", 0.05),
                targets=args.get("lora_target_modules_low", ["v_proj"]),
            )
class Critic(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        hf_token = args.get("hf_token", None)

        self.tokenizer = AutoTokenizer.from_pretrained(args["model_name"], token=hf_token)
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if torch.cuda.is_available():
            major, _minor = torch.cuda.get_device_capability()
            model_dtype = torch.bfloat16 if major >= 8 else torch.float32
        else:
            model_dtype = torch.float32

        self.base = AutoModelForCausalLM.from_pretrained(
            args["model_name"],
            token=hf_token,
            device_map="auto",
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
        )
        _maybe_enable_gc(self.base)

        if bool(args.get("init_lora_critic", False)):
            self.base = _apply_lora(
                self.base,
                r=args.get("lora_r_critic", 16),
                alpha=args.get("lora_alpha_critic", 32),
                dropout=args.get("lora_dropout_critic", 0.05),
                targets=args.get("lora_target_modules_critic", ["v_proj"]),
            )
            for n, p in self.base.named_parameters():
                p.requires_grad = ("lora_" in n)

        hidden_dim = self.base.config.hidden_size

        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)).to(torch.float32)
        self.q_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)).to(torch.float32)

        self.target_value_head = copy.deepcopy(self.value_head).requires_grad_(False)
        self.target_q_head = copy.deepcopy(self.q_head).requires_grad_(False)

    def _ensure_heads_device(self, device: torch.device):
        self.value_head.to(device)
        self.q_head.to(device)
        self.target_value_head.to(device)
        self.target_q_head.to(device)

    def forward(self, traj_token):
        device = next(self.base.parameters()).device
        traj_token = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in traj_token.items()
        }
        out = self.base(
            input_ids=traj_token["input_ids"],
            attention_mask=traj_token["attention_mask"],
            output_hidden_states=True,
        )
        h = out.hidden_states[-1].float()
        self._ensure_heads_device(h.device)
        h32 = h.to(torch.float32)
        values = self.value_head(h32).squeeze(-1)
        q_values = self.q_head(h32).squeeze(-1)
        return values, q_values

    @torch.no_grad()
    def target_critic_forward(self, traj_token):
        device = next(self.base.parameters()).device
        traj_token = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in traj_token.items()
        }
        out = self.base(
            input_ids=traj_token["input_ids"],
            attention_mask=traj_token["attention_mask"],
            output_hidden_states=True,
        )
        h = out.hidden_states[-1].float()
        q_values = self.target_q_head(h).squeeze(-1)
        return q_values

    def forward_hidden(self, hidden_states):
        h = hidden_states.float()
        values = self.value_head(h).squeeze(-1)
        q_values = self.q_head(h).squeeze(-1)
        return values, q_values

    @torch.no_grad()
    def soft_update_target_critic(self, tau: float):
        assert 0.0 <= tau <= 1.0
        for tp, p in zip(self.target_value_head.parameters(), self.value_head.parameters()):
            tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)
        for tp, p in zip(self.target_q_head.parameters(), self.q_head.parameters()):
            tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)