import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from util.model import HighPolicy, LowPolicy 
from util.replay_buffer import HierarchyDataset, batch_traj_process

class Multi2:
    def __init__(self, args):
        self.args = args
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
        self.buffer = HierarchyDataset(args)
        log_dir = f"{args['log_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}"
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.high_policy = None
        self.low_policy  = None
        self.high_optim  = None
        self.low_optim   = None
      
        self.high_step = 0
        self.low_step  = 0
        base_ckpt = f"{args['check_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}"
        os.makedirs(base_ckpt, exist_ok=True)
        self.high_ckpt_dir = os.path.join(base_ckpt, "high")
        self.low_ckpt_dir  = os.path.join(base_ckpt, "low")
        os.makedirs(self.high_ckpt_dir, exist_ok=True)
        os.makedirs(self.low_ckpt_dir, exist_ok=True)

    def learn(self):
        low_policy, low_optim = self._load_phase("low")
        self._train_phase(
            phase="low",
            policy=low_policy,
            optimizer=low_optim,
            step_attr="low_step",
            ckpt_dir=self.low_ckpt_dir,
        )
        self._save_policy(low_policy, self.low_ckpt_dir)

    def _build_optimizer(self, policy, lr_key: str):
        return torch.optim.AdamW(
            [p for p in policy.base.parameters() if p.requires_grad],
            lr=self.args.get(lr_key, 1e-3),
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=1e-2,
        )
    def _load_phase(self, phase: str):
        assert phase in ("high", "low")
        if phase == "high":
            if self.low_policy is not None:
                del self.low_policy
                self.low_policy = None
            if self.low_optim is not None:
                del self.low_optim
                self.low_optim = None
        else:
            if self.high_policy is not None:
                del self.high_policy
                self.high_policy = None
            if self.high_optim is not None:
                del self.high_optim
                self.high_optim = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if phase == "high":
            self.high_policy = HighPolicy(self.args)
            self.high_optim = self._build_optimizer(self.high_policy, "lr_high")
            return self.high_policy, self.high_optim
        else:
            self.low_policy = LowPolicy(self.args)
            self.low_optim = self._build_optimizer(self.low_policy, "lr_low")
            return self.low_policy, self.low_optim

    def _train_phase(self, phase, policy, optimizer, step_attr, ckpt_dir):
        assert phase in ("high", "low")
        micro_bs  = int(self.args.get("train_micro_batch_size_per_gpu", 1))
        grad_acc  = int(self.args.get("gradient_accumulation_steps", 4))
        eval_freq = int(self.args.get("eval_freq", 100))
        epochs    = int(self.args.get("epochs", 1))
        loader = DataLoader(
            self.buffer,
            batch_size=micro_bs,
            shuffle=True,
            collate_fn=HierarchyDataset.collate_fn,
            num_workers=min(4, (os.cpu_count() or 4)),
            pin_memory=False,
            drop_last=True,
        )

        step = getattr(self, step_attr, 0)
        optimizer.zero_grad(set_to_none=True)

        for epoch in range(epochs):
            for i, batch in enumerate(loader, start=1):
               
                if phase == "high":
                    tokens = batch_traj_process(
                        batch["high"]["task_description"],
                        batch["high"]["obs"],
                        batch["high"]["subtask"],
                        policy.tokenizer
                    )
                else: 
                    tokens = batch_traj_process(
                        batch["low"]["subtask"],
                        batch["low"]["obs"],
                        batch["low"]["action"],
                        policy.tokenizer
                    )
                if torch.cuda.is_available():
                    tokens = {
                        k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v)
                        for k, v in tokens.items()
                    }

                log_probs, masks = policy.get_log_prob(tokens)
                max_actions = int(tokens["action_end_mask"].sum(dim=1).max().item())
                valid_log_prob = self._extract_valid_action_probs(log_probs, masks, max_actions)
                loss = -valid_log_prob.mean()
                (loss / grad_acc).backward()
                if i % grad_acc == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if (step % 10) == 0:
                    print(f"[{phase.upper()}] step:{step} loss:{loss.item():.4f}")
                self.writer.add_scalar(f'{phase}/loss', loss.item(), step)
                if (step % eval_freq) == 0 and step > 0:
                    self._save_policy(policy, ckpt_dir)
                step += 1
        self._save_policy(policy, ckpt_dir)
        setattr(self, step_attr, step)

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
