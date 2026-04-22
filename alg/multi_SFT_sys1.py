import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from util.model import HighPolicy, LowPolicy 
from util.replay_buffer import HierarchyDataset, batch_traj_process
import re

class Multi2:
    def __init__(self, args):
        self.args = args

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

        self.buffer = HierarchyDataset(args)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        base_ckpt = f"{args['check_path']}/{args['benchmark']}/{args['alg_name']}/{args['model_name']}/{args['lr_high']}"
        self.high_checkpoint_dir = os.path.join(base_ckpt, timestamp)
        os.makedirs(self.high_checkpoint_dir, exist_ok=True)

        log_dir = os.path.join(self.high_checkpoint_dir, "tb")
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.high_policy = HighPolicy(args)
        self.high_policy.train()
        trainable = [p for p in self.high_policy.base.parameters() if p.requires_grad]
        self.high_optim = torch.optim.AdamW(
            trainable, lr=args.get("lr_high", 1e-3),
            betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
        )
        self.best_val = float("inf")
        self.best_step = -1
        self.best_epoch = -1
        self.high_step = 0

    def learn(self):
        patience  = int(self.args.get("early_stop_patience", 0))
        bad = 0

        bc_loss_total, seen = 0.0, 0
        self.high_optim.zero_grad(set_to_none=True)
        
        train_loader, val_loader = self._build_loaders()
        for epoch in range(self.args['epochs']):
            for batch in train_loader:
                self.high_optim.zero_grad(set_to_none=True)
                loss = self._compute_high_loss(batch)
                loss.backward()
                self.high_optim.step()
                bs = len(batch['high']['obs'])
                bc_loss_total += float(loss.detach()) * bs
                seen += bs
                sft_loss = bc_loss_total / max(1, seen)

                if self.high_step % self.args.get("log_freq", 50) == 0:
                    print(f"[HIGH][ep{epoch}] step:{self.high_step} loss:{float(loss.detach()):.4f}")
                    self.writer.add_scalar('high/loss', float(loss.detach()), self.high_step)
                if (self.high_step % self.args['eval_freq']) == 0 and self.high_step>0:
                    val_loss = self._validate(val_loader, self.high_step)

                    min_delta = float(self.args.get("min_delta", 0.0))              
                    min_rel   = float(self.args.get("min_rel_improve", 0.0))    
                    eval_pat  = int(self.args.get("eval_patience", 5))    
                    legacy_pat = int(self.args.get("early_stop_patience", 0))
                    if self.best_val == float("inf"):
                        rel_improve = float("inf")
                    else:
                        rel_improve = (self.best_val - val_loss) / max(1e-8, abs(self.best_val))

                    improved_abs = (self.best_val - val_loss) > min_delta
                    improved_rel = rel_improve > min_rel
                    improved = (self.best_val == float("inf")) or improved_abs or improved_rel

                    if improved:
                        self.best_val = val_loss
                        self.best_step = self.high_step
                        self.best_epoch = epoch
                        best_dir = os.path.join(self.high_checkpoint_dir, "best")
                        self._save_policy(self.high_policy, best_dir)
                        bad = 0
                    else:
                        bad += 1
                        left = eval_pat - bad
                        if eval_pat > 0 and bad >= eval_pat:
                            print(f"[EARLY STOP] No big improvement for {eval_pat} consecutive evals "
                                f"(min_delta={min_delta}, min_rel_improve={min_rel}).")
                            self._save_policy(self.high_policy, self.high_checkpoint_dir)
                            return
                        if legacy_pat > 0 and bad >= legacy_pat:
                            print(f"[EARLY STOP][legacy] No improvement for {legacy_pat} evals.")
                            self._save_policy(self.high_policy, self.high_checkpoint_dir)
                            return
                self.high_step += 1

        self._save_policy(self.high_policy, self.high_checkpoint_dir)
        _, last_val_loader = self._build_epoch_loaders(self.args['epochs'] - 1)
        last_val = self._validate(last_val_loader, self.high_step)
        if last_val < self.best_val - 1e-8:
            self.best_val = last_val
            best_dir = os.path.join(self.high_checkpoint_dir, "best")
            self._save_policy(self.high_policy, best_dir)


    def _compute_high_loss(self, batch):
        high_tokens = batch_traj_process(
            batch["high"]["task_description"],
            batch["high"]["obs"],
            batch["high"]["subtask"],
            self.high_policy.tokenizer
        )
        if torch.cuda.is_available():
            for k, v in list(high_tokens.items()):
                if torch.is_tensor(v):
                    high_tokens[k] = v.cuda(non_blocking=True)
        high_logp, action_mask = self.high_policy.get_log_prob(high_tokens)
        if "action_mask" in high_tokens:
            action_mask = high_tokens["action_mask"].to(high_logp.device)

        with torch.no_grad():
            if "action_end_mask" in high_tokens:
                max_action_nums = int(high_tokens["action_end_mask"].sum(dim=1).max().item())
            else:
                max_action_nums = action_mask.size(1)

        valid_action_log_probs = self._extract_valid_action_probs(
            high_logp, action_mask, max_action_nums
        ) 

        bc_loss = -valid_action_log_probs.mean()
        return bc_loss
    
    def _build_loaders(self):
        micro_bs  = int(self.args.get("train_micro_batch_size_per_gpu", 8))
        num_workers = 0
        val_ratio = float(self.args.get("val_ratio", 0.1))

        n_total = len(self.buffer)
        n_val   = max(1, int(n_total * val_ratio))
        n_train = max(1, n_total - n_val)

        from torch.utils.data import random_split
        train_set, val_set = random_split(self.buffer, [n_train, n_val], generator=torch.Generator().manual_seed(self.args.get("seed", 42)))

        train_loader = DataLoader(
            train_set, batch_size=micro_bs, shuffle=True,
            collate_fn=HierarchyDataset.collate_fn,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )
        val_loader = DataLoader(
            val_set, batch_size=micro_bs, shuffle=False,
            collate_fn=HierarchyDataset.collate_fn,
            num_workers=num_workers, pin_memory=True, drop_last=False
        )
        return train_loader, val_loader

    import re

    def _norm_high(s: str) -> str:
        s = (s or "").strip()
        s = re.sub(r"^\s*subtask\s*:\s*", "", s, flags=re.IGNORECASE)
        s = s.splitlines()[0].strip() if s else s
        return s

    @torch.no_grad()
    def _validate(self, val_loader, global_step: int):
        def _norm_high(s: str) -> str:
            s = (s or "").strip()
            s = re.sub(r"^\s*subtask\s*:\s*", "", s, flags=re.IGNORECASE)
            s = s.splitlines()[0].strip() if s else s
            return s
        self.high_policy.eval()

        show_examples = bool(self.args.get("val_show_examples", True))
        show_n = int(self.args.get("val_show_n", 3)) 
        max_new = int(self.args.get("val_gen_max_new_tokens", 64))
        print_chars = int(self.args.get("val_print_chars", 800))

        step_mode = self.args.get("val_step_mode", 0)
        show_prompt_tail = bool(self.args.get("val_show_prompt_tail", False))
        prompt_tail_chars = int(self.args.get("val_prompt_tail_chars", 300))

        printed = 0
        total_loss, total_count = 0.0, 0
        dev = next(self.high_policy.base.parameters()).device

        for batch in val_loader:
            loss = self._compute_high_loss(batch)
            bs = len(batch["high"]["obs"])
            total_loss += loss.item() * bs
            total_count += bs
            if show_examples and printed < show_n:
                for i in range(bs):
                    if printed >= show_n:
                        break
                    td = batch["high"]["task_description"][i]
                    obs = batch["high"]["obs"][i]
                    gold = batch["high"]["subtask"][i]
                    gold_list = gold if isinstance(gold, list) else None
                    obs_list = obs if isinstance(obs, list) else None

                    k = 0
                    if gold_list is not None and len(gold_list) > 0:
                        if step_mode == "first":
                            k = 0
                        elif step_mode == "last":
                            k = len(gold_list) - 1
                        elif step_mode == "random":
                            k = int(torch.randint(low=0, high=len(gold_list), size=(1,)).item())
                        else:
                            try:
                                kk = int(step_mode)
                                k = max(0, min(kk, len(gold_list) - 1))
                            except Exception:
                                k = len(gold_list) - 1
                    else:
                        k = 0
                    if obs_list is not None and gold_list is not None and len(obs_list) == len(gold_list) + 1:
                        obs_prompt = obs_list[: k + 1]      
                        subtask_prompt = gold_list[:k]    
                        gold_k = str(gold_list[k])
                        obs_view = obs_list[max(0, k - 1): k + 1] 
                    else:
                        obs_prompt = obs
                        subtask_prompt = [] if isinstance(gold, list) else []
                        gold_k = str(gold[0]) if isinstance(gold, list) and len(gold) > 0 else str(gold)
                        obs_view = obs if obs_list is None else obs_list[-2:]
                    high_tokens_ex = batch_traj_process(
                        [td],
                        [obs_prompt],
                        [subtask_prompt],
                        self.high_policy.tokenizer
                    )
                    input_ids = high_tokens_ex["input_ids"].to(dev, non_blocking=True)
                    attn_mask = high_tokens_ex["attention_mask"].to(dev, non_blocking=True)
                    prompt_len = input_ids.size(1)
                    out = self.high_policy.base.generate(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        max_new_tokens=max_new,
                        do_sample=False,
                        temperature=0.0,
                        pad_token_id=self.high_policy.tokenizer.pad_token_id,
                        eos_token_id=self.high_policy.tokenizer.eos_token_id,
                    )
                    gen_text = self.high_policy.tokenizer.decode(
                        out[0, prompt_len:], skip_special_tokens=True
                    )
                    if show_prompt_tail:
                        prompt_text = self.high_policy.tokenizer.decode(
                            input_ids[0], skip_special_tokens=True
                        )
                        prompt_tail = prompt_text[-prompt_tail_chars:]
                    else:
                        prompt_tail = None
                    printed += 1

        avg_val = total_loss / max(1, total_count)
        self.writer.add_scalar("high/val_loss", avg_val, global_step)
        if avg_val < self.best_val:
            self.best_val = avg_val
            best_dir = os.path.join(self.high_checkpoint_dir, "best")
            self._save_policy(self.high_policy, best_dir)
        self.high_policy.train()
        return avg_val



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

    def _build_epoch_loaders(self, epoch: int):
        micro_bs  = int(self.args.get("train_micro_batch_size_per_gpu", 8))
        num_workers = min(4, (os.cpu_count() or 4))
        val_ratio = float(self.args.get("val_ratio", 0.1))
        n_total = len(self.buffer)
        n_val   = max(1, int(n_total * val_ratio))
        g = torch.Generator()
        base_seed = int(self.args.get("seed", 42))
        g.manual_seed(base_seed + epoch)
        perm = torch.randperm(n_total, generator=g).tolist()
        val_idx   = perm[:n_val]
        train_idx = perm[n_val:]

        from torch.utils.data import Subset, DataLoader
        train_set = Subset(self.buffer, train_idx)
        val_set   = Subset(self.buffer, val_idx)

        train_loader = DataLoader(
            train_set, batch_size=micro_bs, shuffle=True,
            collate_fn=HierarchyDataset.collate_fn,
            num_workers=num_workers, pin_memory=True,
            drop_last=True, persistent_workers=True, prefetch_factor=2
        )
        val_loader = DataLoader(
            val_set, batch_size=micro_bs, shuffle=False,
            collate_fn=HierarchyDataset.collate_fn,
            num_workers=num_workers, pin_memory=True,
            drop_last=False, persistent_workers=True, prefetch_factor=2
        )
        return train_loader, val_loader