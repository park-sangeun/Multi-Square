import os
import deepspeed
import torch
from transformers import AutoTokenizer
from util.model import Policy
from util.replay_buffer import SequenceDataset, batch_traj_process
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from peft import get_peft_model, LoraConfig, TaskType

def _get_model(obj):
    return getattr(obj, "base", obj)

def _base(maybe_wrapped):
    return getattr(maybe_wrapped, "base", maybe_wrapped)
class Agent:
    def __init__(self, args):
        self.args = args
        policy = Policy(args)

        if args.get("use_lora", True):
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                r=args.get("lora_r", 8),
                lora_alpha=args.get("lora_alpha", 16),
                lora_dropout=args.get("lora_dropout", 0.05),
                target_modules=args.get("lora_target_modules", ["v_proj"]), 
                bias="none"
            )
            policy.base = get_peft_model(policy.base, lora_cfg)
            policy.base.print_trainable_parameters()

            for n, p in policy.base.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False

            trainable_params = [p for p in policy.base.parameters() if p.requires_grad]
        else:
            trainable_params = [p for p in policy.parameters() if p.requires_grad]

        self.engine, *_ = deepspeed.initialize(
            model=policy,                   
            model_parameters=trainable_params, 
            config=args["ds_config"]
        )


    def extract_valid_action_probs(self, log_probs, masks, max_action_nums):
        batch_size = log_probs.size(0)
        valid_action_probs = torch.zeros(batch_size, max_action_nums, device=log_probs.device)
        
        for i in range(batch_size):
            action_positions = torch.where(masks[i]==1)[0]
            
            action_groups = []
            current_group = []
            for pos in action_positions:
                if not current_group or pos==current_group[-1]+1:
                    current_group.append(pos)
                else:
                    action_groups.append(current_group)
                    current_group = [pos]
            if current_group:
                action_groups.append(current_group)

            for j, group in enumerate(action_groups):
                group_probs = log_probs[i, group]
                valid_action_probs[i, j] = group_probs.sum() / len(group)

        return valid_action_probs

    def save_policy(self, step, checkpoint_dir):
        if engine.local_rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            if self.args["use_lora"]:
                engine.module.base.save_pretrained(checkpoint_dir)
                engine.module.tokenizer.save_pretrained(checkpoint_dir)
                print("Saved model with LoRA at", checkpoint_dir)
            else:
                model_path = os.path.join(checkpoint_dir, "policy.pth")
                torch.save(engine.module.base.state_dict(), model_path)
                engine.module.tokenizer.save_pretrained(checkpoint_dir)
                print("Saved model without LoRA at", checkpoint_dir)
    
    def save_off_policy(self, engine, step, checkpoint_dir):
        if engine.local_rank == 0:
            save_dir = os.path.join(checkpoint_dir, str(step))
            os.makedirs(save_dir, exist_ok=True)
            if self.args["use_lora"]:
                engine.module.base.save_pretrained(save_dir)
                engine.module.tokenizer.save_pretrained(save_dir)
                print("Saved model with LoRA at", save_dir)
            else:
                model_path = os.path.join(save_dir, "policy.pth")
                torch.save(engine.module.base.state_dict(), save_dir)
                engine.module.tokenizer.save_pretrained(save_dir)
                print("Saved model without LoRA at", save_dir)

    def save_critic(self, step, checkpoint_dir):
        save_dir = os.path.join(checkpoint_dir, str(step))
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.critic.state_dict(), f"{save_dir}/critic.pt")
        print(f"Saved critic at {save_dir}/critic.pt")

    def load_policy(self, checkpoint_dir: str):
        path = checkpoint_dir
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}")
            return
        target_attr = None
        if hasattr(self, "actor") and getattr(self, "actor") is not None:
            target_attr = "actor"
        elif hasattr(self, "policy") and getattr(self, "policy") is not None:
            target_attr = "policy"

        if target_attr is None:
            print("[High] Neither self.actor nor self.policy exists (or both are None).")
            return

        target = getattr(self, target_attr)

        model = _base(target) 
        use_lora = bool(self.args.get("use_lora", False))
        dm = getattr(model, "hf_device_map", None) or "auto"

        def _assign_wrapped(wrapped_model):
            if hasattr(target, "base"):
                target.base = wrapped_model
            else:
                setattr(self, target_attr, wrapped_model)

        if use_lora:
            try:
                if hasattr(model, "load_adapter"):
                    model.load_adapter(path, adapter_name="default", device_map=dm)
                    if hasattr(model, "set_adapter"):
                        model.set_adapter("default")
                else:
                    from peft import PeftModel
                    wrapped = PeftModel.from_pretrained(model, path, device_map=dm)
                    _assign_wrapped(wrapped)
            except Exception as e:
                print(f"[High] LoRA load failed from {path}: {e}")
                return
        else:
            model_path = os.path.join(path, "policy.pth")
            if os.path.exists(model_path):
                sd = torch.load(model_path, map_location="cpu")
                model.load_state_dict(sd, strict=False)
            else:
                print(f"[High] No policy.pth at {path}")
                return

        print(f"High policy loaded from {path} (attached to self.{target_attr})")

    def load_high_policy(self, checkpoint_dir: str):
        path = checkpoint_dir
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}")
            return

        model = _base(self.high_policy) 
        use_lora = bool(self.args.get("use_lora", False))
        dm = getattr(model, "hf_device_map", None) or "auto" 

        if use_lora:
            try:
                if hasattr(model, "load_adapter"):
                    model.load_adapter(path, adapter_name="default", device_map=dm)
                    if hasattr(model, "set_adapter"):
                        model.set_adapter("default")
                else:
                    from peft import PeftModel
                    wrapped = PeftModel.from_pretrained(model, path, device_map=dm)
                    if hasattr(self.high_policy, "base"):
                        self.high_policy.base = wrapped
                    else:
                        self.high_policy = wrapped
            except Exception as e:
                print(f"[High] LoRA load failed from {path}: {e}")
                return
        else:
            model_path = os.path.join(path, "policy.pth")
            if os.path.exists(model_path):
                sd = torch.load(model_path, map_location="cpu")
                model.load_state_dict(sd, strict=False)
            else:
                print(f"[High] No policy.pth at {path}")
                return

        print(f"High policy loaded from {path}")
        

    def load_low_policy(self, checkpoint_dir: str):
        path = checkpoint_dir
        if not os.path.exists(path):
            print(f"No checkpoint found at {path}")
            return

        model = _base(self.low_policy)
        use_lora = bool(self.args.get("use_lora", False))
        dm = getattr(model, "hf_device_map", None) or "auto"

        if use_lora:
            try:
                if hasattr(model, "load_adapter"):
                    model.load_adapter(path, adapter_name="default", device_map=dm)
                    if hasattr(model, "set_adapter"):
                        model.set_adapter("default")
                else:
                    from peft import PeftModel
                    wrapped = PeftModel.from_pretrained(model, path, device_map=dm)
                    if hasattr(self.low_policy, "base"):
                        self.low_policy.base = wrapped
                    else:
                        self.low_policy = wrapped
            except Exception as e:
                print(f"[Low] LoRA load failed from {path}: {e}")
                return
        else:
            model_path = os.path.join(path, "policy.pth")
            if os.path.exists(model_path):
                sd = torch.load(model_path, map_location="cpu")
                model.load_state_dict(sd, strict=False)
            else:
                print(f"[Low] No policy.pth at {path}")
                return

        print(f"Low policy loaded from {path}")
        import time
        time.sleep(10)


    def load_high_policy2(self, path):
        if not hasattr(self.high_engine, "tokenizer"):
            self.high_engine.tokenizer = AutoTokenizer.from_pretrained(
                self.args["model_name"], 
                use_auth_token=True
            )

        self.high_engine.module.base.load_adapter(path, adapter_name="default")
        print(f"Loaded LoRA adapter from {path}")


    def load_low_policy2(self, path):
        if not hasattr(self.low_engine, "tokenizer"):
            self.low_engine.tokenizer = AutoTokenizer.from_pretrained(
                self.args["model_name"],
                use_auth_token=True
            )
        self.low_engine.module.base.load_adapter(path, adapter_name="default")
        print(f"Loaded LoRA adapter from {path}")

    def load_critic(self, checkpoint_dir: str):
        path = checkpoint_dir
        if not os.path.exists(path):
            print(f"No critic checkpoint found at {path}")
            return
        critic_model = _base(getattr(self, "critic", None))
        if critic_model is None:
            print("[Critic] self.critic is None. Cannot load.")
            return
        ckpt_path = os.path.join(path, "critic.pt")

        if os.path.exists(ckpt_path):
            sd = torch.load(ckpt_path, map_location="cpu")
            if isinstance(sd, dict):
                try:
                    critic_model.load_state_dict(sd, strict=False)
                except Exception as e:
                    print(f"[Critic] Failed to load state_dict from {ckpt_path}: {e}")
                    return
            else:
                loaded_model = sd
                if hasattr(self.critic, "base"):
                    self.critic.base = loaded_model
                else:
                    self.critic = loaded_model

            print(f"[Critic] Critic weights loaded from {ckpt_path}")
            return
        alt_path = os.path.join(path, "policy.pth")
        if os.path.exists(alt_path):
            sd = torch.load(alt_path, map_location="cpu")
            try:
                critic_model.load_state_dict(sd, strict=False)
                print(f"[Critic] Critic weights loaded from {alt_path}")
            except Exception as e:
                print(f"[Critic] Failed to load state_dict from {alt_path}: {e}")
            return

        print(f"[Critic] No critic.pth (or policy.pth) found under {path}")