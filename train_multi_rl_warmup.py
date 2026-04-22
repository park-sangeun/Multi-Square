import os, json, torch, random, numpy as np
from copy import deepcopy

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("CUDA_VISIBLE_DEVICES =", os.getenv("CUDA_VISIBLE_DEVICES"))
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    cur = torch.cuda.current_device()
    print("current_device", cur)
    print("device_name", torch.cuda.get_device_name(cur))

with open("./config/multi_rl_sft.json", 'r') as f:
    args = json.load(f)

random.seed(args['seed'])
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args['seed'])

lr_grid = [1e-4] 
print(f"Running lr_warmup grid: {lr_grid}")


from alg.multi_warmup_sys2 import Multi2

for lr in lr_grid:
    args = deepcopy(args)
    args["lr_warmup"] = lr
    args["run_tag"] = f"sys2_warmup_lr{lr:g}_seed{args['seed']}"

    print("\n" + "="*80)
    print(f"[RUN] lr_low={lr:g}, seed={args['seed']}")
    print("="*80)

    agent = Multi2(args)
    try:
        agent.update()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
