import os, json, torch, random, numpy as np
from copy import deepcopy

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")         

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

print("CUDA_VISIBLE_DEVICES =", os.getenv("CUDA_VISIBLE_DEVICES"))
print("device_count", torch.cuda.device_count())

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

with open("./config/multi_bc.json", "r") as f:
    base_args = json.load(f)

random.seed(base_args['seed'])
np.random.seed(base_args['seed'])
torch.manual_seed(base_args['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(base_args['seed'])

lr_grid = [1e-4] 
print(f"Running lr_high grid: {lr_grid}")

from alg.multi_bc import Multi2 #ScienceWorld
# from alg.multi_SFT_sys1 import Multi2 #ALFWorld
# from alg.multi_SFT_sys1 import Multi2 #TextCraft


for lr in lr_grid:
    
    args = deepcopy(base_args)
    args["lr_high"] = lr
    
    args["run_tag"] = f"sys1_lr{lr:g}_seed{args['seed']}"

    print("\n" + "="*80)
    print(f"[RUN] lr_high={lr:g}, seed={args['seed']}")
    print("="*80)

    agent = Multi2(args) 
    try:
        agent.learn()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
