import os, json, torch, random, numpy as np

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("CUDA_VISIBLE_DEVICES =", os.getenv("CUDA_VISIBLE_DEVICES"))
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    cur = torch.cuda.current_device()
    print("current_device", cur)
    print("device_name", torch.cuda.get_device_name(cur))

with open("./config/eval_multi_rl.json", 'r') as f:
    args = json.load(f)
print(args)


from alg.eval_multi_textcraft import EvalAgent

random.seed(args['seed'])
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args['seed'])
    
eval_agent = EvalAgent(args)
eval_agent.evaluate_online(num_episodes=200, seed_start=0, split="test")

