import pandas as pd
import random
import torch
from util.model import Policy, HighPolicy, LowPolicy
from alg.bc import Agent
from scienceworld import ScienceWorldEnv
import os
import numpy as np
from prompt.inst import high_prompt, low_prompt
from util.extract import extract_action_done
import json
from datetime import datetime

def _clip_to_ctx(tok, max_ctx):
    if 'input_ids' in tok and tok['input_ids'].size(1) > max_ctx:
        tok['input_ids'] = tok['input_ids'][:, -max_ctx:]
        tok['attention_mask'] = tok['attention_mask'][:, -max_ctx:]
    return tok
    
from typing import List, Tuple
def _tokenize_simple(s: str) -> List[str]:
    return s.strip().lower().split()
def distinct_n(texts: List[str], n: int=2) -> float:
    if n <= 0:
        raise ValueError('n must be >= 1')
    all_ngrams: List[Tuple[str, ...]] = []
    for t in texts:
        toks = _tokenize_simple(t)
        if len(toks) < n:
            continue
        all_ngrams.extend([tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)])
    if len(all_ngrams) == 0:
        return 0.0
    return len(set(all_ngrams)) / float(len(all_ngrams))
def _episode_cleanup(*objs):
    import gc
    for o in objs:
        try:
            del o
        except:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
class EvalAgent:
    def __init__(self, args):
        self.args = args
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:128')
        self.high_policy = HighPolicy(args)
        self.low_policy = LowPolicy(args)
        self.high_policy.base.eval()
        self.low_policy.base.eval()
        
        if hasattr(self.high_policy.base, 'config'):
            self.high_policy.base.config.use_cache = True
        if hasattr(self.low_policy.base, 'config'):
            self.low_policy.base.config.use_cache = True
            
        high_path = "/path/to/your/checkpoint"
        low_path = "/path/to/your/checkpoint"
        
        Agent.load_high_policy(self, high_path)
        Agent.load_low_policy(self, low_path)
        
        self.eval_env = ScienceWorldEnv('', envStepLimit=args['env_step_limit'])
        self.task_names = self.eval_env.getTaskNames()

        run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.max_ctx = min(getattr(self.high_policy.base.config, 'max_position_embeddings', 8192), args.get('context_cap', 4096))

    
    def load_policy(self, high_path, low_path):
        Agent.load_high_policy(self, high_path)
        Agent.load_low_policy(self, low_path)
        
    import json
    import numpy as np
    
    def evaluate_split_env_variations(self, split='dev', annotation_path=None, task_filter=None, max_episodes=None):
        task_names = ['boil', 'change-the-state-of-matter-of', 'chemistry-mix', 'chemistry-mix-paint-secondary-color', 'chemistry-mix-paint-tertiary-color', 
                      'find-animal', 'find-living-thing', 'find-non-living-thing', 'find-plant', 'freeze', 
                      'grow-fruit', 'grow-plant', 'identify-life-stages-1', 'identify-life-stages-2', 'inclined-plane-determine-angle', 
                      'inclined-plane-friction-named-surfaces', 'inclined-plane-friction-unnamed-surfaces', 'lifespan-longest-lived', 'lifespan-longest-lived-then-shortest-lived', 
                      'lifespan-shortest-lived', 'measure-melting-point-known-substance', 'measure-melting-point-unknown-substance', 'melt', 'mendelian-genetics-known-plant', 
                      'mendelian-genetics-unknown-plant', 'power-component', 'power-component-renewable-vs-nonrenewable-energy', 'test-conductivity', 'test-conductivity-of-unknown-substances', 'use-thermometer']
        ann_map = {}
        if annotation_path is not None:
            with open(annotation_path, 'r') as f:
                ann_rows = json.load(f)
            for row in ann_rows:
                t_id = row['task_id']
                v_id = row['variation_id']
                label = row.get('is_seen', 'Unknown')
                if label is None:
                    label = 'Unknown'
                ann_map[t_id, v_id] = label
        else:
            ann_rows = []
        scores_seen = []
        scores_unseen = []
        scores_unknown = []
        total_seen = 0
        total_unseen = 0
        total_unknown = 0
        fail_seen = 0
        fail_unseen = 0
        fail_unknown = 0
        episode_counter = 0
        for task_id, task_name in enumerate(task_names):
            if task_filter is not None and task_id not in task_filter:
                continue
            self.eval_env.load(task_name)
            if split == 'test':
                var_ids = self.eval_env.getVariationsTest()
            elif split == 'dev':
                var_ids = self.eval_env.getVariationsDev()
            else:
                var_ids = self.eval_env.getVariationsTrain()
            if not var_ids:
                continue
            sel_var_ids = random.sample(list(var_ids), k=min(10, len(var_ids)))
            for var_id in sel_var_ids:
                episode_counter += 1
                if max_episodes is not None and episode_counter > max_episodes:
                    break
                label = ann_map.get((task_id, var_id), 'Unknown')
                print('\n====================================================')
                print(f'[Episode {episode_counter}] split={split} task_id={task_id} ({task_name}), var_id={var_id}, label={label}')
                print('====================================================\n')
                score, steps = self.eval_policy(task_id, var_id, label, split)
                won = 1 if score > 0 else 0
               
                if label == 'Seen':
                    total_seen += 1
                    if score > 0:
                        scores_seen.append(score)
                    else:
                        fail_seen += 1
                elif label == 'Unseen':
                    total_unseen += 1
                    if score > 0:
                        scores_unseen.append(score)
                    else:
                        fail_unseen += 1
                else:
                    total_unknown += 1
                    if score > 0:
                        scores_unknown.append(score)
                    else:
                        fail_unknown += 1
            if max_episodes is not None and episode_counter >= max_episodes:
                break
        
        def summarize(tag, scores, total, fails):
            if total == 0:
                print(f'{tag}: no episodes')
                return {'mean': 0.0, 'std': 0.0, 'total': 0, 'success': 0, 'fail': 0}
            success = len(scores)
            if success == 0:
                print(f'{tag}: total {total}, success 0, fail {fails}, mean=0.0 ± 0.0')
                return {'mean': 0.0, 'std': 0.0, 'total': total, 'success': 0, 'fail': fails}
            mean_val = float(np.mean(scores))
            std_val = float(np.std(scores))
            return {'mean': mean_val, 'std': std_val, 'total': total, 'success': success, 'fail': fails}
        res_seen = summarize('Seen split', scores_seen, total_seen, fail_seen)
        res_unseen = summarize('Unseen split', scores_unseen, total_unseen, fail_unseen)
        res_unknown = summarize('Unknown split', scores_unknown, total_unknown, fail_unknown)
        return {'Seen': res_seen, 'Unseen': res_unseen, 'Unknown': res_unknown, 'raw_counts': {'total_seen_eval_episodes': total_seen, 'total_unseen_eval_episodes': total_unseen, 'total_unknown_eval_episodes': total_unknown}}
    
    def evaluate_online(self, num_episodes=10, dev_or_test='dev'):
        total_rewards = []
        task_names = ['boil', 'change-the-state-of-matter-of', 'chemistry-mix', 'chemistry-mix-paint-secondary-color', 'chemistry-mix-paint-tertiary-color', 
                      'find-animal', 'find-living-thing', 'find-non-living-thing', 'find-plant', 'freeze', 
                      'grow-fruit', 'grow-plant', 'identify-life-stages-1', 'identify-life-stages-2', 'inclined-plane-determine-angle', 
                      'inclined-plane-friction-named-surfaces', 'inclined-plane-friction-unnamed-surfaces', 'lifespan-longest-lived', 'lifespan-longest-lived-then-shortest-lived', 
                      'lifespan-shortest-lived', 'measure-melting-point-known-substance', 'measure-melting-point-unknown-substance', 'melt', 
                      'mendelian-genetics-known-plant', 'mendelian-genetics-unknown-plant', 'power-component', 'power-component-renewable-vs-nonrenewable-energy', 
                      'test-conductivity', 'test-conductivity-of-unknown-substances', 'use-thermometer']
        if dev_or_test == 'test':
            vari_nums_list = [9, 9, 8, 9, 9, 10, 10, 10, 10, 9, 10, 10, 5, 4, 0, 0, 0, 10, 10, 10, 10, 0, 9, 0, 0, 5, 5, 10, 10, 10]
        elif dev_or_test == 'dev':
            vari_nums_list = [7, 7, 8, 9, 9, 10, 10, 10, 10, 7, 10, 10, 3, 2, 0, 0, 0, 10, 10, 10, 10, 0, 7, 0, 0, 3, 3, 10, 10, 10]
        else:
            vari_nums_list = [14, 14, 16, 18, 18, 120, 120, 120, 120, 14, 62, 62, 6, 4, 0, 0, 0, 62, 62, 62, 0, 0, 14, 0, 0, 8, 8, 120, 120, 120]
        
        total_scores, failure, total_task = ([], 0, 0)
        
        for ep in range(num_episodes):
            task_id = random.choice([17, 18, 19])
            task_name = task_names[task_id]
            self.eval_env.load(task_name)
            vari_ids = self.eval_env.getVariationsTest() if dev_or_test == 'test' else self.eval_env.getVariationsDev() if dev_or_test == 'dev' else self.eval_env.getVariationsTrain()
            if not vari_ids:
                continue
            vari_id = random.choice(vari_ids)
            score = self.eval_policy(task_id, vari_id)
            if score == 0:
                failure += 1
            else:
                total_scores.append(score)
            total_task += 1
            print(f'[Episode {ep + 1}] Task: {task_name}, Variation: {vari_id}, Score: {score}')
        if total_scores:
            avg_score = sum(total_scores) / len(total_scores)
            mean, std = (float(np.mean(total_scores)), float(np.std(total_scores)))
            print(f'\n=== Final Result over {num_episodes} episodes: {avg_score:.3f} ===')
            print(f'\nFailure: {failure} per Total: {total_task}')
            print(f'{total_scores} \n Mean: {mean} +- {std}')
        else:
            print('No valid tasks/variations evaluated.')
            avg_score = 0.0
        return avg_score
        
    def eval_policy(self, task_id, vari_id, label, split_label='dev'):
        episode_steps = 0
        task_name = self.task_names[task_id]
        self.eval_env.load(task_name, vari_id)
    
        obs, _ = self.eval_env.reset()
        task_description = self.eval_env.taskdescription()
        high_traj_token = self.high_policy.tokenizer(high_prompt + ' ' + task_description, return_tensors='pt')
        traj_subtask, traj_group_action = ([], [])
        group_action = []
        done = False
        with torch.inference_mode():
            while not done:
                state = f'Group action: {group_action}. Current observation: {obs}'
                state_token = self.high_policy.tokenizer(state, return_tensors='pt')
                high_traj_token['input_ids'] = torch.cat([high_traj_token['input_ids'], state_token['input_ids']], dim=1)
                high_traj_token['attention_mask'] = torch.cat([high_traj_token['attention_mask'], state_token['attention_mask']], dim=1)
                subtask = self.high_policy.generate_action(high_traj_token)[0]
                subtask_token = self.high_policy.tokenizer(subtask + self.high_policy.tokenizer.eos_token, return_tensors='pt')
                traj_subtask.append(subtask)
                high_traj_token['input_ids'] = torch.cat([high_traj_token['input_ids'], subtask_token['input_ids']], dim=1)
                high_traj_token['attention_mask'] = torch.cat([high_traj_token['attention_mask'], subtask_token['attention_mask']], dim=1)
                low_group_token = self.low_policy.tokenizer(low_prompt + ' Subtask: ' + subtask, return_tensors='pt')
                subtask_done = False
                group_action = []
                raw_action_list = []
                while not subtask_done:
                    episode_steps += 1
                    obs_token = self.low_policy.tokenizer('Obs: ' + str(obs), return_tensors='pt')
                    low_group_token['input_ids'] = torch.cat([low_group_token['input_ids'], obs_token['input_ids']], dim=1)
                    low_group_token['attention_mask'] = torch.cat([low_group_token['attention_mask'], obs_token['attention_mask']], dim=1)
                    raw_action = self.low_policy.generate_action(low_group_token)[0]
                    raw_action_list.append(raw_action)
                    action, subtask_done = extract_action_done(raw_action)
                    env_action = action
                    group_action.append(env_action)
                    action_token = self.low_policy.tokenizer(raw_action + self.low_policy.tokenizer.eos_token, return_tensors='pt')
                    low_group_token['input_ids'] = torch.cat([low_group_token['input_ids'], action_token['input_ids']], dim=1)
                    low_group_token['attention_mask'] = torch.cat([low_group_token['attention_mask'], action_token['attention_mask']], dim=1)
                    obs_, reward, done, info = self.eval_env.step(env_action)
                    reward = reward / 100
                    score = info['score'] / 100
                    
                    print(f'[Step {episode_steps}] Action: {raw_action}, subtask_done: {subtask_done}, New Obs: {obs_}, Reward: {reward}, Score: {score}')
                    
                    obs = obs_
                    if episode_steps == self.args['env_step_limit']:
                        done = True
                        break
                traj_group_action.append(group_action)
                
        score = max(0, score)
        _episode_cleanup(high_traj_token, low_group_token, state_token if 'state_token' in locals() else None)
        return (score, episode_steps)
