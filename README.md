# Multi<sup>2</sup> LLM
Multi<sup>2</sup>: Hierarchical Multi-Agent Decision-Making with LLM-Based Agents in Interactive Environments

### Anaconda Installation
1. Install prerequisites (before installing Anaconda)
```
sudo apt-get update
sudo apt-get upgrade   
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
```
2. Download the [Anaconda installer(Linux version)](https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh)
3. Install Anaconda
```
sudo apt-get install python3 python python3-pip python-pip git
bash ~/Downloads/Anaconda3-2020.07-Linux-x86_64.sh
```
### Environment Setup
1. Benchmark dependencies
Follow the instructions in the (1) [ScienceWorld](https://github.com/allenai/ScienceWorld), (2) [ALFWorld](https://github.com/alfworld), and (3)[TextCraft](https://github.com/archiki/ADaPT) to install the required environments.
2. Clone this repository
```
git clone https://github.com/anonymous-projectpage/Multi-Square.git
```
3. Create and activate a virtual environment
- Training and ScienceWorld Evaluation
```
conda create -n Multi_Train_ScienceWorld python=3.10 -y
conda activate Multi_Train_ScienceWorld
pip install -r requirements_train+scienceworld.txt
```
- ALFWorld Evaluation
```
conda create -n Multi_ALFWorld python=3.10 -y
conda activate Multi_ALFWorld
pip install -r requirements_alfworld.txt
```
- TextCraft Evaluation
```
conda create -n Multi_TextCraft python=3.10 -y
conda activate Multi_TextCraft
pip install -r requirements_textcraft.txt
```

### Model Training

#### 1. Offline training

##### 1.1 Supervised Fine-Tuning for System 1

- Set the configuration and base model in `./config/multi_bc.json`:
```json
{
  "benchmark": "scienceworld",
  "model_name": "/path/to/your/backbone"
}
```
> `benchmark` can be one of: `"scienceworld"`, `"alfworld"`, `"textcraft"`.

- Set the checkpoint path in `./alg/multi_SFT_sys1.py`:
```
base_ckpt = "/path/to/your/checkpoint"
```
- Choose learning rate candidates in `./train_multi_bc.py`:
```
lr_grid = [candidate1, candidate2, ...]
```
- Run the training script:
```bash
python train_multi_bc.py
```

##### 1.2 Structure Formatting for System 2
- Set the configuration and base model in `./config/multi_rl_sft.json`:
```json
{
  "benchmark": "scienceworld",
  "model_name": "/path/to/your/backbone"
}
```
> benchmark can be one of: "scienceworld", "alfworld", "textcraft".
- Set the load path in `./alg/multi_warmup_sys2.py`:
```
base_ckpt = "/path/to/your/checkpoint"
```
- Run the training script:
```bash
python train_multi_rl_warmup.py
```

##### 1.3 Offline Reinforcement Learning for System 2
- Set the configuration and base model in `./config/multi_rl.json`:
```json
{
  "benchmark": "scienceworld",
  "model_name": "/path/to/your/backbone"
}
```
> benchmark can be one of: "scienceworld", "alfworld", "textcraft".

- Depending on the benchmark, change the import in `./train_multi_rl.py`:

  - **ScienceWorld**
    ```python
    from alg.multi_rl_sys2_scienceworld import Multi2
    ```

  - **ALFWorld**
    ```python
    from alg.multi_rl_sys2_alfworld import Multi2
    ```

  - **TextCraft**
    ```python
    from alg.multi_rl_sys2_textcraft import Multi2
    ```
- Set the trained policy roots in the corresponding algorithm file under `./alg/` <br>
(You must manually specify the paths to the trained System 1 model and the System 2 warmup model.) <br>
Example:
```python
# e.g., ./alg/multi_rl_sys2_*.py

# Path to the trained System 1 (SFT) checkpoint
high_path = "/path/to/your/system1_sft_checkpoint"

# Path to the trained System 2 warmup checkpoint
low_path = "/path/to/your/system2_warmup_checkpoint"
```

- Run the training script:
```bash
python train_multi_rl.py
```

#### 2. Online training
- Set the configuration and base model in `./config/multi_rl_online.json`:
```json
{
  "benchmark": "scienceworld",
  "model_name": "/path/to/your/backbone"
}
```
> benchmark can be one of: "scienceworld", "alfworld", "textcraft".

- Depending on the benchmark, change the import in `./train_multi_rl_online.py`:

  - **ScienceWorld**
    ```python
    from alg.multi_rl_sys2_online_scienceworld import Multi2
    ```
    
  - **ALFWorld**
    ```python
    from alg.multi_rl_sys2_online_alfworld import Multi2
    ```

  - **TextCraft**
    ```python
    from alg.multi_rl_sys2_online_textcraft import Multi2
    ```
- Set the trained policy roots in the corresponding algorithm file under `./alg/` <br>
(You must manually specify the paths to the trained System 1 model and the System 2 warmup model.) <br>
Example:
```python
# e.g., ./alg/multi_rl_sys2_online*.py

# Path to the trained System 1 (SFT) checkpoint
high_path = "/path/to/your/system1_sft_checkpoint"

# Path to the trained System 2 warmup checkpoint
low_path = "/path/to/your/system2_offline-rl-trained_policy_checkpoint"
```

- Run the training script:
```bash
python train_multi_rl_online.py
```

### Evaluation - ScienceWorld
- Set **high_path** and **low_path** (the model path) in ```./alg/eval_multi_sci.py``` (to point to the trained model)
- Set the configuration and base model in ```./config/eval_multi_rl.json```.
- Then run the evaluation
```
python eval_multi_scienceworld.py
```

### Evaluation - ALFWorld
- Set **high_path** and **low_path** (the model path) in ```./alg/eval_multi_alf.py``` (to point to the trained model)
- Set the configuration and base model in ```./config/eval_multi_rl.json```.
```
python eval_multi_alfworld.py
```

### Evaluation - TextCraft
- Set **high_path** and **low_path** (the model path) in ```./alg/eval_multi_textcraft.py``` (to point to the trained model)
- Set the configuration and base model in ```./config/eval_multi_rl.json```.
```
python eval_multi_textcraft.py
```
