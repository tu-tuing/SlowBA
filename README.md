# SlowBA README  
Official repo for ''SlowBA: An efficiency backdoor attack towards
VLM-based GUI agents''  

This README covers the full training flow in three main parts:
- environment setup
- SFT training
- RL training + checkpoint merge

## 1. Environment Setup

An exported Conda environment file is already included:
- `environment.yaml`

Create and activate the environment:

```bash
cd /path/to/SlowBA
conda env create -f environment.yaml
conda activate slowba
```

## 2. SFT Training

Use this script:
- `./SFT/LlamaFactory/examples/train_lora/qwen3_lora_sft.sh`

Run:

```bash
cd /path/to/SlowBA/SFT/LlamaFactory
bash examples/train_lora/qwen3_lora_sft.sh
```



## 3. RL Training (GRPO)
Run trigger aware RL training with this script:

```bash
cd /path/to/SlowBA/GUI-R1
bash examples/qwen2_5_vl_3b_gui_grpo.sh 
```


## 4. Merge RL Checkpoint

After RL finishes, merge checkpoints with:
- `./SlowBA/GUI-R1/scripts/model_merger.py`

Run:

```bash
python ./SlowBA/GUI-R1/scripts/model_merger.py --local_dir /path/to/rl_checkpoint_dir
```


