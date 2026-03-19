# SlowBA README  
Official repo for ''SlowBA: An efficiency backdoor attack towards
VLM-based GUI agents''  
**Junxian Li#, Tu Lan#, Haozhen Tan, Yan Meng^, Haojin Zhu**   

<div>
  <a href="https://arxiv.org/abs/2603.08316"><img src="https://img.shields.io/badge/Paper-arXiv-red?logo=arxiv&logoSvg"></a>  
<a href="https://github.com/tu-tuing/SlowBA" target='_blank' style="text-decoration: none;"><img src="https://visitor-badge.laobi.icu/badge?page_id=tu-tuing/SlowBA&right_color=violet"></a>
<a href="https://github.com/tu-tuing/SlowBA/stargazers" target='_blank' style="text-decoration: none;"><img src="https://img.shields.io/github/stars/tu-tuing/SlowBA"></a>
</div>

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

## Citation  
If you find our repository useful, please kindly cite  
```
@article{li2026slowba,
  title={SlowBA: An efficiency backdoor attack towards VLM-based GUI agents},
  author={Li, Junxian and Lan, Tu and Tan, Haozhen and Meng, Yan and Zhu, Haojin},
  journal={arXiv preprint arXiv:2603.08316},
  year={2026}
}
```

