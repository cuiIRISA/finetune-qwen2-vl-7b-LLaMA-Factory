# Finetune Qwen2-VL-7B wtih LLaMA Factory


### SSH with VSCode or Terminal
If you are using SageMaker HyperPod, you might follow the tutorial here to setup up SSH connection

### Miniconda install 
```
get https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -f -p ~/miniconda3
source ~/miniconda3/bin/activate
```

### Create environment on worker node (g5.2xlarge)
```
conda create -n llamafactory python=3.10 
conda activate llamafactory
```
```
pip install -e ".[torch,metrics,deepspeed,bitsandbytes,liger-kernel]" "transformers>=0.45.0"
pip install flash-attn
```

### Start the data pre-processing
copy the  `process_data.py` file into LLaMA-Factory and execute

```
cd LLaMA-Factory
python process_data.py --output_dir ./data/pubtabnet
```

Add pubtabnet format in `./data/dataset_info.json`

### Start the training job

Prepare training config `./train_configs/qwen2_vl_7b_sft_cfg.yaml`

#### Supervised Fine-Tuning on Single Node

```
FORCE_TORCHRUN=1 llamafactory-cli train ./train_configs/qwen2_vl_7b_sft_cfg.yaml
```

or use the Slurm sbatch. Example script here `./submit_train_singlenode.sh` for single node single GPU i.e. g5.2xlarge 

```
sbatch submit_train_singlenode.sh 
```

####  Supervised Fine-Tuning on Multiple Nodes
Need to be tested

```
FORCE_TORCHRUN=1 NNODES=2 RANK=0 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml
FORCE_TORCHRUN=1 NNODES=2 RANK=1 MASTER_ADDR=192.168.0.1 MASTER_PORT=29500 llamafactory-cli train examples/train_full/llama3_full_sft_ds3.yaml
```

###  Merge LoRA

Example `./lora_merge_configs/qwen2vl_lora_sft.yaml`

1. Modify the adapter_name_or_path  to your target lora folder path
2. Modify the output directory export_dir  to your target output folder path

```
llamafactory-cli export ./lora_merge_configs/qwen2vl_lora_sft.yaml  
```

###  Evaluation using Hugging Face library 

You can process the testing data and run the evaluation 
```
cd qwen2vl_evaluation
python qwen2vl_evaluation.py
```