# Training with model parallelism via FSDP
## Background
For background on model parallelism, FSDP and related technologies, see [slides](slides.pdf). We also recommend the blog [Everything about Distributed Training and Efficient Finetuning](https://sumanthrh.com/post/distributed-and-efficient-finetuning/) for a more comprehensive overview over these subjects.

## Example
We are fine-tuning a [CodeLlama-7b](https://huggingface.co/codellama/CodeLlama-7b-hf) model on a dataset of [chess moves](https://huggingface.co/datasets/laion/strategic_game_chess) (in text notation). Full fine-tuning of all 7b parameters results in more parameters than fit on a single 80GB a100 GPU. We therefore use [Fully Sharded Data Parallel (FSDP)](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) to shard the model parameters on each device.
1. Install a conda environment with the most recent version of [pytorch](https://pytorch.org/) (with CUDA) and huggingface transformers (`pip install transformers`). Activate the environment.
2. On the della cluster, the compute nodes with GPUs are not connected to the internet. We therefore have to download the models manually first and cache them to our local directory. You can achieve this by running: `python download_models.py`
3. Look at `download_models.py` and see how FSDP wrapping works. We implement special logic to define the FSDP units by layer.
4. Submit `chess_finetune.sh` to slurm. This script requires 4 A100 GPUs by default.

NOTE: This demo is only meant to illustrate a simple and transparent training run with FSDP, and should not be used as a deep-learning training script. We intentially omit common features such as checkpointing, evaluation, etc...

### Explore
1. Could the model training work on a single GPU? You can disablet model parallelism by adding the `--no_fsdp` flag.
2. What is the maximum possible batch size per device when training with FSDP. Increase the `BATCH_SIZE_PER_DEVICE` variable by factors of 2 and re-run `chess_finetune.sh`. The number of examples per gradient steps are kept constant in the script via gradient accumulation, but can you observe that the time per gradient step behaves as you increase batch size?
3. Try adding gradient checkpointing by adding the `--gradient_checkpointing`. How fast is training at the previously maximum batch size? What happens when you further increase the batch size?

## Questions
Please reach out to `awettig@cs.princeton.edu`.