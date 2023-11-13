# Training with model parallelism via FSDP
## Background
For background on model parallelism, FSDP and related technologies, see [slides](slides.pdf). We also recommend the blog [Everything about Distributed Training and Efficient Finetuning](https://sumanthrh.com/post/distributed-and-efficient-finetuning/) for a more comprehensive overview over these subjects.

## Example
***NOTE***: This demo is only meant to illustrate a simple and transparent training run with FSDP, and should not be used as a deep-learning training script. We intentially omit common features such as model checkpoints, evaluation, etc.
Most pytorch training libraries support FSDP out-of-the-box, e.g., see the docs for [huggingface accelerate](https://huggingface.co/docs/accelerate/usage_guides/fsdp), [pytorch lightning](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.strategies.FSDPStrategy.html), [mosaic composer](https://docs.mosaicml.com/projects/composer/en/stable/notes/distributed_training.html#fullyshardeddataparallel-fsdp).

We are fine-tuning a [CodeLlama-7b](https://huggingface.co/codellama/CodeLlama-7b-hf) model on a dataset of [chess moves](https://huggingface.co/datasets/laion/strategic_game_chess) (in text notation). Full fine-tuning of all 7b parameters results in more parameters than fit on a single 80GB A100 GPU. We therefore use [Fully Sharded Data Parallel (FSDP)](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) to shard the model parameters on each device.
1. Install a conda environment with the most recent version of [PyTorch](https://pytorch.org/) (with CUDA) and huggingface transformers (`pip install transformers`). Activate the environment. See the directions on the Research Computing [PyTorch webpage](https://researchcomputing.princeton.edu/support/knowledge-base/pytorch).
2. On the Research Computing clusters, the compute nodes with GPUs are not connected to the internet. We therefore have to download the models manually first and cache them to our local directory. You can achieve this by running: `python download_models.py`
3. Look at `chess_finetune.py` and see how FSDP wrapping works. We implement special logic to define the FSDP units by layer.
4. Submit `chess_finetune.sh` to Slurm. This script requires 4 A100 GPUs by default.

### DDP vs. FSDP
Distributed data parallel (DDP) means that GPUs perform forward and backward pass on different examples in parallel, and gradients are exchanged between the GPUs before each optimization step. This means the total batch size is `batch_size_per_device * num_gpus`.
Fully-sharded data parallel (FSDP) still runs data parallelism, but in addition also shards the model parameters, and exchanges them between GPUs when necessary for a certain computation.

**Question:** Could the example training work on a single GPU? See what happens with a without model parallelism by adding the `--no_fsdp` flag and setting the `--batch_size_per_device` to 1.

### Tuning FSDP
We will see how FSDP can be tuned further. By sharding model parameters, we free up a lot of GPU memory, which can allow us to run with batch sizes larger than 1.

There are two different notions of batch size:
1. `batch_size_per_device` is the number of sequences that are processed in one pass per device. *You should always try to maximize this batch size for optimal GPU utilization and throughput.*
2. The other notion of batch size is from an optimization point of view, i.e. how many sequences to use per gradient update step. This can be a multiple of `batch_size_per_device` by using gradient accumulation, which sums the gradients without updating the parameters. Therefore, the total batch size for optimization is `batch_size_per_device * num_gpus * gradient_accumulation_steps`. Our script sets `gradient_accumulation_steps` to keep the total batch size constant. Therefore, we can change the number of GPUs and tune the `batch_size_per_device` without changing the optimization. Fixing the total batch size means that we won't have to tune the learning rate and other hyperparameters again.

**Question:** What is the maximum possible batch size per device when training with FSDP? Increase `BATCH_SIZE_PER_DEVICE` by factors of 2 and re-run `chess_finetune.sh`. How does the time per gradient step change as you increase the batch size per device?

**Question:** Try adding gradient checkpointing via `--gradient_checkpointing`. How fast is training without changing batch size? What happens if you further increase the batch size?

## Questions
Please reach out to `awettig@cs.princeton.edu`.
