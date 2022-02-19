# Single-GPU Training

## Installation and Single-GPU Training Example

See the installation directions for [PyTorch](https://researchcomputing.princeton.edu/support/knowledge-base/pytorch) and then work through the MNIST example on that page. Below we will extended the MNIST example to use multiple GPUs.

```
#!/bin/bash
#SBATCH --job-name=mnist         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2021.11
conda activate /scratch/network/jdh4/CONDA/envs/torch-env

python mnist_classify.py --epochs=3
```

Watch a [video](https://www.youtube.com/watch?v=wqTgM-Wq4YY&t=296s) of this procedure.

## Optimizing the Single GPU Case

Make sure you optimize the single GPU case before going to multiple GPUs by working through the [Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html). You should also profile your code using [line_profiler](https://researchcomputing.princeton.edu/python-profiling) or another tools like dlprof.
