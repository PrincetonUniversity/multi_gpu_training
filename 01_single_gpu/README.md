# Single-GPU Training

It is imperative to optimize your script using a single GPU before going to multiple GPUs. This is because as you request more resources, your queue time increases. We also don't want to waste resources by running code this not optimized.

Here we train a CNN on the MNIST dataset using a single GPU as an example. We the profile the code using various tools and make performance improvements.

This tutorial uses PyTorch but the steps are the same for TensorFlow. See the TensorFlow [installation directions](https://researchcomputing.princeton.edu/support/knowledge-base/tensorflow#install) and [performance tuning guide](https://tigress-web.princeton.edu/~jdh4/TensorflowPerformanceOptimization_GTC2021.pdf).

## Step 1: Installation

See the installation directions for [PyTorch](https://researchcomputing.princeton.edu/support/knowledge-base/pytorch) and then work through the MNIST example on that page. Below we will extended the MNIST example to use multiple GPUs.

Watch a [video](https://www.youtube.com/watch?v=wqTgM-Wq4YY&t=296s) of this procedure.

Learning about [profiling Python](https://researchcomputing.princeton.edu/python-profiling) codes using line_profiler.


## Step 2: Run the Script

First, inspect the script:

```
$ vim mnist_classifier.py  # or emacs, nano, micro, cat
```

```bash
#!/bin/bash
#SBATCH --job-name=mnist         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH --reservation=multigpu   # REMOVE THIS LINE AFTER THE WORKSHOP

module purge
module load anaconda3/2021.11
conda activate /scratch/network/jdh4/CONDA/envs/torch-env

python mnist_classify.py --epochs=3
```

Next, download the data and submit the job:

```bash
$ cd multi_gpu_training/01_single_gpu
$ module load anaconda3/2021.11
$ conda activate /scratch/network/jdh4/CONDA/envs/torch-env
$ python download_data.py
$ sbatch job.slurm.single
```

You should find that the code runs in about 64 seconds on a V100 GPU using 1 CPU-core:

```
$ seff 1268057
Job ID: 1268057
Cluster: adroit
User/Group: jdh4/cses
State: COMPLETED (exit code 0)
Cores: 1
CPU Utilized: 00:00:59
CPU Efficiency: 92.19% of 00:01:04 core-walltime
Job Wall-clock time: 00:01:04
Memory Utilized: 2.59 GB
Memory Efficiency: 64.72% of 4.00 GB
```

Some variation in the run time is expected when multiple users are running on the same node.

## Analyze the Profiling Data

We installed [line_profiler](https://researchcomputing.princeton.edu/python-profiling) into the Conda environment and profiled the code. To analyze the profiling data:



## Step 3: Work through the Performance Tuning Guide

Make sure you optimize the single GPU case before going to multiple GPUs by working through the [Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html). You should also profile your code using  or another tools like dlprof.

## Ste
