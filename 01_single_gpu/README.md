# Single-GPU Training

It is important to optimize your script for single-GPU training before going to multiple GPUs. This is because as you request more resources, your queue time increases. We also want to avoid wasting resources by running code that is not optimized.

Here we train a CNN on the MNIST dataset using a single GPU as an example. We profile the code and make performance improvements.

This tutorial uses PyTorch but the steps are the similar for TensorFlow. See our [TensorFlow](https://researchcomputing.princeton.edu/support/knowledge-base/tensorflow#install) page and the [performance tuning guide](https://tigress-web.princeton.edu/~jdh4/TensorflowPerformanceOptimization_GTC2021.pdf).

## Step 1: Installation

**You do not need to install anything!** Throughout the live workshop we will use the following Conda environment:

```bash
module load anaconda3/2021.11
conda activate /scratch/network/jdh4/CONDA/envs/torch-env
```

After the workshop, see the installation directions for [PyTorch](https://researchcomputing.princeton.edu/support/knowledge-base/pytorch). See the [example job](https://researchcomputing.princeton.edu/support/knowledge-base/pytorch#example) on that webpage. During this workshop we will extend the MNIST example to use multiple GPUs.

Watch a [video](https://www.youtube.com/watch?v=wqTgM-Wq4YY&t=296s) of this procedure.

Learning about [profiling Python](https://researchcomputing.princeton.edu/python-profiling) codes using line_profiler.


## Step 2: Run and Profile the Script

First, inspect the script:

```
$ vim mnist_classify.py  # or emacs, nano, micro, cat
# type `:q` and then press the Enter/Return key to leave vim
```

Note that we will profile to train function using line_profiler:

```python
@profile
def train(args, model, device, train_loader, optimizer, epoch):
```

Notice in the Slurm script (see below) that a specific Conda environment belonging to jdh4 is used. This environment was created using the directions on the PyTorch RC webpage with the addition of line_profiler.

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

## Step 3: Analyze the Profiling Data

We installed [line_profiler](https://researchcomputing.princeton.edu/python-profiling) into the Conda environment and profiled the code. To analyze the profiling data:

```
$ python -m line_profiler mnist_classify.py.lprof 
Timer unit: 1e-06 s

Total time: 48.428 s
File: mnist_classify.py
Function: train at line 39

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    39                                           @profile
    40                                           def train(args, model, device, train_loader, optimizer, epoch):
    41         3        343.0    114.3      0.0      model.train()
    42      2817   40013383.0  14204.3     82.6      for batch_idx, (data, target) in enumerate(train_loader):
    43      2814     195724.0     69.6      0.4          data, target = data.to(device), target.to(device)
    44      2814     353065.0    125.5      0.7          optimizer.zero_grad()
    45      2814    2238901.0    795.6      4.6          output = model(data)
    46      2814      95510.0     33.9      0.2          loss = F.nll_loss(output, target)
    47      2814    2848827.0   1012.4      5.9          loss.backward()
    48      2814    2661568.0    945.8      5.5          optimizer.step()
    49      2814       5023.0      1.8      0.0          if batch_idx % args.log_interval == 0:
    50       564       2482.0      4.4      0.0              print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    51       282       3041.0     10.8      0.0                  epoch, batch_idx * len(data), len(train_loader.dataset),
    52       282       9895.0     35.1      0.0                  100. * batch_idx / len(train_loader), loss.item()))
    53       282        201.0      0.7      0.0              if args.dry_run:
    54                                                           break
```

The slowest line in number 42 which consumes 82.6% of the time in the training function. That line involves train_loader which is the data loader for the training set. Are you surprised that the data loader is the slowest step? Can we improve on this?

## Examine Your GPU Utilization

Use tools like [jobstats](https://researchcomputing.princeton.edu/support/knowledge-base/job-stats#jobstats), [gpudash](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing#gpudash) and [stats.rc](https://researchcomputing.princeton.edu/support/knowledge-base/job-stats#stats.rc) to measure your GPU utilization. You can also do this on a [compute node in real time](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing#gpu-utilization).

Note that GPU utilization as measured using nvidia-smi is only a measure of the fraction of the time that a GPU kernel is running on the GPU. It says nothing about how many CUDA cores are being used or how efficiently the GPU kernels have been written. However, for codes used by large communities, one can generally associate GPU utilization with overall GPU efficiency. For a more accurate measure of GPU utilization, use [Nsight Systems or Nsight Compute](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing#profiling) to measure the occupancy.

## Step 3: Work through the Performance Tuning Guide

Make sure you optimize the single GPU case before going to multiple GPUs by working through the [Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html). You should also profile your code using  or another tools like dlprof.

## Step 4: Optimize Your Script

One technique that was discussed in the Performance Tuning Guide was using multiple CPU-cores to speed-up data loading. Let's put this into practice.

In `mnist_classify.py`, change `num_workers` from 1 to 8. And then in `job.slurm.single` change `--cpus-per-task` from 1 to 8. Then run the script again and note the speed-up. How did the profiling data change?

![multiple_workers](https://www.telesens.co/wp-content/uploads/2019/04/img_5ca4eff975d80.png)

*Credit for image above is [here](https://www.telesens.co/2019/04/04/distributed-data-parallel-training-using-pytorch-on-aws/).*

Consider these external data loading libraries: [ffcv](https://github.com/libffcv/ffcv) and [NVIDIA DALI](https://developer.nvidia.com/dali).

## Summary

It is essential to optimize your code before going to multi-GPU training since the inefficiencies will only be magnified otherwise. The more GPUs you request in a Slurm job, the longer you wait for the job to run. Don't waste resources. Optimize your code and then scale it. Next, we focus on scaling the code to multiple GPUs.
