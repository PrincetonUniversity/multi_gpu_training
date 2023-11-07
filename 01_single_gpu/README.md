# Single-GPU Training

It is important to optimize your script for the single-GPU case before moving to multi-GPU training. This is because as you request more resources, your queue time increases. We also want to avoid wasting resources by running code that is not optimized.

Here we train a CNN on the MNIST dataset using a single GPU as an example. We profile the code and make performance improvements.

This tutorial uses PyTorch but the steps are the similar for TensorFlow. See our [TensorFlow](https://researchcomputing.princeton.edu/support/knowledge-base/tensorflow#install) page and the [performance tuning guide](https://tigress-web.princeton.edu/~jdh4/TensorflowPerformanceOptimization_GTC2021.pdf).

## Step 1: Installation

Follow the directions on our [PyTorch](https://researchcomputing.princeton.edu/support/knowledge-base/pytorch) webpage to install PyTorch. Then activate the environment and install the profiler:

```
$ conda activate torch-env
(torch-env) $ conda install line_profiler --channel conda-forge
```

Watch a [video](https://www.youtube.com/watch?v=wqTgM-Wq4YY&t=296s) that covers everything on this page for single-GPU training with [profiling Python](https://researchcomputing.princeton.edu/python-profiling) using `line_profiler`.


## Step 2: Run and Profile the Script

First, inspect the script ([see script](mnist_classify.py)).

Note that we will profile the `train` function using `line_profiler` (see line 39):

```python
@profile
def train(args, model, device, train_loader, optimizer, epoch):
```

Next, download the data while on the login node since the compute nodes do not have internet access:

```bash
(torch-env) $ cd multi_gpu_training/01_single_gpu
(torch-env) $ python download_mnist.py
```

Below is the Slurm script:

```bash
#!/bin/bash
#SBATCH --job-name=mnist         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=8G                 # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:05:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=<YourNetID>@princeton.edu

module purge
module load anaconda3/2023.9
conda activate torch-env

kernprof -l mnist_classify.py --epochs=3
```

Finally, submit the job:

```bash
(torch-env) $ sbatch job.slurm  # edit your email address in job.slurm before submitting
```

You should find that the code runs in about 1 minute on an A100 GPU using 1 CPU-core:

```
$ seff 51876015
Job ID: 51876015
Cluster: della
User/Group: aturing/cses
State: COMPLETED (exit code 0)
Cores: 1
CPU Utilized: 00:00:42
CPU Efficiency: 95.45% of 00:00:44 core-walltime
Job Wall-clock time: 00:00:44
Memory Utilized: 438.79 MB
Memory Efficiency: 5.36% of 8.00 GB
```

For jobs that run for longer than 1 minute, one should use the `jobstats` command instead of `seff`.

Some variation in the run time is expected when multiple users are running on the same node.

## Step 3: Analyze the Profiling Data

We installed [line_profiler](https://researchcomputing.princeton.edu/python-profiling) into the Conda environment and profiled the code. To analyze the profiling data:

```
(torch-env) $ python -m line_profiler mnist_classify.py.lprof 
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

The slowest line is number 42 which consumes 82.6% of the time in the training function. That line involves train_loader which is the data loader for the training set. Are you surprised that the data loader is the slowest step? Can we improve on this?

## Examine Your GPU Utilization

Use tools like [jobstats](https://researchcomputing.princeton.edu/support/knowledge-base/job-stats#jobstats), [gpudash](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing#gpudash) and [stats.rc](https://researchcomputing.princeton.edu/support/knowledge-base/job-stats#stats.rc) to measure your GPU utilization. You can also do this on a [compute node in real time](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing#gpu-utilization).

Note that GPU utilization as measured using nvidia-smi is only a measure of the fraction of the time that a GPU kernel is running on the GPU. It says nothing about how many CUDA cores are being used or how efficiently the GPU kernels have been written. However, for codes used by large communities, one can generally associate GPU utilization with overall GPU efficiency. For a more accurate measure of GPU utilization, use [Nsight Systems or Nsight Compute](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing#profiling) to measure the occupancy.

## Step 4: Work through the Performance Tuning Guide

Make sure you optimize the single GPU case before going to multiple GPUs by working through the [Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

## Step 5: Optimize Your Script

One technique that was discussed in the [Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) was using multiple CPU-cores to speed-up [ETL](https://en.wikipedia.org/wiki/Extract,_transform,_load). Let's put this into practice.

In `mnist_classify.py`, change `num_workers` from 1 to 8. And then in `job.slurm` change `--cpus-per-task` from 1 to 8. Then run the script again and note the speed-up. How did the profiling data change? Watch the [video](https://www.youtube.com/watch?v=wqTgM-Wq4YY&t=296s) for the solution.

![multiple_workers](https://www.telesens.co/wp-content/uploads/2019/04/img_5ca4eff975d80.png)

*Credit for image above is [here](https://www.telesens.co/2019/04/04/distributed-data-parallel-training-using-pytorch-on-aws/).*

Consider these external data loading libraries: [ffcv](https://github.com/libffcv/ffcv) and [NVIDIA DALI](https://developer.nvidia.com/dali).

## Summary

It is essential to optimize your code before going to multi-GPU training since the inefficiencies will only be magnified otherwise. The more GPUs you request in a Slurm job, the longer you will wait for the job to run. Don't waste resources. Optimize your code and then scale it. Next, we focus on scaling the code to multiple GPUs (go to [next section](../02_pytorch_ddp)).
