# Single-GPU Training

It is important to optimize your script for the single-GPU case before moving to multi-GPU training. This is because as you request more resources, your queue time increases. We also want to avoid wasting resources by running code that is not optimized.

Here we train a CNN on the MNIST dataset using a single GPU as an example. We profile the code and make performance improvements.

This tutorial uses PyTorch but the steps are the similar for TensorFlow. See our [TensorFlow](https://researchcomputing.princeton.edu/support/knowledge-base/tensorflow#install) page and the [performance tuning guide](https://tigress-web.princeton.edu/~jdh4/TensorflowPerformanceOptimization_GTC2021.pdf).

## Step 1: Activate the Environment

For simplicity we will use a pre-installed Conda environmnet. Run these commands to activate the environment:

```bash
$ ssh <YourNetID>@adroit.princeton.edu
$ module load anaconda3/2023.9
$ conda activate /home/jdh4/.conda/envs/torch-env
```

Watch a [video](https://www.youtube.com/watch?v=wqTgM-Wq4YY&t=296s) that covers everything on this page for single-GPU training with [profiling Python](https://researchcomputing.princeton.edu/python-profiling) using `line_profiler`.

## Step 2: Run and Profile the Script

First, inspect the script ([see script](mnist_classify.py)) by running these commands:

```bash
(torch-env) $ cd multi_gpu_training/01_single_gpu
(torch-env) $ cat mnist_classify.py
```

We will profile the `train` function using `line_profiler` (see line 39) by adding the following decorator:

```python
@profile
def train(args, model, device, train_loader, optimizer, epoch):
```

Next, download the data while on the login node since the compute nodes do not have internet access:

```
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

# which gpu node was used
echo "Running on host" $(hostname)

# print the slurm environment variables sorted by name
printenv | grep -i slurm | sort

module purge
module load anaconda3/2023.9
conda activate /home/jdh4/.conda/envs/torch-env

kernprof -o ${SLURM_JOBID}.lprof -l mnist_classify.py --epochs=3
```

`kernprof` is a profiler that wraps Python. Adroit has two different A100 nodes. Learn how to choose [specific nodes](https://researchcomputing.princeton.edu/systems/adroit#gpus).

Finally, submit the job while specifying the reservation:

```bash
(torch-env) $ sbatch --reservation=multigpu job.slurm
```

You should find that the code runs in about 20-40 seconds with 1 CPU-core depending on which A100 GPU node was used:

```
$ seff 1937315
Job ID: 1937315
Cluster: adroit
User/Group: aturing/cses
State: COMPLETED (exit code 0)
Cores: 1
CPU Utilized: 00:00:36
CPU Efficiency: 94.74% of 00:00:38 core-walltime
Job Wall-clock time: 00:00:38
Memory Utilized: 593.32 MB
Memory Efficiency: 7.24% of 8.00 GB
```

For jobs that run for longer than 1 minute, one should use the `jobstats` command instead of `seff`. Use `shistory -n` to see which node was used or look in the `slurm-#######.out` file.

Some variation in the run time is expected when multiple users are running on the same node. Also, the two A100 GPU nodes are not equal:

| hostname | CPU | GPU |
| ----------- | ----------- | ----------- |
| adroit-h11g1 | Intel Xeon Gold 6442Y @ 2.6GHz | NVIDIA A100 80GB PCIe |
| adroit-h11g2 | Intel Xeon Gold 6342  @ 2.8GHz | NVIDIA A100-PCIE-40GB |

## Step 3: Analyze the Profiling Data

We installed [line_profiler](https://researchcomputing.princeton.edu/python-profiling) into the Conda environment and profiled the code. To analyze the profiling data:

```
(torch-env) $ python -m line_profiler -rmt *.lprof 
Timer unit: 1e-06 s

Total time: 30.8937 s
File: mnist_classify.py
Function: train at line 39

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    39                                           @profile
    40                                           def train(args, model, device, train_loader, optimizer, epoch):
    41         3        213.1     71.0      0.0      model.train()
    42      2817   26106124.7   9267.3     84.5      for batch_idx, (data, target) in enumerate(train_loader):
    43      2814     286242.0    101.7      0.9          data, target = data.to(device), target.to(device)
    44      2814     296440.2    105.3      1.0          optimizer.zero_grad()
    45      2814    1189206.1    422.6      3.8          output = model(data)
    46      2814      81578.6     29.0      0.3          loss = F.nll_loss(output, target)
    47      2814    1979990.2    703.6      6.4          loss.backward()
    48      2814     841861.9    299.2      2.7          optimizer.step()
    49      2814       2095.3      0.7      0.0          if batch_idx % args.log_interval == 0:
    50       564       1852.9      3.3      0.0              print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    51       282       2218.6      7.9      0.0                  epoch, batch_idx * len(data), len(train_loader.dataset),
    52       282     105753.3    375.0      0.3                  100. * batch_idx / len(train_loader), loss.item()))
    53       282        119.2      0.4      0.0              if args.dry_run:
    54                                                           break

 30.89 seconds - mnist_classify.py:39 - train
```

The slowest line is number 42 which consumes 84.5% of the time in the training function. That line involves `train_loader` which is the data loader for the training set. Are you surprised that the data loader is the slowest step and not the forward pass or calculation of the gradients? Can we improve on this?

### Examine Your GPU Utilization

Use tools like [jobstats](https://researchcomputing.princeton.edu/support/knowledge-base/job-stats#jobstats), [gpudash](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing#gpudash) and [stats.rc](https://researchcomputing.princeton.edu/support/knowledge-base/job-stats#stats.rc) to measure your GPU utilization. You can also do this on a [compute node in real time](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing#gpu-utilization).

Note that GPU utilization as measured using nvidia-smi is only a measure of the fraction of the time that a GPU kernel is running on the GPU. It says nothing about how many CUDA cores are being used or how efficiently the GPU kernels have been written. However, for codes used by large communities, one can generally associate GPU utilization with overall GPU efficiency. For a more accurate measure of GPU utilization, use [Nsight Systems or Nsight Compute](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing#profiling) to measure the occupancy.

## Step 4: Work through the Performance Tuning Guide

Make sure you optimize the single GPU case before going to multiple GPUs by working through the [Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

## Step 5: Optimize Your Script

One technique that was discussed in the [Performance Tuning Guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) was using multiple CPU-cores to speed-up [ETL](https://en.wikipedia.org/wiki/Extract,_transform,_load). Let's put this into practice.

![multiple_workers](https://www.telesens.co/wp-content/uploads/2019/04/img_5ca4eff975d80.png)

*Credit for image above is [here](https://www.telesens.co/2019/04/04/distributed-data-parallel-training-using-pytorch-on-aws/).*

In `mnist_classify.py`, change `num_workers` from 1 to 8. And then in `job.slurm` change `--cpus-per-task` from 1 to 8. Then run the script again and note the speed-up:

```
(torch-env) $ sbatch --reservation=multigpu job.slurm
```

How did the profiling data change? Watch the [video](https://www.youtube.com/watch?v=wqTgM-Wq4YY&t=296s) for the solution. For consistency between the Slurm script and PyTorch script, one can use:

```python
import os
...
    cuda_kwargs = {'num_workers': int(os.environ["SLURM_CPUS_PER_TASK"]),
...
```

Several environment variables are set in the Slurm script. These can be referenced by the PyTorch script as demonstrated above. To see all of the available environment variables that are set in the Slurm script, add this line to `job.slurm`:

```
printenv | sort
```

Consider these external data loading libraries: [ffcv](https://github.com/libffcv/ffcv) and [NVIDIA DALI](https://developer.nvidia.com/dali).

## Summary

It is essential to optimize your code before going to multi-GPU training since the inefficiencies will only be magnified otherwise. The more GPUs you request in a Slurm job, the longer you will wait for the job to run. If you can get your work done using an optimized script running on a single GPU then proceed that way. Do not use multiple GPUs if your GPU efficiency is low. The average GPU efficiency on Della is around 50%.

Next, we focus on scaling the code to multiple GPUs (go to [next section](../02_pytorch_ddp)).

## How was the Conda environment made?

Please do not do this during the workshop. Your `/home` directory on Adroit probably has a capacity of 9.3 GB. To store Conda environments in another location see [this page](https://researchcomputing.princeton.edu/support/knowledge-base/checkquota). See the Research Computing knowledge base on [PyTorch](https://researchcomputing.princeton.edu/support/knowledge-base/pytorch).

```bash
$ ssh <YourNetID>@adroit.princeton.edu
$ module load anaconda3/2023.9
$ conda create --name torch-env pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
$ conda activate torch-env
$ conda install line_profiler --channel conda-forge
```
