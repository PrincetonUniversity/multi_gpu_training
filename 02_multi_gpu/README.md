# Multi-GPU Training with PyTorch

One should always try to use only a single GPU. This maximizes efficiency. However, there are two common reasons for using multiple GPUs when training neural networks:
- the execution time is too long with a single GPU
- the model is too large to fit on a single GPU

The more GPUs you request for a Slurm job, the longer the queue time will be.

## Overall Idea of Distributed Data Parallel

SPMD paradigm is used. Model is copied on each GPU so want an optimized version.

![ddp](https://www.telesens.co/wp-content/uploads/2019/04/img_5ca570946ee1c.png)

*Credit for the image above is [here](https://www.telesens.co/wp-content/uploads/2019/04/img_5ca570946ee1c.png).*

## Read about DistributedDataParallel

[Getting Started with Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)  
[See the docs on DPP](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)  
[blog](https://github.com/pytorch/examples/blob/master/distributed/ddp/README.md)

## Caveats

Do not use DataParallel in PyTorch since it gives poor performance relative to DDP.

## Main changes needed in going from single-GPU to multi-GPU training with DDP

This completely new piece is needed to form the process group:

```python
def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    print("group initialized?", dist.is_initialized(), flush=True)
```

Note that dist.init_process_group() is blocking. That means the code waits until all processes have reached that line and the command is successfully executed before going on.

For the single-GPU training:

```python
model = Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
```

For multi-GPU training with DPP:

```python
model = Net().to(local_rank)
ddp_model = DDP(model, device_ids=[local_rank])
optimizer = optim.Adadelta(ddp_model.parameters(), lr=args.lr)
```

## Simple DDP Script

The following can be used for debugging and provides a simple intro to DPP:

```python
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from socket import gethostname

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(42, 3)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        output = F.log_softmax(x, dim=1)
        return output

world_size = int(os.environ["WORLD_SIZE"])
rank = int(os.environ["SLURM_PROCID"])
gpus_per_node = int(os.environ["GPUS_PER_NODE"])
assert gpus_per_node == torch.cuda.device_count()
print(f"Hello from rank {rank} on {gethostname()} with {gpus_per_node} GPUs per node.", flush=True)

dist.init_process_group("nccl", rank=rank, world_size=world_size)
if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

local_rank = rank - gpus_per_node * (rank // gpus_per_node)
torch.cuda.set_device(local_rank)

model = Net().to(local_rank)
ddp_model = DDP(model, device_ids=[local_rank])

ddp_model.eval()
with torch.no_grad():
  data = torch.rand(1, 42)
  data = data.to(local_rank)
  output = ddp_model(data)
  print(f"host: {gethostname()}, rank: {rank}, output: {output}")

dist.destroy_process_group()
```

`GPUS_PER_NODE` is defined in the Slurm script (see below) and `SLURM_PROCID` is assigned by Slurm. `SLURM_PROCID` varies from 0 to N - 1 where N is the number of tasks running under `srun`. For instance, consider the abbreviated Slurm script below:

```
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
...
srun python myscript.py
```

The Python interpreter will be launched 8 times (2 x 4) and each of the 8 tasks will have a different value of `SLURM_PROCID` from the set 0, 1, 2, 3, 4, 5, 6, 7.

## What is `local_rank`?

The indices of the GPUs on each node of your Slurm allocation begin at 0 and end at N - 1, where N is the total number of GPUs in your allocation on each node. Consider the case of 2 nodes and 8 tasks with 4 GPUs per node. The process ranks will be 0, 1, 2, 3 on the first node and 4, 5, 7 on the second node while the GPU indices will be 0, 1, 2, 3 on the first and 0, 1, 2, 3 on the second. Thus, one cannot make calls such as `data.to(rank)` since this will fail on the second node where there is a mismatch between the process ranks and the GPU indices. To deal with this a local rank is introduced:

```python
rank = int(os.environ["SLURM_PROCID"])
gpus_per_node = int(os.environ["GPUS_PER_NODE"])
local_rank = rank - gpus_per_node * (rank // gpus_per_node)
```

The `local_rank` shoud be used everywhere in your script except when initializing the DDP process group where `rank` should be used. In Python, one uses the  `//` operator for integer division. For example, `1 / 2 = 0.5` while `1 // 2 = 0`.

# DDP and Slurm

## Total number of tasks equals total number of GPUs

When using DDP, the total number of tasks must equal the total number of allocated GPUs. Therefore, if `--ntasks-per-node=<N>` then you must have `--gres=gpu:<N>`. Here are two examples:

```
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
```

```
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
```

You should take all of the GPUs on a node before going to multiple nodes. Never do one GPU per node for multinode jobs.

## Slurm Script

Below is an example Slurm script for using DDP for Della (GPU) where there are 2 GPUs per node:

```bash
#!/bin/bash
#SBATCH --job-name=ddp-torch     # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=2      # total number of tasks per node
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of gpus per node
#SBATCH --time=00:01:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=<YourNetID>@princeton.edu

export MASTER_PORT=$((10000 + $(echo -n $SLURM_JOBID | tail -c 4)))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

gpus_per_node=$(printf ${SLURM_JOB_GPUS} | sed 's/[^0-9]*//g' | wc --chars)
export GPUS_PER_NODE=$gpus_per_node
echo "GPUS_PER_NODE="$GPUS_PER_NODE
# or use SLURM_GPUS_ON_NODE

module purge
module load anaconda3/2021.11
conda activate torch-env

srun python myscript.py
```

The script above uses 3 nodes with 2 tasks per node and therefore 2 GPUs per node. This yields a total of 4 processes and each process can use 8 CPU-cores for data loading. An allocation of 3 nodes is substantial so the queue will be quite long. Try using 1 or 2 nodes per job. In all cases make sure that the GPUs are being used efficiently by monitoring the [GPU utilization](https://researchcomputing.princeton.edu/support/knowledge-base/gpu-computing).

## Full script

```python
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from socket import gethostname

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': int(os.environ["SLURM_CPUS_PER_TASK"]),
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('data', train=True, download=False,
                       transform=transform)
    dataset2 = datasets.MNIST('data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["SLURM_PROCID"])
    gpus_per_node = int(os.environ["GPUS_PER_NODE"])
    assert gpus_per_node == torch.cuda.device_count()
    print(f"Hello from rank {rank} on {gethostname()} with {gpus_per_node} GPUs per node.", flush=True)

    setup(rank, world_size)
    if rank == 0: print(f"Group initialized? {dist.is_initialized()}", flush=True)

    local_rank = rank - gpus_per_node * (rank // gpus_per_node)
    torch.cuda.set_device(local_rank)
    print(f"host: {gethostname()}, rank: {rank}, local_rank: {local_rank}")

    model = Net().to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])
    optimizer = optim.Adadelta(ddp_model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, ddp_model, local_rank, train_loader, optimizer, epoch)
        test(ddp_model, local_rank, test_loader)
        scheduler.step()

    if args.save_model and rank == 0:
        torch.save(model.state_dict(), "mnist_cnn.pt")

    dist.destroy_process_group()


if __name__ == '__main__':
    main()
```

# TDO

Need to plot loss or accuracy as a function of epochs for different choices of number of GPUs and cpus-per-task.

```
>>> import torch
# next line in blocking until all processes join
>>> torch.distributed.init_process_group("nccl", init_method='file:///scratch/network/jdh4/sharedfile', rank=0, world_size=1)
>>> torch.distributed.is_initialized()
True
>>> torch.distributed.is_initialized()
True
```

[https://github.com/libffcv/ffcv](https://github.com/libffcv/ffcv)
