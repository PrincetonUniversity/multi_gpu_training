# PyTorch Lightning

[PyTorch Lightning](https://www.pytorchlightning.ai) wraps PyTorch to provide easy and distributed training. One simply needs to:

+ restructure the code by moving the network definition and optimizer to a subclass of `pl.LightningModule`  
+ remove .cuda() and .to() calls since Lightning code should be hardware agnostic  

Once these changes have, one can simply choose how many nodes or GPUs to use and Lightning will take care of the rest. One can also use different numerical precisions (fp16, bf16), there is tensorboard support and DDP.
## Installation

Della-GPU or Adroit (A100):

```bash
$ module load anaconda3/2021.11
$ conda create --name torch-lit-env pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
$ conda activate torch-pl-env
$ pip install pytorch-lightning
```

Run the example job:

```
$ cd /scratch/gpfs/<YourNetID>  # or /scratch/network on adroit
$ git clone https://github.com/PrincetonUniversity/multi_gpu_training.git
$ cd multi_gpu_training/03_multi_gpu_lightning
$ wget https://raw.githubusercontent.com/PrincetonUniversity/install_pytorch/master/download_mnist.py
$ python download_mnist.py
```

See the [Trainer API](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-class-api).

## Single-GPU Example

```
#!/bin/bash
#SBATCH --job-name=myjob         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=1:00:00           # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1             # number of gpus per node

module purge
module load anaconda3/2020.11
conda activate torch-pl-env

python myscript.py
```

## Multi-GPU Example

Let's work through this [example](https://pytorch-lightning.readthedocs.io/en/latest/notebooks/lightning_examples/cifar10-baseline.html) where a modified resnet18 model is training on CIFAR10. Here is the application script:

```python
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pl_bolts.datamodules import CIFAR10DataModule
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy

seed_everything(7)

BATCH_SIZE = 256
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
NUM_WORKERS   = int(os.environ["SLURM_CPUS_PER_TASK"])
NUM_NODES     = int(os.environ["SLURM_NNODES"])
GPUS_PER_NODE = int(os.environ["SLURM_GPUS_ON_NODE"])

train_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

test_transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        cifar10_normalization(),
    ]
)

cifar10_dm = CIFAR10DataModule(
    data_dir=PATH_DATASETS,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    train_transforms=train_transforms,
    test_transforms=test_transforms,
    val_transforms=test_transforms,
)

def create_model():
    model = torchvision.models.resnet18(pretrained=False, num_classes=10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    model.maxpool = nn.Identity()
    return model

class LitResnet(LightningModule):
    def __init__(self, lr=0.05):
        super().__init__()

        self.save_hyperparameters()
        self.model = create_model()

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        steps_per_epoch = 45000 // BATCH_SIZE
        scheduler_dict = {
            "scheduler": OneCycleLR(
                optimizer,
                0.1,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=steps_per_epoch,
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

model = LitResnet(lr=0.05)
model.datamodule = cifar10_dm

trainer = Trainer(
    gpus=GPUS_PER_NODE,
    num_nodes=NUM_NODES,
    strategy='ddp',
    precision=32,
    max_epochs=10,
    progress_bar_refresh_rate=10,
    logger=TensorBoardLogger("lightning_logs/", name="resnet"),
    callbacks=[LearningRateMonitor(logging_interval="step")],
)

trainer.fit(model, cifar10_dm)
trainer.test(model, datamodule=cifar10_dm)
```


### Step 1: Installation

**You can skip this step during the live workshop.**

Create the following Conda environment:

```
$ module load anaconda3/2021.11
$ conda create --name bolts python=3.9 -y
$ conda activate bolts
$ pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 pytorch-lightning lightning-bolts -f https://download.pytorch.org/whl/torch_stable.html
```

### Step 2: Download the data

The compute nodes do not have internet access so download the data on the login node:

```
$ cd multi_gpu_training/03_lightning
$ python download_cifar10.py
```

### Step 3: Submit the Job

Below is the Slurm script:

```
#!/bin/bash
#SBATCH --job-name=myjob         # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=1:00:00           # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:2             # number of gpus per node
# SBATCH --constraint=a100

module purge
module load anaconda3/2021.11
conda activate /scratch/network/jdh4/CONDA/envs/bolts

export PL_TORCH_DISTRIBUTED_BACKEND=gloo

srun python myscript.py
```

Submit the job:

```
$ sbatch job.slurm
```

By default, DDP uses "nccl" as its backend. The code was found to hang so "gloo" was used.

How does the training time decrease in going from 1, 2 to 4 GPUs? What happens if you use `precision=16`?

## Numerical Precision

You can try adjusting the `precision` to accelerate training. The choice of `precision="bf16"` can only be used with PyTorch 1.10.

## Debugging

For troubleshooting NCCL try adding these environment variables to your Slurm script:

```
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

## Useful Links

+ [PyTorch Lightning and Slurm](https://pytorch-lightning.readthedocs.io/en/latest/clouds/cluster.html)  
+ [PyTorch LightningLite](https://pytorch-lightning.readthedocs.io/en/latest/starter/lightning_lite.html)
