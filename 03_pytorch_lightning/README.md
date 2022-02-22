# PyTorch Lightning

[PyTorch Lightning](https://www.pytorchlightning.ai) provides extension to PyTorch to ease and distributed training. One simply needs to restructure their code to take advantange of all the offerings of Lightning. Once the code is restructured, one can simply choose how many nodes or GPUs to use and Lightning will take care of the rest.

Installation for Della-GPU or Adroit (A100):

```bash
$ module load anaconda3/2021.11
$ conda create --name torch-lit-env pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
$ conda activate torch-lit-env
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
conda activate torch-lit-env

python myscript.py
```

## Multi-GPU Example

```
#!/bin/bash
#SBATCH --job-name=myjob         # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --ntasks=2               # total number of tasks across all nodes
#SBATCH --cpus-per-task=8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G per cpu-core is default)
#SBATCH --time=1:00:00           # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:2             # number of gpus per node

module purge
module load anaconda3/2020.11
conda activate torch-lit-env

srun python myscript.py
```
