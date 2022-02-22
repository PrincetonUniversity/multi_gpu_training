# PyTorch Lightning

Installation for Della-GPU or Adroit (A100):

```bash
$ module load anaconda3/2021.11
$ conda create --name torch-env pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
$ conda activate torch-env
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
