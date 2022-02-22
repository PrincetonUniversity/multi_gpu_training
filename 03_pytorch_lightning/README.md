# PyTorch Lightning

[PyTorch Lightning](https://www.pytorchlightning.ai) provides extension to PyTorch to ease and distributed training. One simply needs to restructure their code to take advantange of all the offerings of Lightning. Once the code is restructured, one can simply choose how many nodes or GPUs to use and Lightning will take care of the rest.

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

See the [Trainer API](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-class-api).
