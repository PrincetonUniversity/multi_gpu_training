# PyTorch Lightning

Installation for Della-GPU or Adroit (A100):

```bash
$ module load anaconda3/2021.11
$ conda create --name torch-env pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
$ conda activate torch-env
$ pip install pytorch-lightning
```
