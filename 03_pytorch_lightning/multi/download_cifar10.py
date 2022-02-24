import torchvision
import warnings
warnings.filterwarnings("ignore")

# compute nodes do not have internet access so download the data before
# submitting the job

_ = torchvision.datasets.CIFAR10('.', train=True, transform=None,
                                 target_transform=None, download=True)
