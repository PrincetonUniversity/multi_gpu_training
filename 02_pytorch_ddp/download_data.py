import torchvision
import warnings
warnings.filterwarnings("ignore")

# compute nodes do not have internet so download the data in advance

_ = torchvision.datasets.MNIST('data', train=True, transform=None,
                               target_transform=None, download=True)
