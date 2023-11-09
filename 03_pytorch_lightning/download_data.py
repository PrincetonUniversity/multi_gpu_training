import os
import torchvision
import warnings
warnings.simplefilter("ignore")

# compute nodes do not have internet so download the data in advance

_ = torchvision.datasets.MNIST(os.getcwd(),
                               transform=torchvision.transforms.ToTensor(),
                               download=True)
