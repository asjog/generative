import torch 
# get the mnist dataset and download it 
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.optim import Adam
from torch.nn import functional as F
from torch import nn
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


# declare mnist dataset and with download 

train_set = MNIST('data/mnist/', download=False,transform=ToTensor(), train=True)    

plt.imshow(train_set[0][0].squeeze().numpy())
plt.show()
