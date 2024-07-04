import torch
from torch import nn 
from torch.nn import functional as F
from monai.networks.layers.convutils import calculate_out_shape, same_padding
from torch.utils.data import DataLoader

from monai.networks.nets import varautoencoder
from torchvision.datasets import CelebA, MNIST
from torch.utils.data import Dataset, DataLoader
import PIL
from torchvision.transforms import ToTensor
import glob 
from matplotlib import pyplot as plt

class vae_encoder(nn.Module):
    def __init__(self, input_shape, latent_dim) -> None:
        super().__init__()
        # initialize the conv layers 
        self.num_layers = 4
        self.start_num_filters = 8
        self.conv_list = []
        prev_in_channels = 3
        for i in range(self.num_layers):
            num_filters = self.start_num_filters * (2**i)
            conv_layer = nn.Conv2d(in_channels=prev_in_channels, out_channels=num_filters, kernel_size=3, stride=1)
            padding = same_padding(kernel_size=3, dilation=1)
            pad_layer = nn.ZeroPad2d(padding=padding)

            pool_layer = nn.MaxPool2d(kernel_size=2)
            act = nn.ReLU()


            self.conv_list.append(conv_layer)
            self.conv_list.append(pad_layer)
            self.conv_list.append(act)
            self.conv_list.append(pool_layer)

            prev_in_channels  = num_filters
        
        self.cnn = nn.Sequential(*self.conv_list)
        in_features = 4096
        self.mu = nn.Linear(in_features=in_features, out_features=latent_dim)
        # sigma is the diagonal of the covariance matrix
        self.logsigma = nn.Linear(in_features=in_features, out_features=latent_dim)

    def forward(self, x):
        x = self.cnn(x)

        print(x.shape)

        x = x.view(x.size(0), -1)
        mu = self.mu(x)
        logsigma = self.logsigma(x)
        # sample z from N(z | mu, sigma)
        # this means sample w (5, 4096) from N(w |0, 1) and mu (5, 4096) + w * sqrt(sigma) (5, 4096)

        z = torch.normal(mu, torch.exp(0.5*logsigma))

        return z, mu, logsigma

class vae_decoder(nn.Module):
    def __init__(self, latent_dim, output_shape) -> None:
        super().__init__()
        self.updim = nn.Linear(latent_dim, 4096)
        self.num_layers = 4
        self.conv_list = []
        self.start_num_filters = 64
        prev_in_channels = 64 
        for i in range(self.num_layers):
            num_filters = self.start_num_filters // (2**i)
            conv_layer = nn.Conv2d(in_channels=prev_in_channels, out_channels=num_filters, kernel_size=3)
            self.conv_list.append(conv_layer)
            padding = same_padding(kernel_size=3, dilation=1)
            pad_layer = nn.ZeroPad2d(padding=padding)
            self.conv_list.append(pad_layer)

            act_layer = nn.ReLU()
            self.conv_list.append(act_layer)
            up_sample_layer = nn.Upsample(scale_factor=2)
            
            self.conv_list.append(up_sample_layer)

            prev_in_channels = num_filters
        

        self.conv_list.append(nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3))
        self.conv_list.append(pad_layer)

        self.cnn = nn.Sequential(*self.conv_list)

    def forward(self, x):
        # x shape is 1x2304
        x = self.updim(x)
        print(x.shape)
        x = x.view(x.size(0), 64, 8, 8)
        print(x.shape)
        x = self.cnn(x)
        return x 
    


class vae(nn.Module):
    def __init__(self, input_shape, latent_dim) -> None:
        super().__init__()
        self.encoder = vae_encoder(input_shape, latent_dim)
        self.decoder = vae_decoder(input_shape, latent_dim)


def l2_loss(x, xhat, sigma_dec=1):
    l2_loss = torch.nn.MSELoss()

    return l2_loss(x, xhat)/sigma_dec

def kl_divergence_loss(mu, logsigma):
    """ kl divergence between a multivariable normal distribution with mean mu and diagonal 
    of covariance matrix sigma"""
    # import pdb 
    # pdb.set_trace()
    term1 = torch.sum(torch.exp(logsigma), axis=1)
    term2 = torch.sum(mu**2, axis=1)
    term3 = torch.sum(logsigma, axis=1)
    term4 = mu.shape[1]

    # kld =  0.5* (torch.sum(sigma, axis=1) + torch.sum(mu**2, axis=1) - torch.sum(torch.log(sigma), axis=1) - mu.shape[1])
    kld = 0.5 * (term1 + term2 - term3 - term4)
    kld = torch.mean(kld)

    return kld


def vae_loss_function(x, xhat, mu, sigma):
    mse = l2_loss(x, xhat)
    kl_loss = kl_divergence_loss(mu, sigma)
    return mse + kl_loss

class CelebADataset(Dataset):
    def __init__(self, img_files) -> None:
        self.img_files = img_files 
    def __getitem__(self, index):
        img_file = self.img_files[index]
        img = PIL.Image.open(img_file)
        # resize the image to 128x128
        img = img.resize((128, 128))
        img = ToTensor()(img)
        return img
    def __len__(self):
        return len(self.img_files)
    

img_dir = "/Users/asjog/VAE-research/data/celeba/celeba/img_align_celeba/"
celeba_img_files = sorted(glob.glob(img_dir + "*.jpg"))

celeba_dataset = CelebADataset(celeba_img_files)

# define a dataloader for celeba dataset
celeba_dataloader = DataLoader(celeba_dataset, batch_size=8, shuffle=True)
# plot a single batch of images with plt imshow
for batch in celeba_dataloader:
    print(batch.shape)
    # plot all images of the batch in a subplots 
    fig, ax = plt.subplots(1, 8)
    for i in range(8):
        ax[i].imshow(batch[i].permute(1, 2, 0))

    plt.show()
    break


enc = vae_encoder((3, 128, 128), 4096)
dec = vae_decoder(4096, (3, 128, 128))

batch_z, batch_mu, batch_logsima = enc(batch)
recon = dec(batch_mu)


print(recon.shape)
fig, ax = plt.subplots(1, 8)
for i in range(8):
    ax[i].imshow(recon[i].permute(1, 2, 0).detach().numpy())

plt.show()


