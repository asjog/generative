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
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime 
from monai.networks.nets.varautoencoder import VarAutoEncoder

IMG_SHAPE = (128, 128)
LATENT_DIM = 10
BATCH_SIZE = 64 

BCELoss = torch.nn.BCELoss(reduction="sum")
MSELOSS = torch.nn.MSELoss(reduction="sum")

def loss_function(recon_x, x, mu, log_var, beta):
    bce = MSELOSS(recon_x, x)
    kld = -0.5 * beta * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce, kld


class vae_encoder(nn.Module):
    def __init__(self, input_shape, latent_dim, in_channels=1) -> None:
        super().__init__()
        # initialize the conv layers 
        self.num_layers = 4
        self.start_num_filters = 8
        self.conv_list = []
        prev_in_channels = in_channels
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
        x = x.view(x.size(0), -1)
        mu = self.mu(x)
        logsigma = self.logsigma(x)
        # sample z from N(z | mu, sigma)
        # this means sample w (5, 4096) from N(w |0, 1) and mu (5, 4096) + w * sqrt(sigma) (5, 4096)

        z = torch.normal(mu, torch.exp(0.5*logsigma))

        return z, mu, logsigma

class vae_decoder(nn.Module):
    def __init__(self, latent_dim, out_channels = 1) -> None:
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
        

        self.conv_list.append(nn.Conv2d(in_channels=8, out_channels=out_channels, kernel_size=3))
        self.conv_list.append(pad_layer)

        self.cnn = nn.Sequential(*self.conv_list)

    def forward(self, x):
        # x shape is 1x2304
        x = self.updim(x)
        x = x.view(x.size(0), 64, 8, 8)
        x = self.cnn(x)
        return x 
    


class vae(nn.Module):
    def __init__(self, input_shape, latent_dim) -> None:
        super().__init__()
        self.encoder = vae_encoder(input_shape, latent_dim)
        self.decoder = vae_decoder(latent_dim=latent_dim, out_channels=input_shape[0])

    


def sq_l2_loss(x, xhat, sigma_dec=1):
    # sql2_loss = torch.sum((x - xhat)**2, axis=(1, 2, 3))/sigma_dec
    # or 
    mse_unreduced_loss = torch.nn.MSELoss(reduction="none")
    mse_loss = mse_unreduced_loss(x, xhat)
    mse_loss = torch.sum(mse_loss, axis=(1, 2, 3))
    # average over the batch
    mse_loss = torch.mean(mse_loss)
    return mse_loss/sigma_dec
    

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
    mse = sq_l2_loss(x, xhat)
    kl_loss = kl_divergence_loss(mu, sigma)
    return mse, 1.0*kl_loss

class CelebADataset(Dataset):
    def __init__(self, img_files) -> None:
        self.img_files = img_files 
    def __getitem__(self, index):
        img_file = self.img_files[index]
        img = PIL.Image.open(img_file)
        # resize the image to 128x128
        img = img.resize(IMG_SHAPE)
        img = ToTensor()(img)
        return img
    def __len__(self):
        return len(self.img_files)
    

class MNISTDataset(Dataset):
    def __init__(self, img_files) -> None:
        self.img_files = img_files 
    def __getitem__(self, index):
        img_file = self.img_files[index]
        img = PIL.Image.open(img_file)
        img = img.resize(IMG_SHAPE)
        img = ToTensor()(img)
        return img
   
    def __len__(self):
        return len(self.img_files)
    

mps_available = torch.backends.mps.is_available()
device = torch.device("mps" if mps_available else "cpu")
print(f"Using device: {device}")


img_dir = "/Users/asjog/VAE-research/data/celeba/celeba/img_align_celeba/"
celeba_img_files = sorted(glob.glob(img_dir + "*.jpg"))

celeba_dataset = CelebADataset(celeba_img_files[0:100])
celeba_dataloader = DataLoader(celeba_dataset, batch_size=BATCH_SIZE, shuffle=True)


mnist_dir = "/Users/asjog/VAE-research/data/mnist/MNIST/png/training/"
mnist_img_files = sorted(glob.glob(mnist_dir + "*/*.png"))

mnist_dataset = MNISTDataset(mnist_img_files[::5])
mnist_dataloader = DataLoader(mnist_dataset, batch_size=BATCH_SIZE, shuffle=True)



# # plot a single batch of images with plt imshow
# for batch in mnist_dataloader:
#     print(batch.shape)
#     # plot all images of the batch in a subplots 
#     fig, ax = plt.subplots(1, 8)
#     for i in range(8):
#         ax[i].imshow(batch[i].permute(1, 2, 0))

#     plt.show()
#     break


# enc = vae_encoder((3, 128, 128), 4096)
# dec = vae_decoder(4096, (3, 128, 128))

# batch_z, batch_mu, batch_logsima = enc(batch)
# recon = dec(batch_mu)


# print(recon.shape)
# fig, ax = plt.subplots(1, 8)
# for i in range(8):
#     ax[i].imshow(recon[i].permute(1, 2, 0).detach().numpy())

# plt.show()


# write training loop
# define the model
# celeb_vae = vae((1, 128, 128), 2)
# celeb_vae.to(device)

monai_vae = VarAutoEncoder(spatial_dims=2, in_shape=(1, IMG_SHAPE[0], IMG_SHAPE[1]), out_channels=1, latent_size=LATENT_DIM, channels=[8, 16, 32,], strides=[2, 2, 2,], use_sigmoid=False)
monai_vae.to(device)


# define the optimizer
optimizer = torch.optim.Adam(monai_vae.parameters(), lr=1e-4)
# optimizer = torch.optim.SGD(celeb_vae.parameters(), lr=1e-2)

# initialize a tensorboard summary writer
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S"
                                   )
writer = SummaryWriter("./runs/mnist_vae_{}".format(timestamp))

for epoch in range(100):

    # celeb_vae.train(True)
    monai_vae.train(True)
    running_loss = 0.0
    running_kld = 0.0
    running_mse = 0.0


    for i, batch in enumerate(mnist_dataloader):
        # zero the gradients
        optimizer.zero_grad()

        # forward pass
        batch = batch.to(device)
        # batch_z, batch_mu, batch_logsima = celeb_vae.encoder(batch)
        recon_batch, mu, log_var, _ = monai_vae(batch)
        mse, kld = vae_loss_function(batch, recon_batch, mu, log_var)

        # recon_batch = celeb_vae.decoder(batch_z) # should be batch_z

        # calculate the loss
        
        # mse, kld = loss_function(recon_batch, batch, mu, log_var, 1.0)

        loss = mse + kld


        # loss = vae_loss_function(batch, recon_batch, mu, log_var)


        # backward pass
        loss.backward() 

        # update the weights
        optimizer.step()

        # update the running loss
        running_loss += loss.item()
        running_kld += kld.item()
        running_mse += mse.item()


        if i % 10 == 9:
            last_loss = running_loss/10
            last_kld = running_kld/10
            last_mse = running_mse/10

            # print(f"batch: {i}, loss: {last_loss}")
            # print(f"batch: {i}, kld: {last_kld}")
            # print(f"batch: {i}, mse: {last_mse}")


            writer.add_scalar("training loss", last_loss, i)
        
    epoch_loss = running_loss / len(mnist_dataloader)
    epoch_mse_loss = running_mse / len(mnist_dataloader)
    epoch_kld_loss = running_kld / len(mnist_dataloader)

    print(f"epoch: {epoch}, loss: {epoch_loss}")
    print(f"epoch: {epoch}, mse: {epoch_mse_loss}")
    print(f"epoch: {epoch}, kld: {epoch_kld_loss}")

    writer.add_scalar("epoch loss", epoch_loss, epoch)
    writer.flush()


fig, ax = plt.subplots(2,1)
ax[0].imshow(batch[20,0,:,:].cpu().numpy())
ax[1].imshow(recon_batch.detach()[20,0,:,:].cpu().numpy())
plt.show()

    
