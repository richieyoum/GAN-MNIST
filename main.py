import torch
import os
from torch import nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from models import generate_noise, Generator, Discriminator, gen_loss, disc_loss
from utils import show_tensor_images, save_tensor_images
from tqdm.auto import tqdm

# set parameters
criterion = nn.BCEWithLogitsLoss()
epochs = 200
noise_dim = 64
display_step = 1000
batch_size=128
lr=1e-5
device=['cuda' if torch.cuda.device_count()>0 else 'cpu'][0]
visualize = False
save = True
image_save_path = './images/'
model_save_path = './model_weights/'

# ensure that images and model_weights folders exist
if not os.path.exists(image_save_path):
    os.mkdir(image_save_path)
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

# load MNIST data
dataloader = DataLoader(
    MNIST('.', download=True, transform=transforms.ToTensor()),
    batch_size=batch_size,
    shuffle=True
)

# define models and their optimizers
gen = Generator(noise_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

# training GAN
curr_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
gen_loss_ = False

for epoch in range(epochs):
    # getting images in a batch from the dataloader
    for real_imgs, _ in tqdm(dataloader):
        curr_batch_size = len(real_imgs)

        # flatten the images, as working with linear model
        real_imgs = real_imgs.view(curr_batch_size, -1).to(device)

        # discriminator is trained first
        # make sure gradients are set to 0 beforehand
        disc_opt.zero_grad()
        disc_loss_ = disc_loss(gen, disc, criterion, real_imgs, curr_batch_size, noise_dim, device)
        # update gradients
        disc_loss_.backward()
        # update optimizer
        disc_opt.step()

        # same steps apply for generator
        gen_opt.zero_grad()
        gen_loss_ = gen_loss(gen, disc, criterion, curr_batch_size, noise_dim, device)
        gen_loss_.backward()
        gen_opt.step()

        
        mean_generator_loss += gen_loss_.item() / display_step
        mean_discriminator_loss += disc_loss_.item() / display_step       
        if curr_step % display_step == 0 and curr_step > 0:
            print(f"Step {curr_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            noise = generate_noise(curr_batch_size, noise_dim, device)
            fake_imgs = gen(noise)
            if visualize:
                show_tensor_images(fake_imgs)
                show_tensor_images(real_imgs)
            if save:
                save_tensor_images(os.path.join(image_save_path,f"step_{curr_step}_epoch_{epoch}_actual"), real_imgs)
                save_tensor_images(os.path.join(image_save_path+f"step_{curr_step}_epoch_{epoch}_generated"), fake_imgs)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        curr_step += 1

if save:
    torch.save(gen.state_dict(), model_save_path)
    torch.save(disc.state_dict(), model_save_path)