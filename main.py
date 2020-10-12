import torch
import os
from torch import nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from models import generate_noise, Generator, Discriminator, DC_Generator, DC_Discriminator, gen_loss, disc_loss
from utils import show_tensor_images, save_tensor_images
from tqdm.auto import tqdm

# configs
criterion = nn.BCEWithLogitsLoss()
epochs = 50
noise_dim = 64
display_step = 2000
batch_size=128
lr=2e-4
# for momentum
beta_1 = .5
beta_2 = .999
device=['cuda' if torch.cuda.device_count()>0 else 'cpu'][0]
visualize = False
save = True
image_save_path = './images/'
model_save_path = './model_weights/'
architecture = 'dcgan' # or simplegan

# ensure that images and model_weights folders exist
if not os.path.exists(image_save_path):
    os.mkdir(image_save_path)
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

# define models and their optimizers
if architecture=='simplegan':
    gen = Generator(noise_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc = Discriminator().to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    transform = transforms.ToTensor()
elif architecture=='dcgan':
    gen = DC_Generator(noise_dim).to(device)
    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
    disc = DC_Discriminator().to(device)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))
    transform = transforms.Compose([
        transforms.ToTensor(),
        # make image values between -1 and 1 for tanh
        transforms.Normalize((0.5,), (0.5,))
    ])

    # initialize weights to the normal distribution with a mean of 0 and std of 0.02
    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if isinstance(m, nn.BatchNorm2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
    gen = gen.apply(weights_init)
    disc = disc.apply(weights_init)
else:
    raise ValueError("Please enter valid architecture type. Currently, 'simplegan' and 'dcgan' are supported.")

# load MNIST data
dataloader = DataLoader(
    MNIST('.', download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True
)

# training GAN
curr_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0
gen_loss_ = False

print("Starting training...")

for epoch in range(epochs):
    # getting images in a batch from the dataloader
    for real_imgs, _ in tqdm(dataloader):
        curr_batch_size = len(real_imgs)

        if architecture=='simplegan':
            # flatten the images, as working with linear model
            real_imgs = real_imgs.view(curr_batch_size, -1).to(device)
        elif architecture=='dcgan':
            # for dcgan, just make sure image and weights are both on the same device
            real_imgs = real_imgs.to(device)

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
                save_tensor_images(os.path.join(image_save_path,f"{architecture}_step_{curr_step}_epoch_{epoch}_actual"), real_imgs)
                save_tensor_images(os.path.join(image_save_path+f"{architecture}_step_{curr_step}_epoch_{epoch}_generated"), fake_imgs)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
        curr_step += 1

if save:
    torch.save(gen.state_dict(), os.path.join(model_save_path, f'{architecture}_gen.pt'))
    torch.save(disc.state_dict(), os.path.join(model_save_path, f'{architecture}_disc.pt'))

print("Training complete!")