import torch
from torch import nn

def generate_noise(n_samples, noise_dim, device='cpu'):
    """
    create noise vectors for generator input
    params:
        n_samples (int): number of samples to generate
        noise_dim (int): noise vector dimension
        device (str): choosing cpu or gpu usage; 'cpu' or 'cuda'
    """
    return torch.randn(n_samples, noise_dim, device=device)

def generator_block(input_dim, output_dim):
    """
    block of layer for generator
    params:
        input_dim (int): input dimension
        output_dim (int): desired output dimension
    returns:
        sequential layer for generator with linear layer, 1D batchnorm and ReLU activation
    """
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.ReLU()
    )

def discriminator_block(input_dim, output_dim):
    """
    block of layer for discriminator
    params:
        input_dim (int): input dimension
        output_dim (int): desired output dimension
    returns:
        sequential layer for discriminator with linear layer and LeakyReLU activation to resolve vanishing gradient problem
    """
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.LeakyReLU(.2)
    )

class Generator(nn.Module):
    """
    class for generator model of the GAN
    params:
        noise_dim (int): dimension of the noise vector
        output_dim (int): desired output dimension
        hidden_dim (int): hidden dimension to be used as a base unit
    returns:
        Generator model
    """
    def __init__(self, noise_dim, output_dim=784, hidden_dim=128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            generator_block(noise_dim, hidden_dim),
            generator_block(hidden_dim, hidden_dim*2),
            generator_block(hidden_dim*2, hidden_dim*4),
            generator_block(hidden_dim*4, hidden_dim*8),
            nn.Linear(hidden_dim*8, output_dim),
            nn.Sigmoid()
        )

    def forward(self, noise):
        """
        forward pass of the generator
        params:
            noise: noise vector (n_samples, noise_dim)
        returns:
        """
        return self.gen(noise)


class Discriminator(nn.Module):
    """
    class for discriminator model of the GAN
    params:
        input_dim (int): dimension of the input, which is the output of the generator. Default set to 784 for MNIST (28 x 28)
        hidden_dim (int): hidden dimension to be used as a base unit
    returns:
        Discriminator model
    """
    def __init__(self, input_dim=784, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            discriminator_block(input_dim, hidden_dim*4),
            discriminator_block(hidden_dim*4, hidden_dim*2),
            discriminator_block(hidden_dim*2, hidden_dim),
            # boolean output for discriminator - Fake or Real
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image):
        """
        forward pass of the discriminator
        params:
            image: image tensor (output of the generator)
        returns:
        """
        return self.disc(image)


def disc_loss(gen, disc, criterion, real_imgs, num_images, noise_dim, device):
    """
    calculate loss of the discriminator
    params:
        gen: generator model
        disc: discriminator model
        criterion: loss function to be used when comparing against generated and real images
        real_imgs: real images in the training set
        num_images: number of images to be generated
        noise_dim (int): dimension of the noise vector
        device (str): choosing cpu or gpu usage; 'cpu' or 'cuda'
    returns:
        disc_loss (float): average loss of the discriminator model
    """
    # generate noise
    noise = generate_noise(num_images, noise_dim, device)

    # generate images from the noise. detaching to not calculate gradient on generator
    fake_imgs = gen(noise).detach()

    # loss for the fake (generated) images
    y_fake_pred = disc(fake_imgs)
    y_fake = torch.zeros_like(y_fake_pred)
    fake_loss = criterion(y_fake_pred, y_fake)

    # loss for the real images
    y_real_pred = disc(real_imgs)
    y_real = torch.ones_like(y_real_pred)
    real_loss = criterion(y_real_pred, y_real)

    # average loss
    disc_loss = (fake_loss + real_loss)/2
    return disc_loss

def gen_loss(gen, disc, criterion, num_images, noise_dim, device):
    """
    calculate loss of the generator
    params:
        gen: generator model
        disc: discriminator model
        criterion: loss function to be used when comparing against generated and real images
        num_images: number of images to be generated
        noise_dim (int): dimension of the noise vector
        device (str): choosing cpu or gpu usage; 'cpu' or 'cuda'
    returns:
        gen_loss (float): average loss of the generator model
    """
    noise_vec = generate_noise(num_images, noise_dim, device)
    fake_imgs = gen(noise_vec)

    # discriminator prediction
    disc_pred = disc(fake_imgs)
    
    # generator "argues" they are real images, hence populate the y's with ones
    y = torch.ones_like(disc_pred)
    gen_loss = criterion(disc_pred, y)
    return gen_loss
