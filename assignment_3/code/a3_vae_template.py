import os
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
import numpy as np

from datasets.bmnist import bmnist


class Encoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        self.linear1 = nn.Linear(784, hidden_dim)
        self.linear21 = nn.Linear(hidden_dim, z_dim)
        self.linear22 = nn.Linear(hidden_dim, z_dim)

    def forward(self, input):
        """
        Perform forward pass of encoder.

        Returns mean and std with shape [batch_size, z_dim]. Make sure
        that any constraints are enforced.
        """
        out = F.relu(self.linear1(input))
        mean, logvar = self.linear21(out), self.linear22(out)
        return mean, logvar


class Decoder(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        self.linear1 = nn.Linear(z_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 784)

    def forward(self, input):
        """
        Perform forward pass of decoder.

        Returns mean with shape [batch_size, 784].
        """
        out = F.relu(self.linear1(input))
        mean = torch.sigmoid(self.linear2(out))
        return mean


class VAE(nn.Module):

    def __init__(self, hidden_dim=500, z_dim=20):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(hidden_dim, z_dim)
        self.decoder = Decoder(hidden_dim, z_dim)

    def forward(self, input):
        """
        Given input, perform an encoding and decoding step and return the
        negative average elbo for the given batch.
        """
        mean, logvar = self.encoder(input.view(-1, 784))
        z = self.reparameterize(mean, logvar)
        input_recon = self.decoder(z)
        average_negative_elbo = self.elbo_loss_function(input, input_recon, mean, logvar)
        return average_negative_elbo

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mean + std*eps

    def elbo_loss_function(self, input, input_recon, mean, logvar):
        BCE = F.binary_cross_entropy(input_recon, input.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return BCE + KLD

    def sample(self, n_samples):
        """
        Sample n_samples from the model. Return both the sampled images
        (from bernoulli) and the means for these bernoullis (as these are
        used to plot the data manifold).
        """
        z = torch.randn((n_samples, self.z_dim))
        im_means = self.decoder(z)
        sampled_ims = torch.bernoulli(im_means)
        return sampled_ims, im_means


def epoch_iter(model, data, optimizer):
    """
    Perform a single epoch for either the training or validation.
    use model.training to determine if in 'training mode' or not.

    Returns the average elbo for the complete epoch.
    """
    elbo_loss = 0
    data_length = len(data)
    for _, batch in enumerate(data):
        batch.to(device)
        if model.training:
            optimizer.zero_grad()
        elbo = model(batch)
        if model.training:
            elbo.backward()
            optimizer.step()
        elbo_loss += elbo.item()
    return elbo_loss / data_length


def run_epoch(model, data, optimizer):
    """
    Run a train and validation epoch and return average elbo for each.
    """
    traindata, valdata = data

    model.train()
    train_elbo = epoch_iter(model, traindata, optimizer)

    model.eval()
    with torch.no_grad():
        val_elbo = epoch_iter(model, valdata, optimizer)

    return train_elbo, val_elbo


def save_elbo_plot(train_curve, val_curve, filename):
    plt.figure(figsize=(12, 6))
    plt.plot(train_curve, label='train elbo')
    plt.plot(val_curve, label='validation elbo')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('ELBO')
    plt.tight_layout()
    plt.savefig(filename)


def save_samples_plot(model, epoch, n_samples=25):
    with torch.no_grad():
        sampled_ims, _ = model.sample(n_samples)
        images = sampled_ims.view(n_samples, 1, 28, 28)
        save_image(images.cpu(),
                'results_vae/samples_{}.png'.format(str(epoch)),
                nrow=5, normalize=True)


def save_manifold_plot(model):
    with torch.no_grad():
        steps = 20
        density_point = torch.linspace(0, 1, steps)
        # Basically use adaptation of torch.distributions.icdf here for manifold z's
        z_tensors = [torch.erfinv(2 * torch.tensor([x, y]) - 1) * np.sqrt(2) for x in density_point for y in density_point]
        z = torch.stack(z_tensors).to(device)
        im_means = model.decoder(z).view(-1, 1, 28, 28)
        image = make_grid(im_means, nrow=steps)
        save_image(image.cpu(), 'results_vae/manifold.png')


def main():
    data = bmnist()[:2]  # ignore test split
    model = VAE(z_dim=ARGS.zdim)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    train_curve, val_curve = [], []
    for epoch in range(ARGS.epochs):
        elbos = run_epoch(model, data, optimizer)
        train_elbo, val_elbo = elbos
        train_curve.append(train_elbo)
        val_curve.append(val_elbo)
        print(f"[Epoch {epoch}] train elbo: {train_elbo} val_elbo: {val_elbo}")

        # --------------------------------------------------------------------
        #  Add functionality to plot samples from model during training.
        #  You can use the make_grid functioanlity that is already imported.
        # --------------------------------------------------------------------
        save_samples_plot(model, epoch)

    # --------------------------------------------------------------------
    #  Add functionality to plot plot the learned data manifold after
    #  if required (i.e., if zdim == 2). You can use the make_grid
    #  functionality that is already imported.
    # --------------------------------------------------------------------
    if ARGS.zdim == 2:
        save_manifold_plot(model)

    save_elbo_plot(train_curve, val_curve, 'elbo.pdf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=40, type=int,
                        help='max number of epochs')
    parser.add_argument('--zdim', default=20, type=int,
                        help='dimensionality of latent space')
    parser.add_argument('--device', default='cpu', type=str,
                        help='device')

    ARGS = parser.parse_args()
    device = torch.device(ARGS.device)

    os.makedirs('/results_vae', exist_ok=True)

    main()
