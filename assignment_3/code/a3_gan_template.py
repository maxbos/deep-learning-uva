import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets

import pandas as pd


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.linear1 = nn.Linear(args.latent_dim, 128)
        self.linear2 = nn.Linear(128, 256)
        self.bnorm1 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, 512)
        self.bnorm2 = nn.BatchNorm1d(512)
        self.linear4 = nn.Linear(512, 1024)
        self.bnorm3 = nn.BatchNorm1d(1024)
        self.linear5 = nn.Linear(1024, 784)

    def forward(self, z):
        x = F.leaky_relu(self.linear1(z), .2)
        x = F.leaky_relu(self.bnorm1(self.linear2(x)), .2)
        x = F.leaky_relu(self.bnorm2(self.linear3(x)), .2)
        x = F.leaky_relu(self.bnorm3(self.linear4(x)), .2)
        return torch.tanh(self.linear5(x))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, img):
        x = F.leaky_relu(self.linear1(img), .2)
        x = F.leaky_relu(self.linear2(x), .2)
        return torch.sigmoid(self.linear3(x))


def train(dataloader, discriminator, generator, optimizer_G, optimizer_D, criterion, device):
    # Set the labels for classifying real and fake images
    real_label = 1
    fake_label = 0
    # Initialize the evaluation metrics
    eval_results = []

    for epoch in range(args.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            if torch.cuda.is_available():
                imgs.cuda()

            # Reshape the images to be one dimensional vectors
            batch_size = imgs.shape[0]
            imgs = imgs.view(batch_size, -1)

            # Train Discriminator
            # -------------------
            optimizer_D.zero_grad()
            ## Train the Discriminator by forward passing real images
            # Forward pass the images through the Discriminator network
            output = discriminator(imgs).view(-1)
            # Construct a vector of real labels for each sample in the batch
            label = torch.full((batch_size,), real_label, device=device)
            # Calculate the loss on the real images
            loss_D_real = criterion(output, label)
            # Calculate the gradients
            loss_D_real.backward()
            # The average probability of discriminator labelling the real images as real
            D_x = output.mean().item()

            ## Train the Discriminator by forward passing fake images
            # Generate fake images from noise
            noise = torch.randn(batch_size, args.latent_dim, device=device)
            fake_imgs = generator(noise)
            # Forward pass fake images through the Discriminator network
            output = discriminator(fake_imgs.detach()).view(-1)
            # Calculate the loss on the fake images
            label.fill_(fake_label)
            loss_D_fake = criterion(output, label)
            # Calculate the gradients
            loss_D_fake.backward()
            D_G_z1 = output.mean().item()
            # The loss of the Discriminator is calculated as the sum of the loss on the
            # real images and the loss on the fake images
            loss_D = loss_D_real + loss_D_fake
            # Update the parameters of the Discriminator network
            optimizer_D.step()

            # Train Generator
            # ---------------
            optimizer_G.zero_grad()
            # We previously generated the fake images, forward pass these through the Discriminator
            # network
            output = discriminator(fake_imgs).view(-1)
            # Since, as the Generator, we want the fake images to be classified as real by
            # the discriminator, we train on real labels
            label.fill_(real_label)
            loss_G = criterion(output, label)
            # Calculate the gradients
            loss_G.backward()
            # The average probability of the discriminator labelling the fake images as real
            D_G_z2 = output.mean().item()
            # Update the parameters of the Generator network
            optimizer_G.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, args.n_epochs, i, len(dataloader),
                    loss_D.item(), loss_G.item(), D_x, D_G_z1, D_G_z2))

                eval_results.extend([
                    [i, 'loss_D', loss_D.item()],
                    [i, 'loss_G', loss_G.item()],
                ])

            # Save Images
            # -----------
            batches_done = epoch * len(dataloader) + i
            if batches_done % args.save_interval == 0:
                gen_imgs = fake_imgs.view(batch_size, 1, 28, 28)
                # You can use the function save_image(Tensor (shape Bx1x28x28),
                # filename, number of rows, normalize) to save the generated
                # images, e.g.:
                save_image(gen_imgs[:25],
                           'images/{}.png'.format(batches_done),
                           nrow=5, normalize=True)

    print('Done training')

    out_dir = './out/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Store the evaluation results
    eval_df = pd.DataFrame(eval_results, columns=['label', 'step', 'value'])
    eval_df.to_csv('./out/eval_results.csv')


def main():
    # Create output image directory
    os.makedirs('images', exist_ok=True)

    # load data
    dataloader = torch.utils.data.DataLoader(
        datasets.MNIST('./data/mnist', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,),
                                                (0.5,))])),
        batch_size=args.batch_size, shuffle=True)

    # Initialize the device that will be used
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Initialize models and optimizers
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr)

    # Start training
    train(dataloader, discriminator, generator, optimizer_G, optimizer_D, criterion, device)

    # You can save your generator here to re-use it to generate images for your
    # report, e.g.:
    torch.save(generator.state_dict(), "mnist_generator.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='dimensionality of the latent space')
    parser.add_argument('--save_interval', type=int, default=500,
                        help='save every SAVE_INTERVAL iterations')
    args = parser.parse_args()

    main()
