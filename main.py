import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn.functional as F
import torch.nn as nn
import torch

from generator import Generator as G
from discriminator import Discriminator as D



def main():
  os.makedirs('output', exist_ok=True)

  # Defining the shape of image
  img_shape=(1, 28, 28)

  loss_func = torch.nn.BCELoss()

  generator = G(img_shape=img_shape)
  discriminator = D()

  dataset = torch.utils.data.DataLoader(
    datasets.MNIST('data/', train=True, download=True,
                    transform=transforms.Compose([
                      transforms.ToTensor(),
                      transforms.Normalize((0.5), (0.5))
                    ])), batch_size=64, shuffle=True,
  )

  if torch.cuda.is_available():
      generator.cuda()
      discriminator.cuda()
      loss_func.cuda()

  optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002,betas=(0.4,0.999))
  optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002,betas=(0.4,0.999))

  Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


  for epoch in range(20):
      for i, (imgs, _) in enumerate(dataset):

          #ground truths
          val = Tensor(imgs.size(0), 1).fill_(1.0)
          fake = Tensor(imgs.size(0), 1).fill_(0.0)

          real_imgs = imgs.cuda()


          optimizer_G.zero_grad()

          gen_input = Tensor(np.random.normal(0, 1, (imgs.shape[0],100)))

          gen = generator(gen_input)

          #measure of generator's ability to fool discriminator
          g_loss = loss_func(discriminator(gen), val)

          g_loss.backward()
          optimizer_G.step()

          optimizer_D.zero_grad()

          real_loss = loss_func(discriminator(real_imgs), val)
          fake_loss = loss_func(discriminator(gen.detach()), fake)
          d_loss = (real_loss + fake_loss) / 2

          d_loss.backward()
          optimizer_D.step()

          print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, 20, i, len(dataset),
                                                              d_loss.item(), g_loss.item()))
          
          total_batch = epoch * len(dataset) + i
          if total_batch % 400 == 0:
              save_image(gen.data[:25], 'output/%d.png' % total_batch, nrow=5, normalize=True)

if __name__=="__main__":
  main()