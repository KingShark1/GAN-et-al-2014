import torch.nn.functional as F
import torch.nn as nn

class Generator(nn.Module):
  def __init__(self, img_shape) -> None:
    super(Generator, self).__init__()
    self.img_shape = img_shape
    self.fc1 = nn.Linear(100, 128)
    self.fc2 = nn.Linear(128, 512)
    self.fc3 = nn.Linear(512, 1024)
    self.fc4 = nn.Linear(1024, 28*28)
    self.in1 = nn.BatchNorm1d(128)  
    self.in2 = nn.BatchNorm1d(512) 
    self.in3 = nn.BatchNorm1d(1024) 
  
  def forward(self, x):
    x = F.leaky_relu(self.fc1(x), 0.2) 
    x = F.leaky_relu(self.in2(self.fc2(x)), 0.2)
    x = F.leaky_relu(self.in3(self.fc3(x)), 0.2)
    x = F.tanh(self.fc4(x))
    return x.view(x.shape[0], *self.img_shape)