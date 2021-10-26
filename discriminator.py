import torch.nn.functional as F
import torch.nn as nn

class Discriminator(nn.Module):
  def __init__(self) -> None:
    super(Discriminator, self).__init__()
    self.fc1 = nn.Linear(28*28, 512)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, 128)
    self.fc4 = nn.Linear(128, 1)
    
  def forward(self, x):
    x = x.view(x.size(0), -1)  
    x = F.leaky_relu(self.fc1(x), 0.2)
    x = F.leaky_relu(self.fc2(x), 0.2)
    x = F.leaky_relu(self.fc3(x), 0.2)
    x = F.sigmoid(self.fc4(x))
    return x