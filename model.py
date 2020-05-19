from torch import nn

# Generator Model
class Generator(nn.Module):
  def __init__(self):
    super().__init__()
    self.main = nn.Sequential(
        nn.ConvTranspose2d( 100, 128 * 8, 4, 1, 0, bias=False),
        nn.BatchNorm2d(128 * 8),
        nn.ReLU(True),
            
        nn.ConvTranspose2d(128 * 8, 128 * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(128 * 4),
        nn.ReLU(True),
            
        nn.ConvTranspose2d( 128 * 4, 128 * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(128 * 2),
        nn.ReLU(True),
            
        nn.ConvTranspose2d( 128 * 2, 128, 4, 2, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU(True),
           
        nn.ConvTranspose2d( 128, 3, 4, 2, 1, bias=False),
        nn.Tanh() 
    )
    
  def forward(self, x):
    return self.main(x)

# Discriminator Model
class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.main = nn.Sequential(
        nn.Conv2d(3, 64, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64 * 2),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64 * 4),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(64 * 8),
        nn.LeakyReLU(0.2, inplace=True),
        
        nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
        nn.Sigmoid()
    )
    
  def forward(self, x):
    return self.main(x)