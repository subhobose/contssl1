import kornia.augmentation as K
import torch
import torch.nn as nn
t = lambda x: torch.tensor(x); p = lambda x: nn.Parameter(t(x))
torch.manual_seed(42);

class CJAugmentationPipeline(nn.Module):
  def __init__(self):
    super(CJAugmentationPipeline, self).__init__()

    self.mu = torch.Tensor([0.485, 0.456, 0.406])
    self.sigma = torch.Tensor([0.229, 0.224, 0.225])
    
    self.jitter = K.ColorJitter(p([0.4,0.4]), p([0.4,0.4]), p([0.4,0.4]), p([0.1,0.1]), p=0.8)
    self.normalize = K.Normalize(self.mu, self.sigma)
    
  def forward(self, input):
    input = self.jitter(input)
    input = self.normalize(input)
    return input