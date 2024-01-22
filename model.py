import torch.nn as nn
from torchvision import models

class MobNetv2_custom_classes(nn.Module):
  def __init__(self):
    super(MobNetv2_custom_classes,self).__init__()
    self.mobnetv2_model = models.mobilenet_v2(pretrained=True)
    self.add_class_layer = nn.Sequential(
                                            nn.Linear(1000,4),
                                            nn.Softmax(dim=1)
                                        )

  def forward(self,x):
    x = self.mobnetv2_model(x)
    x = self.add_class_layer(x)
    return x