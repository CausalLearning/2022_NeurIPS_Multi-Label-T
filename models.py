#used Resnet50
import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
import sys

class Resnet(nn.Module):
    def __init__(self, model, num_classes):
        super(Resnet, self).__init__()

        modules = list(model.children())[:-1]
        self.features = nn.Sequential(*modules)
        
        self.linear = nn.Linear(2048, num_classes)

    def forward(self, feature, get_feature=False):
        feature = self.features(feature)
        
        gf = self.linear(feature.view(feature.size(0), -1))
        gf = gf.view(gf.size(0), -1)         
        
        if(get_feature):
            return feature
        else:
            return gf
        
def get_resnet50(num_classes, pretrained=True):
    model = models.resnet50(pretrained=pretrained)
    return Resnet(model, num_classes)
