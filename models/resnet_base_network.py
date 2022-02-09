import torchvision.models as models
import torch
from models.mlp_head import MLPHead


class ResNet18(torch.nn.Module):
    def __init__(self, name, hidden_dim, proj_size):
        super(ResNet18, self).__init__()
        if name == 'resnet18':
            resnet = models.resnet18(pretrained=False)
        elif name == 'resnet50':
            resnet = models.resnet50(pretrained=False)

        self.encoder = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.projetion = MLPHead(in_channels=resnet.fc.in_features, 
                    mlp_hidden_size = hidden_dim, projection_size = proj_size)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projetion(h)
