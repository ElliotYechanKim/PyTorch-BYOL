from torch import nn

class MLPHead(nn.Module):
    def __init__(self, in_channels, name):
        super(MLPHead, self).__init__()

        if name == 'resnet18':
            mlp_hidden_size = 512
            projection_size = 128
        elif name == 'resnet50':
            mlp_hidden_size = 2048
            projection_size = 256

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.net(x)

    def __iter__(self):
        for m in self.net:
            yield m
    
    def __len__(self):
        return len(self.net)

    def to_sequential(self) -> nn.Sequential :
        return self.net