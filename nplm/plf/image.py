import torch.nn as nn
import torch.nn.init as init
from torchvision.models import resnet101, resnet50

r50_in_dim = 2048
r101_in_dim = 2048


class ResNetFeaturesLC(nn.Module):
    def __init__(self, in_features=r101_in_dim, target_dim=2, intermediate_1=1024, intermediate_2=1024):
        super(ResNetFeaturesLC, self).__init__()

        self.target_fc = nn.Sequential(
            nn.Linear(in_features, intermediate_1),
            nn.BatchNorm1d(num_features=intermediate_1),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(intermediate_1, intermediate_2),
            nn.BatchNorm1d(num_features=intermediate_2),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(intermediate_2, target_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                init.normal_(m.bias.data)

    def forward(self, x):
        return self.target_fc(x)


class ResNetAttr(nn.Module):
    def __init__(self, backbone='r50',
                 pretrained=True,
                 target_dim=2,
                 intermediate_1=1024,
                 intermediate_2=1024):
        super(ResNetAttr, self).__init__()

        self.backbone_model = resnet50(pretrained=pretrained) if backbone == 'r50' else resnet101(pretrained=pretrained)

        in_features = self.backbone_model.fc.in_features

        self.target_fc = nn.Sequential(
            nn.Linear(in_features, intermediate_1),
            nn.BatchNorm1d(num_features=intermediate_1),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(intermediate_1, intermediate_2),
            nn.BatchNorm1d(num_features=intermediate_2),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(intermediate_2, target_dim),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                init.normal_(m.bias.data)

    def forward(self, x):
        return self.target_fc(self.backbone_model(x))
