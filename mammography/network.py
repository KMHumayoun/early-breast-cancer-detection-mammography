import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class ScreeningModel(nn.Module):
    """Shared feature extractor, spatial pooling and ordered CC/MLO fusion."""
    def __init__(self, nviews, pretrained=True):
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        self.spatial = nn.Conv2d(512, 1, 1)
        self.head = nn.Sequential(nn.LayerNorm(512*nviews), nn.Dropout(.2),
                                  nn.Linear(512*nviews, 1))
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).reshape(1,3,1,1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).reshape(1,3,1,1))

    def forward(self, pixels):
        batch, views, _, height, width = pixels.shape
        x = pixels.reshape(batch*views, 1, height, width).expand(-1,3,-1,-1)
        features = self.encoder((x-self.mean)/self.std)
        weights = self.spatial(features).flatten(2).softmax(dim=-1)
        pooled = (features.flatten(2)*weights).sum(dim=-1)
        return self.head(pooled.reshape(batch, -1)).squeeze(-1)
