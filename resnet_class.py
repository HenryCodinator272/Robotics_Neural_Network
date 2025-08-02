import torch.nn as nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class ResnetClass(nn.Module):
    def __init__(self, classes):

        super().__init__()

        self.backbone = resnet_fpn_backbone('resnet50', pretrained = True)

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        for params in self.backbone.parameters():
            params.requires_grad = False

        self.classes = classes
        self.training_head = nn.Sequential(nn.Conv2d(256, 256, 3, 1, 1),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(),
                                      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm2d(256),
                                      nn.ReLU(),
                                      nn.Conv2d(256, classes, kernel_size=1),
                                      self.upsample)

    def forward(self, rgb_tensor):

        resnet_maps = self.backbone(rgb_tensor)['0']
        output = self.training_head(resnet_maps)

        return output

