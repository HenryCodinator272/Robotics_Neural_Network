import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class ResnetClass(nn.Module):
    def __init__(self, classes, additional_layers = 1, fpn_back_bone = True):

        super().__init__()

        self.fpn_back_bone = fpn_back_bone
        self.additional_layers = additional_layers
        self.classes = classes

        if self.fpn_back_bone is True:
            self.backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=ResNet50_Weights.IMAGENET1K_V1)
            for params in self.backbone.parameters():
                params.requires_grad = False

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        def make_layer():
            output = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                          nn.BatchNorm2d(256),
                          nn.ReLU())
            return output


        self.training_head = nn.Sequential(*[make_layer() for n in range(self.additional_layers)],
                                           nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU(),
                                           nn.Conv2d(256, self.classes, kernel_size=1))
        if self.fpn_back_bone is False:
            self.initiation = nn.Sequential(nn.Conv2d(3, 256, 3, 1, 1),)

    def forward(self, rgb_tensor):

        if self.fpn_back_bone is True:
            resnet_maps = self.backbone(rgb_tensor)['0']
            inter = self.training_head(resnet_maps)
            output = self.upsample(inter)
        else:
            inter = self.initiation(rgb_tensor)
            output = self.training_head(inter)

        return output

