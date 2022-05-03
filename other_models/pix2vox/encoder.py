# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import torch
import torchvision.models

from pix2vox.attention import AttentionConv, AttentionStem

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.cfg = cfg

        # Layer Definition
        resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet = torch.nn.Sequential(*[
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3,
            resnet.layer4
        ])[:6]
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            AttentionConv(512, 512, kernel_size=7, padding=3, groups=8),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        """
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 384, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(384),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        """

    def forward(self, rendering_images):
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        # rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = rendering_images.permute(0, 1, 4, 2, 3).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []

        for img in rendering_images:
            features = self.resnet(img.squeeze(dim=0))
            # print(features.size())    # torch.Size([batch_size, 512, 28, 28])
            features = self.layer1(features)
            # print(features.size())    # torch.Size([batch_size, 512, 28, 28])
            features = self.layer2(features)
            # print(features.size())    # torch.Size([batch_size, 256, 14, 14])
            features = self.layer3(features)
            # print(features.size())    # torch.Size([batch_size, 256, 7, 7])
            features = self.layer4(features)
            image_features.append(features)

        image_features = torch.stack(image_features).contiguous()
        # image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
        # print(image_features.size())  # torch.Size([batch_size, n_views, 256, 7, 7])
        return image_features
