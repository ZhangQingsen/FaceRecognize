import torch.nn as nn
import torch.nn.functional as F

def conv_bn(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6()
    )


def conv_dw(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU6(),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(),
    )


class ConvNet(nn.Module):
    def __init__(self, num_classes=None, mode="train"):

        super(ConvNet, self).__init__()
        self.stage1 = nn.Sequential(
            # 160,160,3 -> 80,80,32
            conv_bn(3, 32, 2),
            # 80,80,32 -> 80,80,64
            conv_dw(32, 64, 1),

            # 80,80,64 -> 40,40,128
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),

            # 40,40,128 -> 20,20,256
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
        )
        self.stage2 = nn.Sequential(
            # 20,20,256 -> 10,10,512
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
        )
        self.stage3 = nn.Sequential(
            # 10,10,512 -> 5,5,1024
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        # self.avg = nn.AdaptiveAvgPool1d(1)
        self.max = nn.MaxPool2d((1, 1))
        self.fc = nn.Linear(1024, 1000)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.Dropout = nn.Dropout(0.5)
        self.Bottleneck = nn.Linear(1024, 128, bias=False)
        self.last_bn = nn.BatchNorm1d(128, eps=0.001, momentum=0.1, affine=True)

        if mode == "train":
            self.classifier = nn.Linear(128, num_classes)

    def backbone(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        # x = self.max(x)
        # x = x.view(-1, 1024)
        # print(x.shape)
        # x = self.fc(x)
        return x

    def forward(self, x, mode="predict"):
        
        x = self.backbone(x)
        # print(x.shape)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.Dropout(x)
        x = self.Bottleneck(x)

        if mode == 'predict':
            x = self.last_bn(x)
            x = F.normalize(x, p=2, dim=1)
            return x
        before_normalize = self.last_bn(x)

        x = F.normalize(before_normalize, p=2, dim=1)
        cls = self.classifier(before_normalize)
        return x, cls

