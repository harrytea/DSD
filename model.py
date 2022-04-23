import torch
import torch.nn.functional as F
from torch import nn
from resnext.resnext101_regular import ResNeXt101


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64))
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32))

    def forward(self, x):
        block1 = F.relu(self.block1(x) + x, True)
        block2 = self.block2(block1)
        return block2


class AttentionModule(nn.Module):
    def __init__(self):
        super(AttentionModule, self).__init__()
        self.att = nn.Sequential(nn.Conv2d(64, 1, 3, bias=False, padding=1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, x):
        block1 = self.att(x)
        block2 = block1.repeat(1, 32, 1, 1)
        return block2


class DSDNet(nn.Module):
    def __init__(self):
        super(DSDNet, self).__init__()
        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.down4 = nn.Sequential(nn.Conv2d(2048, 512, 3, bias=False, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                                   nn.Conv2d(512, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.down3 = nn.Sequential(nn.Conv2d(1024, 512, 3, bias=False, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
                                   nn.Conv2d(512, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),)
        self.down2 = nn.Sequential(nn.Conv2d(512, 256, 3, bias=False, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
                                   nn.Conv2d(256, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.down1 = nn.Sequential(nn.Conv2d(256, 128, 3, bias=False, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                                   nn.Conv2d(128, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())
        self.down0 = nn.Sequential(nn.Conv2d(64, 64, 3, bias=False, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                                   nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())

        self.shad_att = nn.Sequential(nn.Conv2d(32, 32, 3, bias=False, padding=1), nn.BatchNorm2d(32), nn.ReLU())

        self.dst1 = nn.Sequential(nn.Conv2d(32, 32, 3, bias=False, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, bias=False, padding=1), nn.BatchNorm2d(32), nn.ReLU())
        self.dst2 = nn.Sequential(nn.Conv2d(32, 32, 3, bias=False, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                                  nn.Conv2d(32, 32, 3, bias=False, padding=1), nn.BatchNorm2d(32), nn.ReLU())

        self.refine4_hl = ConvBlock()
        self.refine3_hl = ConvBlock()
        self.refine2_hl = ConvBlock()
        self.refine1_hl = ConvBlock()

        self.refine0_hl = ConvBlock()

        self.attention4_hl = AttentionModule()
        self.attention3_hl = AttentionModule()
        self.attention2_hl = AttentionModule()
        self.attention1_hl = AttentionModule()
        self.attention0_hl = AttentionModule()

        self.conv1x1_ReLU_down4 = nn.Sequential(nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                                                nn.Conv2d(32, 1, 1, bias=False))
        self.conv1x1_ReLU_down3 = nn.Sequential(nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                                                nn.Conv2d(32, 1, 1, bias=False))
        self.conv1x1_ReLU_down2 = nn.Sequential(nn.Conv2d(96, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                                                nn.Conv2d(32, 1, 1, bias=False))
        self.conv1x1_ReLU_down1 = nn.Sequential(nn.Conv2d(128, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                                                nn.Conv2d(32, 1, 1, bias=False))
        self.conv1x1_ReLU_down0 = nn.Sequential(nn.Conv2d(160, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
                                                nn.Conv2d(32, 1, 1, bias=False))

        self.fuse_predict = nn.Sequential(nn.Conv2d(5, 1, 1, bias=False))

    def forward(self, x):
        layer0 = self.layer0(x)      # (B, 64, 160, 160)
        layer1 = self.layer1(layer0) # (B, 256, 80, 80)
        layer2 = self.layer2(layer1) # (B, 512, 40, 40)
        layer3 = self.layer3(layer2) # (B, 1024, 20, 20)
        layer4 = self.layer4(layer3) # (B, 2048, 10, 10)

        down4 = self.down4(layer4)   # (B, 32, 10, 10)
        down3 = self.down3(layer3)   # (B, 32, 20, 20)
        down2 = self.down2(layer2)   # (B, 32, 40, 40)
        down1 = self.down1(layer1)   # (B, 32, 80, 80)
        down0 = self.down0(layer0)   # (B, 32, 160, 160)


        ### DSC - down4
        down4_dst1 = self.dst1(down4) # (B, 32, 10, 10)
        down4_dst2 = self.dst2(down4) # (B, 32, 10, 10)
        down4_dst1_3 = F.interpolate(down4_dst1, size=down3.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 20, 20)
        down4_dst1_2 = F.interpolate(down4_dst1, size=down2.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 40, 40)
        down4_dst1_1 = F.interpolate(down4_dst1, size=down1.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 80, 80)
        down4_dst1_0 = F.interpolate(down4_dst1, size=down0.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 160, 160)
        down4_dst2_3 = F.interpolate(down4_dst2, size=down3.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 20, 20)
        down4_dst2_2 = F.interpolate(down4_dst2, size=down2.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 40, 40)
        down4_dst2_1 = F.interpolate(down4_dst2, size=down1.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 80, 80)
        down4_dst2_0 = F.interpolate(down4_dst2, size=down0.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 160, 160)
        down4_shad = down4            # (B, 32, 10, 10)
        down4_shad = (1 + self.attention4_hl(torch.cat((down4_shad, down4_dst2), 1))) * down4_shad       # (B, 32, 10, 10)
        down4_shad = F.relu(-self.refine4_hl(torch.cat((down4_shad, down4_dst1), 1)) + down4_shad, True) # (B, 32, 10, 10)
        down4_shad_3 = F.interpolate(down4_shad, size=down3.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 20, 20)
        down4_shad_2 = F.interpolate(down4_shad, size=down2.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 40, 40)
        down4_shad_1 = F.interpolate(down4_shad, size=down1.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 80, 80)
        down4_shad_0 = F.interpolate(down4_shad, size=down0.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 160, 160)
        up_down4_dst1 = self.conv1x1_ReLU_down4(down4_dst1) # (B, 1, 10, 10)
        up_down4_dst2 = self.conv1x1_ReLU_down4(down4_dst2) # (B, 1, 10, 10)
        up_down4_shad = self.conv1x1_ReLU_down4(down4_shad) # (B, 1, 10, 10)
        pred_down4_dst1 = F.interpolate(up_down4_dst1, size=x.size()[2:], mode='bilinear', align_corners=False) # (B, 1, 320, 320)
        pred_down4_dst2 = F.interpolate(up_down4_dst2, size=x.size()[2:], mode='bilinear', align_corners=False) # (B, 1, 320, 320)
        pred_down4_shad = F.interpolate(up_down4_shad, size=x.size()[2:], mode='bilinear', align_corners=False) # (B, 1, 320, 320)


        ### DSC - down3
        down3_dst1 = self.dst1(down3) # (B, 32, 20, 20)
        down3_dst2 = self.dst2(down3) # (B, 32, 20, 20)
        down3_dst1_2 = F.interpolate(down3_dst1, size=down2.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 40, 40)
        down3_dst1_1 = F.interpolate(down3_dst1, size=down1.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 80, 80)
        down3_dst1_0 = F.interpolate(down3_dst1, size=down0.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 160, 160)
        down3_dst2_2 = F.interpolate(down3_dst2, size=down2.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 40, 40)
        down3_dst2_1 = F.interpolate(down3_dst2, size=down1.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 80, 80)
        down3_dst2_0 = F.interpolate(down3_dst2, size=down0.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 160, 160)
        down3_shad = down3            # (B, 32, 20, 20)
        down3_shad = (1 + self.attention3_hl(torch.cat((down3_shad, down3_dst2), 1))) * down3_shad       # (B, 32, 20, 20)
        down3_shad = F.relu(-self.refine3_hl(torch.cat((down3_shad, down3_dst1), 1)) + down3_shad, True) # (B, 32, 20, 20)
        down3_shad_2 = F.interpolate(down3_shad, size=down2.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 40, 40)
        down3_shad_1 = F.interpolate(down3_shad, size=down1.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 80, 80)
        down3_shad_0 = F.interpolate(down3_shad, size=down0.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 160, 160)
        up_down3_dst1 = self.conv1x1_ReLU_down3(torch.cat((down3_dst1, down4_dst1_3), 1)) # (B, 1, 20, 20)
        up_down3_dst2 = self.conv1x1_ReLU_down3(torch.cat((down3_dst2, down4_dst2_3), 1)) # (B, 1, 20, 20)
        up_down3_shad = self.conv1x1_ReLU_down3(torch.cat((down3_shad, down4_shad_3), 1)) # (B, 1, 20, 20)
        pred_down3_dst1 = F.interpolate(up_down3_dst1, size=x.size()[2:], mode='bilinear', align_corners=False) # (B, 1, 320, 320)
        pred_down3_dst2 = F.interpolate(up_down3_dst2, size=x.size()[2:], mode='bilinear', align_corners=False) # (B, 1, 320, 320)
        pred_down3_shad = F.interpolate(up_down3_shad, size=x.size()[2:], mode='bilinear', align_corners=False) # (B, 1, 320, 320)


        ### DSC - down2
        down2_dst1 = self.dst1(down2) # (B, 32, 40, 40)
        down2_dst2 = self.dst2(down2) # (B, 32, 40, 40)
        down2_dst1_1 = F.interpolate(down2_dst1, size=down1.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 80, 80)
        down2_dst1_0 = F.interpolate(down2_dst1, size=down0.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 160, 160)
        down2_dst2_1 = F.interpolate(down2_dst2, size=down1.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 80, 80)
        down2_dst2_0 = F.interpolate(down2_dst2, size=down0.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 160, 160)
        down2_shad = down2            # (B, 32, 40, 40)
        down2_shad = (1 + self.attention2_hl(torch.cat((down2_shad, down2_dst2), 1))) * down2_shad       # (B, 32, 40, 40)
        down2_shad = F.relu(-self.refine2_hl(torch.cat((down2_shad, down2_dst1), 1)) + down2_shad, True) # (B, 32, 40, 40)
        down2_shad_1 = F.interpolate(down2_shad, size=down1.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 80, 80)
        down2_shad_0 = F.interpolate(down2_shad, size=down0.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 160, 160)
        up_down2_dst1 = self.conv1x1_ReLU_down2(torch.cat((down2_dst1, down3_dst1_2, down4_dst1_2), 1)) # (B, 1, 40, 40)
        up_down2_dst2 = self.conv1x1_ReLU_down2(torch.cat((down2_dst2, down3_dst2_2, down4_dst2_2), 1)) # (B, 1, 40, 40)
        up_down2_shad = self.conv1x1_ReLU_down2(torch.cat((down2_shad, down3_shad_2, down4_shad_2), 1)) # (B, 1, 40, 40)
        pred_down2_dst1 = F.interpolate(up_down2_dst1, size=x.size()[2:], mode='bilinear', align_corners=False) # (B, 1, 320, 320)
        pred_down2_dst2 = F.interpolate(up_down2_dst2, size=x.size()[2:], mode='bilinear', align_corners=False) # (B, 1, 320, 320)
        pred_down2_shad = F.interpolate(up_down2_shad, size=x.size()[2:], mode='bilinear', align_corners=False) # (B, 1, 320, 320)


        ### DSC - down1
        down1_dst1 = self.dst1(down1) # (B, 32, 80, 80)
        down1_dst2 = self.dst2(down1) # (B, 32, 80, 80)
        down1_dst1_0 = F.interpolate(down1_dst1, size=down0.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 160, 160)
        down1_dst2_0 = F.interpolate(down1_dst2, size=down0.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 160, 160)
        down1_shad = down1            # (B, 32, 80, 80)
        down1_shad = (1 + self.attention1_hl(torch.cat((down1_shad, down1_dst2), 1))) * down1_shad       # (B, 32, 80, 80)
        down1_shad = F.relu(-self.refine1_hl(torch.cat((down1_shad, down1_dst1), 1)) + down1_shad, True) # (B, 32, 80, 80)
        down1_shad_0 = F.interpolate(down1_shad, size=down0.size()[2:], mode='bilinear', align_corners=False) # (B, 32, 160, 160)
        up_down1_dst1 = self.conv1x1_ReLU_down1(torch.cat((down1_dst1, down2_dst1_1, down3_dst1_1, down4_dst1_1), 1)) # (B, 1, 80, 80)
        up_down1_dst2 = self.conv1x1_ReLU_down1(torch.cat((down1_dst2, down2_dst2_1, down3_dst2_1, down4_dst2_1), 1)) # (B, 1, 80, 80)
        up_down1_shad = self.conv1x1_ReLU_down1(torch.cat((down1_shad, down2_shad_1, down3_shad_1, down4_shad_1), 1)) # (B, 1, 80, 80)
        pred_down1_dst1 = F.interpolate(up_down1_dst1,size=x.size()[2:], mode='bilinear', align_corners=False) # (B, 1, 320, 320)
        pred_down1_dst2 = F.interpolate(up_down1_dst2,size=x.size()[2:], mode='bilinear', align_corners=False) # (B, 1, 320, 320)
        pred_down1_shad = F.interpolate(up_down1_shad,size=x.size()[2:], mode='bilinear', align_corners=False) # (B, 1, 320, 320)


        ### DSC - down0
        down0_dst1 = self.dst1(down0) # (B, 32, 160, 160)
        down0_dst2 = self.dst2(down0) # (B, 32, 160, 160)
        down0_shad = down0            # (B, 32, 160, 160)
        down0_shad = (1 + self.attention0_hl(torch.cat((down0_shad, down0_dst2), 1))) * down0_shad       # (B, 32, 160, 160)
        down0_shad = F.relu(-self.refine0_hl(torch.cat((down0_shad, down0_dst1), 1)) + down0_shad, True) # (B, 32, 160, 160)
        up_down0_dst1 = self.conv1x1_ReLU_down0(torch.cat((down0_dst1, down1_dst1_0, down2_dst1_0, down3_dst1_0, down4_dst1_0), 1)) # (B, 1, 160, 160)
        up_down0_dst2 = self.conv1x1_ReLU_down0(torch.cat((down0_dst2, down1_dst2_0, down2_dst2_0, down3_dst2_0, down4_dst2_0), 1)) # (B, 1, 160, 160)
        up_down0_shad = self.conv1x1_ReLU_down0(torch.cat((down0_shad, down1_shad_0, down2_shad_0, down3_shad_0, down4_shad_0), 1)) # (B, 1, 160, 160)
        pred_down0_dst1 = F.interpolate(up_down0_dst1, size=x.size()[2:], mode='bilinear', align_corners=False) # (B, 1, 320, 320)
        pred_down0_dst2 = F.interpolate(up_down0_dst2, size=x.size()[2:], mode='bilinear', align_corners=False) # (B, 1, 320, 320)
        pred_down0_shad = F.interpolate(up_down0_shad, size=x.size()[2:], mode='bilinear', align_corners=False) # (B, 1, 320, 320)


        ### Fuse the predicted mask
        fuse_pred_shad = self.fuse_predict(torch.cat((pred_down0_shad, pred_down1_shad, pred_down2_shad, pred_down3_shad, pred_down4_shad), 1))
        fuse_pred_dst1 = self.fuse_predict(torch.cat((pred_down0_dst1, pred_down1_dst1, pred_down2_dst1, pred_down3_dst1, pred_down4_dst1), 1))
        fuse_pred_dst2 = self.fuse_predict(torch.cat((pred_down0_dst2, pred_down1_dst2, pred_down2_dst2, pred_down3_dst2, pred_down4_dst2), 1))

        if self.training:
            return fuse_pred_shad, pred_down1_shad, pred_down2_shad, pred_down3_shad, pred_down4_shad, \
            fuse_pred_dst1, pred_down1_dst1, pred_down2_dst1, pred_down3_dst1, pred_down4_dst1,\
            fuse_pred_dst2, pred_down1_dst2, pred_down2_dst2, pred_down3_dst2, pred_down4_dst2, \
                   pred_down0_dst1, pred_down0_dst2, pred_down0_shad
        return torch.sigmoid(fuse_pred_shad)
