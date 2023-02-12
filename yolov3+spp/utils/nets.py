import torch
import torch.functional as F
from torch import nn
import os
import sys
from yolo_dataset import *

os.chdir(sys.path[0])


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-1)//2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.avt = nn.LeakyReLU()

    def forward(self, x):
        return self.avt(self.bn(self.conv2d(x)))

class ConvSet(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, 1)
        self.conv2 = Conv(256, 512, 3)
        self.conv3 = Conv(512, 256, 1)
        self.conv4 = Conv(256, 512, 3)
        self.conv5 = Conv(512, 256, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x





class SPP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.maxpool5 = nn.MaxPool2d(5, 1, 2)
        self.maxpool9 = nn.MaxPool2d(9, 1, 4)
        self.maxpool13 = nn.MaxPool2d(13, 1, 6)
    
    def forward(self, x):
        return torch.concat([x, self.maxpool5(x), self.maxpool9(x), self.maxpool13(x)], dim=1)

class ResBlock(nn.Module):
    def __init__(self, n, channels) -> None:
        super().__init__()
        self.n = n
        self.res= nn.Sequential(
            Conv(channels, channels//2, 1),
            Conv(channels//2, channels, 3)
        )

    def forward(self, x):
        for _ in range(self.n):
            x = x + self.res(x)

        return x
    





class Darknet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)

        self.conv1 = Conv(3, 32, 3)


        self.conv2 = Conv(32, 64, 3, 2)
        self.res_blk1 = ResBlock(1, 64)
        

        self.conv3 = Conv(64, 128, 3, 2)
        self.res_blk2 = ResBlock(2, 128)

        
        self.conv4 = Conv(128, 256, 3, 2)
        self.res_blk3 = ResBlock(8, 256)

        self.conv5 = Conv(256, 512, 3, 2)
        self.res_blk4 = ResBlock(8, 512)

        self.conv6 = Conv(512, 1024, 3, 2)
        self.res_blk5 = ResBlock(8, 1024)


        self.conv7 = Conv(1024, 512, 1)
        self.conv8 = Conv(512, 1024, 3)
        self.conv9 = Conv(1024, 512, 1)
        self.spp = SPP()
        self.conv10 = Conv(2048, 512, 1)
        self.conv11 = Conv(512, 1024, 3)
        self.conv12 = Conv(1024, 512, 1)
        self.conv_out_low = Conv(512, 1024, 3)


        self.conv13 = Conv(512, 256, 1)

        self.conv14 = Conv(256, 128, 1)

        self.convset1 = ConvSet(768, 256)
        self.conv_out_mid = Conv(256, 512, 3)



        self.conv15 = Conv(256, 128, 1)
        self.convset2 = ConvSet(384, 256)








    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res_blk1(x)

        x = self.conv3(x)
        x = self.res_blk2(x)
        
        x = self.conv4(x)
        x = self.res_blk3(x)
        feature_map_52 = x
        
        x = self.conv5(x)
        x = self.res_blk4(x)

        feature_map_26 = x

        
        x = self.conv6(x)
        x = self.res_blk5(x)

        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.spp(x)

        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        low_output = self.conv_out_low(x)

        x = self.conv13(x)
        x = self.upsample(x)
        x = torch.concat([x, feature_map_26], dim=1)

        x = self.convset1(x)
        mid_output = self.conv_out_mid(x)
        
        x = self.conv15(x)
        x = self.upsample(x)
        x = torch.concat([x, feature_map_52], dim=1)

        high_output = self.convset2(x)





        return low_output, mid_output, high_output


class YoloHead(nn.Module):
    def __init__(self, in_channels, out_channels ) -> None:
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        return self.conv2d(x)



class YoLoBody(nn.Module):
    def __init__(self, num_cclasses, num_anchor=3) -> None:
        super().__init__()

        self.darknet = Darknet()

        self.out_13 = YoloHead(1024, (num_cclasses+1+4)*num_anchor)
        self.out_26 = YoloHead(512, (num_cclasses+1+4)*num_anchor)
        self.out_52 = YoloHead(256, (num_cclasses+1+4)*num_anchor)

    def forward(self, x):
        out13, out26, out52 = self.darknet(x)
        return self.out_13(out13), self.out_26(out26), self.out_52(out52)


class YoLoLayer(nn.Module):
    def __init__(self, anchors, num_classes, img_size, stride) -> None:
        super().__init__()
        self.anchors = torch.Tensor(anchors) # ((), (), ...) # 9个
        self.stride = stride # 特征图上对应原图的缩放比例。
        self.num_anchors = len(anchors)
        self.num_output = num_classes + 1 + 4 # (tx, ty, tw, th, score...)

        # 初始化. 
        self.nx, self.ny, self.ng = 0, 0, (0, 0)  # initialize number of x, y gridpoints # 预测特征层初始化.
        
        # 缩放到不同尺度上.
        self.anchor_vec = self.anchors / self.stride # 将anchors缩放到不同预测特征层上的尺度 (3, 2) # 
        #
        self.anchor_wh = self.anchor_vec.reshape(1, self.num_anchors, 1, 1, 2) # 调整我们的视图
        self.grid = None
        



def test_darknet() -> None:
    x = torch.zeros(size=(1, 3, 416, 416))
    net = Darknet()
    low, mig, high = net(x)
    print(low.shape, mig.shape, high.shape)

def test_yolobody():
    x = torch.zeros(size=(1, 3, 416, 416))
    yolo = YoLoBody(20)
    out13, out26, out52 = yolo(x) 
    print(out13.shape, out26.shape, out52.shape)

if __name__ == "__main__":
    test_yolobody()
