import sys
sys.path.append(r"/data/Desktop/Semi-NC/")
import torch
import torch.nn as nn
from Network.Pro_Extract.DEAM_Module import DEAM
import torch.nn.functional as F

class Net_S(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Net_S, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.feas = []
        
        self.encoder = Encoder(n_channels)
        
        self.up11 = Up(1024, 512)
        self.up12 = Up(512, 256)
        self.up13 = Up(256, 128)
        self.up14 = Up(128, 64)
        self.nerve_out = OutConv(64, n_classes)

        self.up21 = Up(1024, 512)
        self.up22 = Up(512, 256)
        self.up23 = Up(256, 128)
        self.up24 = Up(128, 64)
        self.cell_out = OutConv(64, n_classes)

        self.combined_features = []

    def forward(self, image):
        self.feas.clear()
        self.combined_features.clear()
        encoder_features = self.encoder(image)
    
        image1, image2, image3, image4, image5 = encoder_features
        
        decoder11 = self.up11(image5, image4)
        self.feas.append(decoder11)
        decoder12 = self.up12(decoder11, image3)
        self.feas.append(decoder12)
        decoder13 = self.up13(decoder12, image2)
        self.feas.append(decoder13)
        decoder14 = self.up14(decoder13, image1)
        self.feas.append(decoder14)
        pred_n = self.nerve_out(decoder14)
        self.feas.append(pred_n)

        decoder21 = self.up21(image5, image4)
        self.feas.append(decoder21)
        decoder22 = self.up22(decoder21, image3)
        self.feas.append(decoder22)
        decoder23 = self.up23(decoder22, image2)
        self.feas.append(decoder23)
        decoder24 = self.up24(decoder23, image1)
        self.feas.append(decoder24)
        pred_c = self.cell_out(decoder24)
        self.feas.append(pred_c)

        self.combined_features = self.encoder.Features + self.feas

        return pred_n, pred_c

class Double_Conv(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        self.use_bn = use_bn
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
        else:
            self.bn1 = nn.BatchNorm2d(out_channels, track_running_stats=False, affine=False)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if self.use_bn:
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.bn2 = nn.BatchNorm2d(out_channels, track_running_stats=False, affine=False)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.double_conv = Double_Conv(in_channels, out_channels, use_bn)

    def forward(self, x):
        x = self.maxpool(x)
        x = self.double_conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True):
        super().__init__()
        self.use_bn = use_bn
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upconv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False)
        if self.use_bn:
            self.bn1 = nn.BatchNorm2d(out_channels)
        else:
            self.bn1 = nn.BatchNorm2d(out_channels, track_running_stats=False, affine=False)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        if self.use_bn:
            self.bn2 = nn.BatchNorm2d(out_channels)
        else:
            self.bn2 = nn.BatchNorm2d(out_channels, track_running_stats=False, affine=False)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, decoder, skip):
        decoder = self.upconv(self.upsample(decoder))
        x = torch.cat([skip, decoder], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()

        self.inc = Double_Conv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.DEAM1 = DEAM(d_model=128, nhead=8, d_ffn=512, dropout=0.1, act="gelu",
                         n_points=4, n_levels=1, n_sa_layers=1,
                         in_channles=[128], proj_idxs=(0,), activation="relu")
        
        self.DEAM2 = DEAM(d_model=128, nhead=8, d_ffn=512, dropout=0.1, act="gelu",
                         n_points=4, n_levels=2, n_sa_layers=1,
                         in_channles=[128, 256], proj_idxs=(0,1), activation="relu")

        self.DEAM3 = DEAM(d_model=128, nhead=8, d_ffn=512, dropout=0.1, act="gelu",
                         n_points=4, n_levels=3, n_sa_layers=1,
                         in_channles=[128, 256, 512], proj_idxs=(0,1,2), activation="relu")
        
        self.DEAM4 = DEAM(d_model=128, nhead=8, d_ffn=512, dropout=0.1, act="gelu",
                         n_points=4, n_levels=4, n_sa_layers=1,
                         in_channles=[128, 128, 128, 1024], proj_idxs=(0, 1, 2, 3), activation="relu")

        self.channel_adjust = nn.Sequential(
            nn.Conv2d(128, 1024, kernel_size=1),
            nn.BatchNorm2d(1024), 
            nn.ReLU(inplace=True)
        )

        self.Features = []

    def forward(self, input):
        self.Features.clear()
        features = []
        
        x1 = self.inc(input)      # [64, H, W]
        features.append(x1)
        self.Features.append(x1)

        x2 = self.down1(x1)       # [128, H/2, W/2]
        features.append(x2)
        self.Features.append(x2)

        x3 = self.down2(x2)       # [256, H/4, W/4]
        features.append(x3)
        self.Features.append(x3)

        x4 = self.down3(x3)       # [512, H/8, W/8]
        features.append(x4)
        self.Features.append(x4)

        x5 = self.down4(x4)  # [1024, H/16, W/16] 
        features.append(x5)
        self.Features.append(x5)
         
        trans_1 = self.DEAM1([x2])
        
        trans_2 = self.DEAM2([x2, x3])
        
        trans_3 = self.DEAM3([x2, x3, x4])
        
        trans_4 = self.DEAM4([trans_1[-1], trans_2[-1], trans_3[-1], x5])
        
        bottleneck = self.channel_adjust(trans_4[-1])  # [1024, H/8, W/8]
        self.Features.append(bottleneck)
    
        return_features = [
            features[0],  # x1: 64 channels [H, W]
            features[1],  # x2: 128 channels [H/2, W/2]  
            features[2],  # x3: 256 channels [H/4, W/4]
            features[3],  # x4: 512 channels [H/8, W/8]
            bottleneck    # 1024 channels [H/8, W/8]
        ]
        
        return return_features

if __name__ == '__main__':
   
    img = torch.Tensor(1, 1, 384, 384)
    net = Net_S(1, 2)
    out_n, out_c = net(img)
    print(f"Output nerve shape: {out_n.shape}")
    print(f"Output cell shape: {out_c.shape}")