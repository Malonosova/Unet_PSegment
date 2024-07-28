from torch import nn
import torch

def blocks(in_channels, hidden_channels, out_channels, kernel_size=3, padding=1, p=0.2):
    block = nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding),
        nn.ReLU(),
        nn.Dropout(p),
        nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU())
    return block

def blocks_end(in_channels, hidden_channels, out_channels, kernel_size=3, padding=1, p=0.2):
    block = nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=padding),
        nn.ReLU(),
        nn.Dropout(p),
        nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=padding))
    return block

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.upsample = nn.Upsample(scale_factor=2)
        
        self.enc0_conv0 = blocks(in_channels=3, hidden_channels=8, out_channels=8)
        self.enc0_conv1 = blocks(in_channels=8, hidden_channels=16, out_channels=16)
        self.enc0_conv2 = blocks(in_channels=16, hidden_channels=32, out_channels=32)
        
        self.hidden0_conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        
        self.dec0_conv0 = blocks(in_channels=64, hidden_channels=32, out_channels=16)
        self.dec0_conv1 = blocks(in_channels=32, hidden_channels=16, out_channels=8)
        self.dec0_conv2 = blocks_end(in_channels=16, hidden_channels=8, out_channels=2)
        
        
        self.enc1_conv0 = blocks(in_channels=5, hidden_channels=8, out_channels=8)
        self.enc1_conv1 = blocks(in_channels=8, hidden_channels=16, out_channels=16)
        self.enc1_conv2 = blocks(in_channels=16, hidden_channels=32, out_channels=32)
        
        self.hidden1_conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        
        self.dec1_conv0 = blocks(in_channels=64, hidden_channels=32, out_channels=16)
        self.dec1_conv1 = blocks(in_channels=32, hidden_channels=16, out_channels=8)
        self.dec1_conv2 = blocks_end(in_channels=16, hidden_channels=8, out_channels=3)
        
        
        self.enc2_conv0 = blocks(in_channels=8, hidden_channels=8, out_channels=8)
        self.enc2_conv1 = blocks(in_channels=8, hidden_channels=16, out_channels=16)
        self.enc2_conv2 = blocks(in_channels=16, hidden_channels=32, out_channels=32)
        
        self.hidden2_conv = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        
        self.dec2_conv0 = blocks(in_channels=64, hidden_channels=32, out_channels=16)
        self.dec2_conv1 = blocks(in_channels=32, hidden_channels=32, out_channels=16)
        self.dec2_conv2 = blocks_end(in_channels=24, hidden_channels=8, out_channels=7)
        
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        # step 0
        e0 = self.pool(self.enc0_conv0(x))
        e1 = self.pool(self.enc0_conv1(e0))
        e2 = self.pool(self.enc0_conv2(e1))

        b = self.hidden0_conv(e2)

        d0 = self.dec0_conv0(self.upsample(torch.cat((b, e2), dim=1)))
        d1 = self.dec0_conv1(self.upsample(torch.cat((d0, e1), dim=1)))
        d2 = self.dec0_conv2(self.upsample(torch.cat((d1, e0), dim=1)))
        out0 = self.softmax(d2)
        
        # step 1
        e0 = self.pool(self.enc1_conv0(torch.cat((x, out0), dim=1)))
        e1 = self.pool(self.enc1_conv1(e0))
        e2 = self.pool(self.enc1_conv2(e1))

        b = self.hidden1_conv(e2)

        d0 = self.dec1_conv0(self.upsample(torch.cat((b, e2), dim=1)))
        d1 = self.dec1_conv1(self.upsample(torch.cat((d0, e1), dim=1)))
        d2 = self.dec1_conv2(self.upsample(torch.cat((d1, e0), dim=1)))
        out1 = self.softmax(d2)
        
        # step 2
        e0 = self.pool(self.enc2_conv0(torch.cat((x, out0, out1), dim=1)))
        e1 = self.pool(self.enc2_conv1(e0))
        e2 = self.pool(self.enc2_conv2(e1))

        b = self.hidden2_conv(e2)

        d0 = self.dec2_conv0(self.upsample(torch.cat((b, e2), dim=1)))
        d1 = self.dec2_conv1(self.upsample(torch.cat((d0, e1), dim=1)))
        d2 = self.dec2_conv2(self.upsample(torch.cat((d1, e0), dim=1)))
        out2 = self.softmax(d2)
        
#         torch.cat()
        return out0, out1, out2 #.squeeze()