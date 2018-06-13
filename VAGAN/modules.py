import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class generator_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(4,4), stride=(2,2), padding=1):
        super(generator_block,self).__init__()
        
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
        
    def forward(self, x):
        return self.block(x)

class downstream_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(downstream_block, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), 
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding), 
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        return self.block(x)

class upstream_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(upstream_block, self).__init__()
        
        self.ups = nn.Upsample(scale_factor=2)
        
        self.block = nn.Sequential(
            #1 part
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            #2 part
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x1, x2):
        x2 = self.ups(x2)
        x = torch.cat([x1, x2], dim=1)
        return self.block(x)
    
class unetGen(nn.Module):
    def __init__(self, img_dim=1, cond_dim=4):
        super(unetGen, self).__init__()
        
        self.inconv1_img = nn.Conv2d(in_channels=img_dim, out_channels=64, kernel_size=3, padding=1)
        self.inconv2_img = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        
        self.inconv1_attr = nn.Conv2d(in_channels=cond_dim, out_channels=64, kernel_size=3, padding=1)
        
        self.down1 = downstream_block(in_channels=128, out_channels=128)
        self.down2 = downstream_block(in_channels=128, out_channels=256)
        self.down3 = downstream_block(in_channels=256, out_channels=512)
        self.down4 = downstream_block(in_channels=512, out_channels=1024)
        
        self.up1 = upstream_block(in_channels=1024+512, out_channels=256)
        self.up2 = upstream_block(in_channels=512, out_channels=128)
        self.up3 = upstream_block(in_channels=256, out_channels=64)
        self.up4 = upstream_block(in_channels=128, out_channels=64)
        
        self.outconv1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.tanh = nn.Tanh()
        
        self.inconv1_img.weight.data.normal_(0.0, 0.02)
        self.inconv1_img.bias.data.zero_()
        
        self.inconv2_img.weight.data.normal_(0.0, 0.02)
        self.inconv2_img.bias.data.zero_()
        
        self.inconv1_attr.weight.data.normal_(0.0, 0.02)
        self.inconv1_attr.bias.data.zero_()

        self.outconv1.weight.data.normal_(0.0, 0.02)
        self.outconv1.bias.data.zero_()
        
    def forward(self, x, y):
        
        x_in = self.inconv2_img(self.inconv1_img(x))
        y_in = self.inconv1_attr(y)
        
        x_cat = torch.cat([x_in,y_in], dim=1)
        x1 = self.down1(x_cat) 
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        x = self.up1(x3, x4)
        x = self.up2(x2, x)
        x = self.up3(x1, x)
        x = self.up4(x_in, x)
        
        x = self.outconv1(x)
        return self.tanh(x)
    
class discriminator_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(discriminator_block, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU()
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        return self.block(x)
    
class discriminator_blockBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super(discriminator_blockBN, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        return self.block(x)
    
class ADVdiscriminator(nn.Module):
    def __init__(self, x_dim=1):
        super(ADVdiscriminator,self).__init__()
        
        self.inconv1 = nn.Conv2d(in_channels=x_dim, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.LeakyReLU(inplace=True)
        
        self.block1 = discriminator_block(64, 128)
        self.block2 = discriminator_block(128,256)
        self.block3 = discriminator_block(256, 512)
        self.block4 = discriminator_block(512, 1024)
        
        self.outconv = nn.Conv2d(1024, 1, kernel_size=4, stride=1)
        
        self.inconv1.weight.data.normal_(0.0, 0.02)
        self.inconv1.bias.data.zero_()
        
        self.outconv.weight.data.normal_(0.0, 0.02)
        self.outconv.bias.data.zero_()
        
    def forward(self, x):
        
        x_conv = self.relu1(self.inconv1(x))
        
        out = self.block1(x_conv)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)

        return self.outconv(out)
    
class discriminator(nn.Module):
    def __init__(self, x_dim=1):
        super(discriminator,self).__init__()
        
        self.inconv1 = nn.Conv2d(in_channels=x_dim, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.LeakyReLU(inplace=True)
        
        self.block1 = discriminator_blockBN(64, 128)
        self.block2 = discriminator_blockBN(128,256)
        self.block3 = discriminator_blockBN(256, 512)
        self.block4 = discriminator_blockBN(512, 1024)
        
        self.outconv = nn.Conv2d(1024, 1, kernel_size=4, stride=1)
        
        self.inconv1.weight.data.normal_(0.0, 0.02)
        self.inconv1.bias.data.zero_()
        
        self.outconv.weight.data.normal_(0.0, 0.02)
        self.outconv.bias.data.zero_()
        
    def forward(self, x):
        
        x_conv = self.relu1(self.inconv1(x))
        
        out = self.block1(x_conv)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        
        out=self.outconv(out)
        return out

class ADASregressor(nn.Module):
    def __init__(self, x_dim=1):
        super(ADASregressor,self).__init__()
        
        self.inconv1 = nn.Conv2d(in_channels=x_dim, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.LeakyReLU(inplace=True)
        
        self.block1 = discriminator_blockBN(64, 128)
        self.block2 = discriminator_blockBN(128,256)
        self.block3 = discriminator_blockBN(256, 512)
        self.block4 = discriminator_blockBN(512, 1024)
        
        self.av = nn.AvgPool2d(kernel_size=4)
        self.do = nn.Dropout()
        self.relu = nn.LeakyReLU(inplace=True)
        self.lin = nn.Linear(1024, 1)
        
    def forward(self, x):
        
        x_conv = self.relu1(self.inconv1(x))
        
        out = self.block1(x_conv)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.av(out)
        out = self.do(out)
        return self.relu(self.lin(out.view(out.size(0),-1)))
    
class ADASregressorFCN(nn.Module): #works faster, gives better results
    def __init__(self, x_dim=1, y_dim=4):
        super(ADASregressorFCN,self).__init__()
        
        self.inconv1 = nn.Conv2d(in_channels=x_dim, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.relu1 = nn.LeakyReLU(inplace=True)
        
        self.block1 = discriminator_blockBN(64, 128)
        self.block2 = discriminator_blockBN(128,256)
        self.block3 = discriminator_blockBN(256, 512)
        self.block4 = discriminator_blockBN(512, 1024)
        
        self.outconv = nn.Conv2d(1024, 1, kernel_size=4, stride=1)
        
        self.inconv1.weight.data.normal_(0.0, 0.02)
        self.inconv1.bias.data.zero_()
        
        self.outconv.weight.data.normal_(0.0, 0.02)
        self.outconv.bias.data.zero_()
        
    def forward(self, x):
        
        x_conv = self.relu1(self.inconv1(x))
        out = self.block1(x_conv)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        return self.outconv(out)