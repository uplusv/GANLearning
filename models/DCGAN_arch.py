from turtle import forward
import torch
import torch.nn as nn

class DCGAN_Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(100,512,4,1,0,bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.act1 = nn.ReLU(True)
        self.conv2 = nn.ConvTranspose2d(512,256,4,2,1,bias=False)
        self.bn2 = nn.BatchNorm2d(256),
        self.conv3 = nn.ConvTranspose2d(256,128,4,2,1,bias=False)
        self.bn3 = nn.BatchNorm2d(128),
        self.conv4 = nn.ConvTranspose2d(128,64,4,2,1,bias=False)
        self.bn4 = nn.BatchNorm2d(64),
        self.conv5 = nn.ConvTranspose2d(64,3,4,2,1,bias=False)
        self.act2 = nn.Tanh()
    
    def forward(self, x):
        feat = self.conv1(x)
        feat = self.act1(self.bn1(feat))
        feat = self.conv2(feat)
        feat = self.act1(self.bn2(feat))
        feat = self.conv3(feat)
        feat = self.act1(self.bn3(feat))
        feat = self.conv4(feat)
        feat = self.act1(self.bn4(feat))
        feat = self.conv5(feat)
        out = self.act2(feat)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
        nn.Conv2d(3,64,4,2,1,bias=False),
        nn.LeakyReLU(0.2,inplace=True),
        nn.Conv2d(64,128,4,2,1,bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2,inplace=True),
        nn.Conv2d(128,256,4,2,1,bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2,inplace=True),
        nn.Conv2d(256,512,4,2,1,bias=False),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2,inplace=True),
        nn.Conv2d(512,1,4,1,0,bias=False),
        nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.main(x)