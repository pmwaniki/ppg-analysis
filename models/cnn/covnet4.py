
import torch.nn as nn
import torch
import torch.nn.functional as F

def convblock(in_channels,out_channels,pooling=True):
    if pooling:
        return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels,out_channels,3,padding=1),
        # nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )
    return nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.Conv2d(in_channels,out_channels,3,padding=1),
        # nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class Convnet4(nn.Module):
    def __init__(self,in_features, hid_dim=64, z_dim=64):
        super().__init__()

        self.encoder=nn.Sequential(

            convblock(in_features,hid_dim,pooling=False),
            # convblock(hid_dim,hid_dim,pooling=False),
            # convblock(hid_dim,hid_dim,pooling=False),
            convblock(hid_dim,z_dim,pooling=True)
        )
    def forward(self,x):
        x=self.encoder(x)
        x=torch.mean(x,dim=(2,3))
        return x

# net=Convnet4(2,64,z_dim=32)
# t=torch.randn((5,2,320,300))
# output=net(t)
# output.shape

class Classifier(nn.Module):
    def __init__(self,in_features,hid_dim,z_dim):
        super().__init__()
        self.encoder=Convnet4(in_features,hid_dim,z_dim)
        self.dense1=nn.Linear(z_dim,512)
        self.dense2=nn.Linear(512,1)

    def forward(self,x):
        x=F.relu(self.encoder(x))
        x=F.relu(self.dense1(x))
        x=self.dense2(x)
        return x


