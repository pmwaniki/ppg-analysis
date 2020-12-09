import torch.nn as nn
import torch.nn.functional as F
import torch
from functools import reduce



class WaveNet(nn.Module):
    def __init__(self,num_samples,num_channels=1,num_classes=1,num_blocks=2,num_layers=14,
                 skip_channels=16,res_channels=32,kernel_size=2,dropout=0.0,bias=False):
        super().__init__()
        self.num_samples=num_samples
        self.num_classes=num_samples
        self.dropout=dropout
        self.receptive_field=1+(kernel_size-1)*num_blocks*sum([2**k for k in range(num_layers)])
        self.output_width=num_samples-self.receptive_field+1
        self.out_features=self.output_width*skip_channels
        print(f"Receptive field={self.receptive_field}, output width={self.output_width}")

        self.start_conv=nn.Conv1d(num_channels,res_channels,1,)

        batch_norms=[]
        hs=[]

        for b in range(num_blocks):
            for i in range(num_layers):
                rate=2**i
                h=ResidualBlock(in_channels=res_channels,
                                out_channels=res_channels,skip_channels=skip_channels,kernel_size=kernel_size,
                                output_width=self.output_width,dilation=rate,bias=bias)
                h.name=f'b{b}-l{i}'
                hs.append(h)
                batch_norms.append(nn.BatchNorm1d(res_channels))
        self.hs=nn.ModuleList(hs)
        self.batch_norms=nn.ModuleList(batch_norms)
        self.relu1=nn.ReLU()
        self.conv_1_1=nn.Conv1d(skip_channels,skip_channels,1)
        self.relu2=nn.ReLU()
        self.conv_1_2=nn.Conv1d(skip_channels,skip_channels,1)
        # self.h_class=nn.Conv1d(res_channels,num_classes,2)
        self.fc=nn.Linear(self.output_width*skip_channels,num_classes)

    def forward(self,x):
        x=self.start_conv(x)
        skips=[]
        for layer,batch_norm in zip(self.hs,self.batch_norms):
            x,skip=layer(x)
            x=F.dropout(x,p=self.dropout)
            skip=F.dropout(skip,p=self.dropout)
            x=batch_norm(x)
            skips.append(skip)

        x=reduce(lambda a,b:a[:,:,-skip.size()[2]:]+b[:,:,-skip.size()[2]:],skips)
        x=self.relu1(self.conv_1_1(x))
        x=self.relu2(self.conv_1_2(x))
        sizes=x.size()
        x=x.view(sizes[0],sizes[1]*sizes[2])
        x=self.fc(x)
        return x




class GatedConv1d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,
                 padding=0,dilation=1,groups=1,bias=False):
        super().__init__()
        self.dilation=dilation

        self.conv_f=nn.Conv1d(in_channels,out_channels,kernel_size,stride=stride,
                              padding=padding,dilation=dilation,groups=groups,bias=bias)
        self.conv_g = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                                padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self,x):
        # padding=self.dilation-(x.shape[-1]+self.dilation-1) % self.dilation
        # x=F.pad(x,(self.dilation,self.dilation))
        return torch.mul(torch.tanh(self.conv_f(x)),torch.sigmoid(self.conv_g(x)))


class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,skip_channels,kernel_size,output_width,stride=1,
                 padding=0,dilation=1,groups=1,bias=True):
        super().__init__()
        self.output_width=output_width
        self.gatedconv=GatedConv1d(in_channels,out_channels,kernel_size,stride=stride,
                                   padding=padding,dilation=dilation,groups=groups,bias=bias)
        self.conv_1=nn.Conv1d(out_channels,out_channels,1,stride=1,padding=0,dilation=1,groups=1,bias=bias)
        self.conv_skip = nn.Conv1d(out_channels, skip_channels, 1, stride=1, padding=0, dilation=1, groups=1, bias=bias)

    def forward(self,x):
        gated=self.gatedconv(x)
        residual=self.conv_1(gated)
        residual=residual + x[:,:,-residual.size()[2]:]

        skip=self.conv_skip(gated)
        # residual=torch.add(skip,x)
        # skip=self.conv_skip(skip)
        return residual,skip




if __name__ == "__main__":
    from torchsummary import summary
    net=WaveNet(num_samples=800,num_channels=2,num_classes=5,num_blocks=5,num_layers=7).cuda()
    summary(net,(2,800))