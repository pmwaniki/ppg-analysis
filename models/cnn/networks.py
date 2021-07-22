
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes,out_planes,stride=1,groups=1,dilation=1):
    return nn.Conv2d(in_planes,out_planes,kernel_size=3,stride=stride,groups=groups,
                     padding=dilation,dilation=dilation,bias=False)

def conv5_1d(in_planes,out_planes,stride=1,dilation=1,groups=1):
    return nn.Conv1d(in_planes,out_planes,kernel_size=3,padding=dilation,stride=stride,
                     dilation=dilation,bias=False,groups=groups)

def conv1_1d(in_planes,out_planes,stride=1):
    return nn.Conv1d(in_planes,out_planes,kernel_size=1,stride=stride,bias=False)

def conv1x1(in_planes,out_planes,stride=1):
    return nn.Conv2d(in_planes,out_planes,stride=stride,kernel_size=1,bias=False)

def downsampler1d(in_planes,out_planes,kernel_size,stride,dilation,norm_layer):
    new_kernel_size=kernel_size*dilation-1 if dilation !=1 else kernel_size
    return nn.Sequential(
        nn.Conv1d(in_planes,out_planes,kernel_size=kernel_size,stride=stride,padding=dilation),
        norm_layer(out_planes)
    )

class Bottleneck_1d(nn.Module):
    expansion=2
    def __init__(self,in_planes,planes,stride=1,downsample=None,groups=1,
                 base_width=64,dilation=1,norm_layer=None):
        super().__init__()
        width=int(planes*(base_width/64))*groups
        if norm_layer is None:
            norm_layer=nn.BatchNorm1d
        self.downsample = downsample
        # if downsample is None:
        #     if stride !=1 or dilation !=1 or in_planes != planes*self.expansion:
        #         self.downsample=downsampler1d(in_planes,planes*self.expansion,kernel_size=3,
        #                                       stride=stride,dilation=dilation,norm_layer=norm_layer)


        self.conv1=conv1_1d(in_planes,width)
        self.bn1 = norm_layer(width)

        self.conv2=conv5_1d(width,width,stride=stride,groups=groups,dilation=dilation)
        self.bn2=norm_layer(width)

        self.conv3=conv1_1d(width,planes*self.expansion)
        self.bn3=norm_layer(planes*self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        identity=x
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)

        if self.downsample:
            identity=self.downsample(x)

        out+=identity
        out=self.relu(out)

        return out
# bottleneck_1d=Bottleneck_1d(32,32,stride=1,dilation=2)

class BasicBlock(nn.Module):
    expansion=1
    def __init__(self,in_planes,planes,stride=1,downsample=None,groups=1,
                 base_width=64,dilation=1,norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer=nn.BatchNorm2d
        if groups !=1 or base_width !=64:
            raise ValueError("basic block only supports group=1 and basewidth=64")
        if dilation>1:
            raise NotImplementedError("Dilation > 1 not supported in basic blocke")
        self.conv1=conv3x3(in_planes,planes,stride=stride)
        self.bn1=norm_layer(planes)
        self.relu=nn.ReLU(inplace=True)
        self.conv2=conv3x3(planes,planes)
        self.bn2=norm_layer(planes)
        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        identity=x
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)

        if self.downsample is not None:
            identity=self.downsample(x)
        out+=identity
        out=self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class Resnet1D(nn.Module):
    def __init__(self,block,layers,num_classes=1,
                 groups=1,width_per_group=64,
                 norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer=nn.BatchNorm1d
        self.norm_layer=norm_layer
        self.num_classes = num_classes
        self.in_planes=64
        self.dilation=1
        self.groups=groups
        self.base_width=width_per_group

        # self.bn0=nn.BatchNorm1d(2)
        self.conv1=nn.Conv1d(2,self.in_planes,kernel_size=7,dilation=2,padding=3,stride=1,bias=False)
        self.bn1=norm_layer(self.in_planes)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool1d(kernel_size=7,stride=3,padding=3)
        # self.maxpool=nn.AvgPool1d(kernel_size=7,stride=3,padding=3)
        self.layer1=self._make_layer(block,planes=64,blocks=layers[0],stride=1)
        self.layer2=self._make_layer(block,planes=128,blocks=layers[1],stride=2)
        self.layer3=self._make_layer(block,planes=256,blocks=layers[2],stride=2)
        self.layer4=self._make_layer(block,planes=512,blocks=layers[3],stride=2)

        self.avg_pool=nn.AdaptiveAvgPool1d(1)
        self.fc=nn.Linear(512*block.expansion,num_classes)
        self.out_features=512*block.expansion

        for m in self.modules():
            if isinstance(m,nn.Conv1d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,(nn.BatchNorm1d,nn.GroupNorm)):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)


    def _make_layer(self,block,planes,blocks,stride=1,dilate=True):
        norm_layer=self.norm_layer
        downsample=None
        previous_dilation=self.dilation
        if dilate:
            self.dilation*=stride
            stride=1
        if stride !=1 or self.in_planes != planes * block.expansion:
            downsample=nn.Sequential(
                conv1_1d(self.in_planes,planes*block.expansion,stride),
                norm_layer(planes*block.expansion)
            )
        # if stride !=1 or self.dilation !=1 or self.in_planes != planes*block.expansion:
        #     downsample=downsampler1d(self.in_planes,planes*block.expansion,kernel_size=3,
        #                              stride=stride,dilation=self.dilation,norm_layer=norm_layer)
        layers=[]
        layers.append(
            block(in_planes=self.in_planes,planes=planes,stride=stride,downsample=downsample,
                  groups=self.groups, base_width=self.base_width,dilation=self.dilation,norm_layer=norm_layer)
        )
        self.in_planes=planes*block.expansion
        for _ in range(1,blocks):
            layers.append(
                block(in_planes=self.in_planes,planes=planes,stride=stride,downsample=None,
                  groups=self.groups, base_width=self.base_width,dilation=1,norm_layer=norm_layer)
            )
        return nn.Sequential(*layers)

    def forward(self,x):
        # x=self.bn0(x)
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        x=self.layer1(x)
        x=self.maxpool(x)
        x=self.layer2(x)
        x = self.maxpool(x)
        x=self.layer3(x)
        x = self.maxpool(x)
        x=self.layer4(x)
        # x = self.maxpool(x)
        x=self.avg_pool(x)
        # sizes=x.size()
        # x=x.view(sizes[0],sizes[1]*sizes[2])
        x=torch.flatten(x,1)
        x=self.fc(x)
        return x


class ResNet(nn.Module):
    def __init__(self,block,layers,num_classes=1,zero_init_residual=False,
                 groups=1,width_per_group=64,replace_stride_with_dilation=None,
                 norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer=nn.BatchNorm2d
        self.norm_layer=norm_layer
        self.num_classes=num_classes

        self.in_planes=64
        self.dilation=1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation=[False,False,False]
        if len(replace_stride_with_dilation) !=3:
            raise ValueError("replace_stride_with_dilation should be None or 3 element tupple")
        self.groups=groups
        self.base_width=width_per_group

        self.bn0=nn.BatchNorm2d(2)
        self.conv1=nn.Conv2d(2,self.in_planes,kernel_size=5,padding=2,stride=1,bias=False)
        self.bn1=norm_layer(self.in_planes)
        self.relu=nn.ReLU(inplace=True)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)

        self.layer1=self._make_layer(block,64,layers[0])
        self.layer2=self._make_layer(block,128,layers[1],stride=2,
                                     dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512*block.expansion,num_classes,bias=True)
        self.out_features = 512 * block.expansion

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,(nn.BatchNorm2d,nn.GroupNorm)):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m,Bottleneck):
                    nn.init.constant_(m.bn3.weight,0)
                elif isinstance(m,BasicBlock):
                    nn.init.constant_(m.bn2.weight,0)

    def _make_layer(self,block,planes,blocks,stride=1,dilate=False):
        norm_layer=self.norm_layer
        downsample=None
        previous_dilation=self.dilation
        if dilate:
            self.dilation *= stride
            stride=1
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample=nn.Sequential(
                conv1x1(self.in_planes,planes*block.expansion,stride),
                norm_layer(planes*block.expansion)
            )
        layers=[]
        layers.append(block(self.in_planes,planes,stride,downsample,self.groups,
                            self.base_width,previous_dilation,norm_layer))
        self.in_planes=planes*block.expansion
        for _ in range(1,blocks):
            layers.append(block(self.in_planes,planes,groups=self.groups,
                                base_width=self.base_width,dilation=self.dilation,
                                norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def _forward_impl(self,x):
        x=self.bn0(x)
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)

        x=self.avgpool(x)
        x=torch.flatten(x,1)
        x=self.fc(x)
        return x

    def forward(self,x):
        return self._forward_impl(x)

def groupNorm(n_groups):
    def fun(planes):
        return nn.GroupNorm(n_groups,planes)
    return fun

def resnet18(num_classes=1,norm_layer=None,**kwargs):
    model=ResNet(Bottleneck,layers=[2,2,2,2],num_classes=num_classes,norm_layer=norm_layer,**kwargs)
    return model

def wideresnet50(num_classes=1,norm_layer=None,**kwargs):
    kwargs['width_per_group'] = 64 * 2
    model=ResNet(Bottleneck,layers=[3, 4, 6, 3],num_classes=num_classes,norm_layer=norm_layer,**kwargs)
    return model


def resnet34(num_classes=1,norm_layer=None,**kwargs):
    model=ResNet(BasicBlock,layers=[3,8,24,3],num_classes=num_classes,norm_layer=norm_layer,**kwargs)
    return model

def resnet50(num_classes=1,norm_layer=None,**kwargs):
    model=ResNet(Bottleneck,layers=[3,4,6,3],num_classes=num_classes,norm_layer=norm_layer,**kwargs)
    return model


def resnet50_1d(num_classes=1,norm_layer=None,**kwargs):
    # kwargs['width_per_group'] = 64 * 2
    model=Resnet1D(block=Bottleneck_1d,
                   layers=[3, 4, 6, 3],
                   # layers=[2,2,2,2],
                   num_classes=num_classes,
                   norm_layer=norm_layer,
                   **kwargs)
    return model

def wideresnet50_1d(num_classes=1,norm_layer=None,**kwargs):
    kwargs['width_per_group'] = 64 * 2
    model=Resnet1D(block=Bottleneck_1d,
                   layers=[3, 4, 6, 3],
                   # layers=[3,8,24,3],
                   num_classes=num_classes,
                   norm_layer=norm_layer,
                   **kwargs)
    return model

class Encoder(nn.Module):
    def __init__(self,raw_model,stft_model,representation_size=256,fc_layer=nn.Identity()):
        super().__init__()
        self.raw_model=raw_model
        self.raw_model.fc=nn.Identity()
        self.stft_model=stft_model
        self.stft_model.fc=nn.Identity()

        # num_features=raw_model.out_features+stft_model.out_features
        num_features=raw_model.num_classes+stft_model.num_classes
        self.bn0 = nn.BatchNorm1d(num_features)
        self.fc0=nn.Linear(num_features,representation_size)
        # self.bn1=nn.BatchNorm1d(256)
        self.relu=nn.ReLU(inplace=True)
        self.fc=fc_layer

        # for m in self.modules():
        #     if isinstance(m,nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)
        #         nn.init.constant_(m.bias,0)

    def forward(self,raw_input,stft_input):
        x1=self.raw_model(raw_input)
        x2=self.stft_model(stft_input)
        x=torch.cat((x1,x2),dim=1)
        x=self.bn0(x)
        x=self.relu(x)
        x=self.fc0(x)
        x=self.fc(x)
        return x

class EncoderRaw(nn.Module):
    def __init__(self,base_model,representation_size=256,num_classes=1,dropout=0.0):
        super().__init__()
        self.base_model=base_model
        self.base_model.fc=nn.Identity()
        self.dropout=dropout
        num_features=base_model.out_features
        self.fc0=nn.Linear(num_features,representation_size)
        self.relu=nn.ReLU(inplace=True)
        self.fc=nn.Linear(representation_size,num_classes)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.xavier_uniform_(self.fc0.weight)



    def forward(self,x):
        x=self.base_model(x)
        x=self.relu(x)
        x=F.dropout(x,p=self.dropout)
        x=self.fc0(x)
        x=self.relu(x)
        x=F.dropout(x,p=self.dropout)
        x=self.fc(x)
        return x


class MLP(nn.Module):
    def __init__(self,representation_size,num_classes,dim_hidden=512,dropout=0.0):
        super().__init__()
        self.fc=nn.Sequential(
            nn.BatchNorm1d(representation_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(representation_size,dim_hidden),
            nn.BatchNorm1d(dim_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(dim_hidden,num_classes)

        )
    def forward(self,x):
        return self.fc(x)











if __name__=="__main__":
    from torchsummary import summary
    model=resnet50_1d().cuda()
    summary(model,(2,800))
    model2=wideresnet50_1d().cuda()
    summary(model2,(2,800))
    # model1d=resnet1d().cuda()
    # summary(model1d,(2,800))
    # raw_model=resnet1d()
    # stft_model=resnet18()
    # full_model=Encoder(raw_model=raw_model,stft_model=stft_model,representation_size=128).cuda()
    # summary(full_model,[(2,800),(2,11,35)])