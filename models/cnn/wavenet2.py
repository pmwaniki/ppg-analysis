
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
import numpy as np


def dilate(x, dilation, init_dilation=1, pad_start=True):
    """
    :param x: Tensor of size (N, C, L), where N is the input dilation, C is the number of channels, and L is the input length
    :param dilation: Target dilation. Will be the size of the first dimension of the output tensor.
    :param pad_start: If the input length is not compatible with the specified dilation, zero padding is used. This parameter determines wether the zeros are added at the start or at the end.
    :return: The dilated tensor of size (dilation, C, L*N / dilation). The output might be zero padded at the start
    """

    [n, c, l] = x.size()
    dilation_factor = dilation / init_dilation
    if dilation_factor == 1:
        return x

    # zero padding for reshaping
    new_l = int(np.ceil(l / dilation_factor) * dilation_factor)
    if new_l != l:
        l = new_l
        x = constant_pad_1d(x, new_l, dimension=2, pad_start=pad_start)

    l_old = int(round(l / dilation_factor))
    n_old = int(round(n * dilation_factor))
    l = math.ceil(l * init_dilation / dilation)
    n = math.ceil(n * dilation / init_dilation)

    # reshape according to dilation
    x = x.permute(1, 2, 0).contiguous()  # (n, c, l) -> (c, l, n)
    x = x.view(c, l, n)
    x = x.permute(2, 0, 1).contiguous()  # (c, l, n) -> (n, c, l)

    return x


class DilatedQueue:
    def __init__(self, max_length, data=None, dilation=1, num_deq=1, num_channels=1, dtype=torch.FloatTensor):
        self.in_pos = 0
        self.out_pos = 0
        self.num_deq = num_deq
        self.num_channels = num_channels
        self.dilation = dilation
        self.max_length = max_length
        self.data = data
        self.dtype = dtype
        if data == None:
            self.data = Variable(dtype(num_channels, max_length).zero_())

    def enqueue(self, input):
        self.data[:, self.in_pos] = input
        self.in_pos = (self.in_pos + 1) % self.max_length

    def dequeue(self, num_deq=1, dilation=1):
        #       |
        #  |6|7|8|1|2|3|4|5|
        #         |
        start = self.out_pos - ((num_deq - 1) * dilation)
        if start < 0:
            t1 = self.data[:, start::dilation]
            t2 = self.data[:, self.out_pos % dilation:self.out_pos + 1:dilation]
            t = torch.cat((t1, t2), 1)
        else:
            t = self.data[:, start:self.out_pos + 1:dilation]

        self.out_pos = (self.out_pos + 1) % self.max_length
        return t

    def reset(self):
        self.data = Variable(self.dtype(self.num_channels, self.max_length).zero_())
        self.in_pos = 0
        self.out_pos = 0


class ConstantPad1d(Function):
    # def __init__(self, target_size, dimension=0, value=0, pad_start=False):
    #     super(ConstantPad1d, self).__init__()
    #     self.target_size = target_size
    #     self.dimension = dimension
    #     self.value = value
    #     self.pad_start = pad_start

    @staticmethod
    def forward(ctx, input,target_size, dimension=0, value=0, pad_start=False):

        num_pad = target_size - input.size(dimension)

        assert num_pad >= 0, 'target size has to be greater than input size'

        input_size = input.size()
        # ctx.save_for_backward(num_pad, input_size,pad_start,dimension)
        ctx.num_pad=num_pad
        ctx.input_size=input_size
        ctx.pad_start=pad_start
        ctx.dimension=dimension

        size = list(input.size())
        size[dimension] = target_size
        output = input.new_empty(size,dtype=input.dtype,device=input.device).fill_(value)
        c_output = output

        # crop output
        if pad_start:
            c_output = c_output.narrow(dimension, num_pad, c_output.size(dimension) - num_pad)
        else:
            c_output = c_output.narrow(dimension, 0, c_output.size(dimension) - num_pad)

        c_output.copy_(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_target_size = grad_dimension = grad_value = grad_pad_start = grad_device = None
        # num_pad,input_size,pad_start,dimension=ctx.saved_tensors
        num_pad = ctx.num_pad
        input_size = ctx.input_size
        pad_start = ctx.pad_start
        dimension = ctx.dimension

        grad_input = grad_output.new_empty(grad_output.size(),dtype=grad_output.dtype,device=grad_output.device).zero_()
        cg_output = grad_output

        # crop grad_output
        if pad_start:
            cg_output = cg_output.narrow(dimension, num_pad, cg_output.size(dimension) - num_pad)
        else:
            cg_output = cg_output.narrow(dimension, 0, cg_output.size(dimension) - num_pad)

        grad_input.copy_(cg_output)
        return grad_input,grad_target_size, grad_dimension, grad_value, grad_pad_start


def constant_pad_1d(input,
                    target_size,
                    dimension=0,
                    value=0,
                    pad_start=False,
                    ):
    return ConstantPad1d.apply(input,target_size, dimension, value, pad_start)


class WaveNetModel(nn.Module):
    """
    A Complete Wavenet Model
    Args:
        layers (Int):               Number of layers in each block
        blocks (Int):               Number of wavenet blocks of this model
        dilation_channels (Int):    Number of channels for the dilated convolution
        residual_channels (Int):    Number of channels for the residual connection
        skip_channels (Int):        Number of channels for the skip connections
        classes (Int):              Number of possible values each sample can have
        input_length (Int):        Length of input
        kernel_size (Int):          Size of the dilation kernel
        dtype:                      Parameter type of this model
    Shape:
        - Input: :math:`(N, C_{in}, L_{in})`
        - Output: :math:`()`
        L should be the length of the receptive field
    """

    def __init__(self,
                 layers=10,
                 blocks=4,
                 dilation_channels=32,
                 residual_channels=32,
                 skip_channels=256,
                 dropout=0.0,
                 classes=256,
                 input_length=800,
                 kernel_size=2,
                 bias=False):

        super(WaveNetModel, self).__init__()

        self.layers = layers
        self.blocks = blocks
        self.dilation_channels = dilation_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.classes = classes
        self.kernel_size = kernel_size
        self.dropout = dropout
        # self.poolsize=10
        self.out_features = skip_channels*input_length

        # build model
        receptive_field = 1
        init_dilation = 1

        self.dilations = []
        self.dilated_queues = []
        # self.main_convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.norm_layers=nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=2,
                                    out_channels=residual_channels,
                                    kernel_size=1,
                                    bias=bias)

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilations of this layer
                self.dilations.append((new_dilation, init_dilation))

                # dilated queues for fast generation
                # self.dilated_queues.append(DilatedQueue(max_length=(kernel_size - 1) * new_dilation + 1,
                #                                         num_channels=residual_channels,
                #                                         dilation=new_dilation,
                #                                         dtype=dtype))

                # dilated convolutions
                self.filter_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=kernel_size,
                                                   padding=kernel_size//2,
                                                   bias=bias))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=kernel_size,
                                                 padding=kernel_size//2,
                                                 bias=bias))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=1,
                                                     bias=bias))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=1,
                                                 bias=bias))
                # self.norm_layers.append(nn.BatchNorm1d(residual_channels))

                receptive_field += additional_scope
                additional_scope *= 2
                init_dilation = new_dilation
                new_dilation *= 2

        # self.end_conv_1 = nn.Conv1d(in_channels=skip_channels,
        #                             out_channels=end_channels,
        #                             kernel_size=1,
        #                             bias=True)
        #
        # self.end_conv_2 = nn.Conv1d(in_channels=end_channels,
        #                             out_channels=classes,
        #                             kernel_size=1,
        #                             bias=True)
        # self.avgpool=nn.AdaptiveAvgPool1d(self.poolsize)
        self.fc=nn.Sequential(
            nn.ReLU(),
            # nn.Dropout(p=dropout),
            nn.Linear(self.out_features,classes))

        # self.output_length = 2 ** (layers - 1)
        # self.output_length = output_length
        self.receptive_field = receptive_field

    def wavenet(self, input, dilation_func):

        x = self.start_conv(input)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            (dilation, init_dilation) = self.dilations[i]

            residual = dilation_func(x, dilation, init_dilation, i)

            # dilated convolution
            filter = self.filter_convs[i](residual)
            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            if x.size(2) != 1:
                s = dilate(x, 1, init_dilation=dilation)
            s = self.skip_convs[i](s)
            # s = F.dropout(s,p=self.dropout,)

            try:
                skip = skip[:, :, -s.size(2):]
            except:
                skip = 0
            skip = s + skip

            x = self.residual_convs[i](x)
            # x = x + residual[:, :, (self.kernel_size - 1):]
            # x = self.norm_layers[i](x)
            x = x + residual

            # x = F.dropout(x,p=self.dropout)

        # x = F.relu(skip)
        # x = F.relu(self.end_conv_1(x))
        # x = self.end_conv_2(x)
        # x=skip.view(-1,self.out_features)
        # x=self.avgpool(skip)
        x=skip
        # x=x[:,:,-self.poolsize:]
        x=torch.flatten(x,start_dim=1)
        # x=self.fc0(x)
        x=self.fc(x)

        return x

    def wavenet_dilate(self, input, dilation, init_dilation, i):
        x = dilate(input, dilation, init_dilation)
        return x

    def queue_dilate(self, input, dilation, init_dilation, i):
        queue = self.dilated_queues[i]
        queue.enqueue(input.data[0])
        x = queue.dequeue(num_deq=self.kernel_size,
                          dilation=dilation)
        x = x.unsqueeze(0)

        return x

    def forward(self, input):
        x = self.wavenet(input,
                         dilation_func=self.wavenet_dilate)

        # reshape output
        # [n, c, l] = x.size()
        # l = self.output_length
        # x = x[:, :, -l:]
        # x = x.transpose(1, 2).contiguous()
        # x = x.view(n * l, c)
        return x


if __name__ == "__main__":
    from torchsummary import summary
    net=WaveNetModel(layers=5,blocks=5,dilation_channels=32,residual_channels=32,skip_channels=1024,
                     classes=1,input_length=800,kernel_size=3).cuda()
    summary(net,(2,800))





