import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseBlock(nn.Module):
    """Block with dense connections

    :param params: {
        'num_channels':1,
        'num_filters':64,
        'kernel_h':5,
        'kernel_w':5,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':28,
        'drop_out':0,2}
    :type params: dict
    :return: forward passed tensor
    :rtype: torch.tonsor [FloatTensor]
    """

    def __init__(self, params):
        super(DenseBlock, self).__init__()

        padding_h = int((params['kernel_h'] - 1) / 2)
        padding_w = int((params['kernel_w'] - 1) / 2)

        conv1_out_size = int(params['num_channels'] + params['num_filters'])
        conv2_out_size = int(
            params['num_channels'] + params['num_filters'] + params['num_filters'])

        self.conv1 = nn.Conv2d(in_channels=params['num_channels'], out_channels=params['num_filters'],
                               kernel_size=(
                                   params['kernel_h'], params['kernel_w']),
                               padding=(padding_h, padding_w),
                               stride=params['stride_conv'])
        self.conv2 = nn.Conv2d(in_channels=conv1_out_size, out_channels=params['num_filters'],
                               kernel_size=(
                                   params['kernel_h'], params['kernel_w']),
                               padding=(padding_h, padding_w),
                               stride=params['stride_conv'])
        self.conv3 = nn.Conv2d(in_channels=conv2_out_size, out_channels=params['num_filters'],
                               kernel_size=(1, 1),
                               padding=(0, 0),
                               stride=params['stride_conv'])
        self.batchnorm1 = nn.BatchNorm2d(num_features=params['num_channels'])
        self.batchnorm2 = nn.BatchNorm2d(num_features=conv1_out_size)
        self.batchnorm3 = nn.BatchNorm2d(num_features=conv2_out_size)
        self.prelu = nn.PReLU()
        if params['drop_out'] > 0:
            self.drop_out_needed = True
            self.drop_out = nn.Dropout2d(params['drop_out'])
        else:
            self.drop_out_needed = False

    def forward(self, input):
        """Forward pass

        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :return: Forward passed tensor
        :rtype: torch.tensor [FloatTensor]
        """

        o1 = self.batchnorm1(input)
        o2 = self.prelu(o1)
        o3 = self.conv1(o2)
        o4 = torch.cat((input, o3), dim=1)
        o5 = self.batchnorm2(o4)
        o6 = self.prelu(o5)
        o7 = self.conv2(o6)
        o8 = torch.cat((input, o3, o7), dim=1)
        o9 = self.batchnorm3(o8)
        o10 = self.prelu(o9)
        out = self.conv3(o10)
        return out


class EncoderBlock(DenseBlock):
    """Dense encoder block with maxpool

    :param params: {
        'num_channels':1,
        'num_filters':64,
        'kernel_h':5,
        'kernel_w':5,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':28,
        'drop_out':0,2}
    :type params: dict
    :return: output tensor with maxpool, output tensor without maxpool, indices for unpooling
    :rtype: torch.tensor [FloatTensor], torch.tensor [FloatTensor], torch.tensor [LongTensor] 
    """

    def __init__(self, params):
        super(EncoderBlock, self).__init__(params)
        self.maxpool = nn.MaxPool2d(kernel_size=params['pool'], stride=params['stride_pool'], return_indices=True)

    def forward(self, input, weights=None):
        """Forward pass   
        
        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :param weights: Weights used for squeeze and excitation, shape depends on the type of SE block, defaults to None
        :type weights: torch.tensor, optional
        :return: output tensor with maxpool, output tensor without maxpool, indices for unpooling
        :rtype: torch.tensor [FloatTensor], torch.tensor [FloatTensor], torch.tensor [LongTensor] 
        """

        out_block = super(EncoderBlock, self).forward(input)

        if self.drop_out_needed:
            out_block = self.drop_out(out_block)

        out_encoder, indices = self.maxpool(out_block)
        return out_encoder, out_block, indices


class DecoderBlock(DenseBlock):
    """Dense decoder block with maxunpool and an optional skip connections

    :param params: {
        'num_channels':1,
        'num_filters':64,
        'kernel_h':5,
        'kernel_w':5,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':28,
        'drop_out':0,2}
    :type params: dict
    :return: forward passed tensor
    :rtype: torch.tensor [FloatTensor]
    """

    def __init__(self, params):
        super(DecoderBlock, self).__init__(params)
        self.unpool = nn.MaxUnpool2d(kernel_size=params['pool'], stride=params['stride_pool'])

    def forward(self, input, out_block=None, indices=None, weights=None):
        """Forward pass

        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :param out_block: Tensor for skip connection, shape = (N x C x H x W), defaults to None
        :type out_block: torch.tensor [FloatTensor], optional
        :param indices: Indices used for unpooling operation, defaults to None
        :type indices: torch.tensor, optional
        :param weights: Weights used for squeeze and excitation, shape depends on the type of SE block, defaults to None
        :type weights: torch.tensor, optional
        :return: Forward passed tensor
        :rtype: torch.tensor [FloatTensor]
        """
        if indices is not None:
            unpool = self.unpool(input, indices)
        else:
            # TODO: Implement Conv Transpose
            print("You have to use Conv Transpose")

        if out_block is not None:
            concat = torch.cat((out_block, unpool), dim=1)
        else:
            concat = unpool
        out_block = super(DecoderBlock, self).forward(concat)

        if self.drop_out_needed:
            out_block = self.drop_out(out_block)
        return out_block


class ClassifierBlock(nn.Module):
    """
    Last layer

    :param params: {
        'num_channels':1,
        'num_filters':64,
        'kernel_c':5,
        'stride_conv':1,
        'pool':2,
        'stride_pool':2,
        'num_classes':28,
        'drop_out':0,2}
    :type params: dict
    :return: forward passed tensor
    :rtype: torch.tensor [FloatTensor]
    """

    def __init__(self, params):
        super(ClassifierBlock, self).__init__()
        self.conv = nn.Conv2d(params['num_channels'], params['num_class'], params['kernel_c'], params['stride_conv'])

    def forward(self, input, weights=None):
        """Forward pass

        :param input: Input tensor, shape = (N x C x H x W)
        :type input: torch.tensor [FloatTensor]
        :param weights: Weights for classifier regression, defaults to None
        :type weights: torch.tensor (N), optional
        :return: logits
        :rtype: torch.tensor
        """
        batch_size, channel, a, b = input.size()
        if weights is not None:
            weights, _ = torch.max(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out_conv = F.conv2d(input, weights)
        else:
            out_conv = self.conv(input)
        return out_conv