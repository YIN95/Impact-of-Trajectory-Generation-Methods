import torch.nn as nn
import torch

class InputLayer(nn.Module):
    """
    model inputlayer. Provides stem operations of Conv, BN, ReLU, MaxPool
    on input data tensor for one or multiple pathways.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu = True,
        pool = None,
        cfg = None
    ):
        """
        The `__init__` method of any subclass should also contain these
        arguments. List size of 1 for single pathway models (C2D, I3D, SlowOnly
        and etc), list size of 2 for two pathway models (SlowFast).

        Args:
            dim_in (list): the list of channel dimensions of the inputs.
            dim_out (list): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernels' size of the convolutions in the stem
                layers. Temporal kernel size, height kernel size, width kernel
                size in order.
            stride (list): the stride sizes of the convolutions in the stem
                layer. Temporal kernel stride, height kernel size, width kernel
                size in order.
            padding (list): the paddings' sizes of the convolutions in the stem
                layer. Temporal padding size, height padding size, width padding
                size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
        """
        super(InputLayer, self).__init__()

        assert (
            len(
                {
                    len(dim_in),
                    len(dim_out),
                    len(kernel),
                    len(stride),
                    len(padding),
                }
            )
            == 1
        ), "Input path dimensions are not consistent."
        self.num_paths = len(dim_in)
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.pool = pool
        self.cfg = cfg
        # Construct the stem layer.
        self._make_layer(dim_in, dim_out)

    def _make_layer(self, dim_in, dim_out):
        for path in range(len(dim_in)):
            layer = HeadLayer(
                dim_in[path],
                dim_out[path],
                self.kernel[path],
                self.stride[path],
                self.padding[path],
                self.inplace_relu,
                pool=self.pool
            )
            self.add_module("path{}".format(path), layer)

    def forward(self, x):

        if self.cfg.MODEL.ARCH == 'slowfast':
            x = x.float()/255
            x = (x - torch.tensor(self.cfg.DATA.MEAN).cuda())/torch.tensor(self.cfg.DATA.STD).cuda()
            x = x.permute(0,4,1,2,3)
            y = x[:,:,::self.cfg.SLOWFAST.ALPHA,:,:]
            slow = getattr(self, "path0")
            fast = getattr(self, "path1")
            return [slow(y),fast(x)]

        else:
            x = x.float()/255
            x = (x - torch.tensor(self.cfg.DATA.MEAN).cuda())/torch.tensor(self.cfg.DATA.STD).cuda()
            x = x.permute(0,4,1,2,3)
            s_stem = getattr(self, "path0")
            return [s_stem(x)]


class HeadLayer(nn.Module):
    """
    ResNe(X)t 3D HeadLayer.
    Performs spatiotemporal Convolution, BN, and Relu following by a
        spatiotemporal pooling.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        kernel,
        stride,
        padding,
        inplace_relu=True,
        pool=None
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.

        Args:
            dim_in (int): the channel dimension of the input. Normally 3 is used
                for rgb input, and 2 or 3 is used for optical flow input.
            dim_out (int): the output dimension of the convolution in the stem
                layer.
            kernel (list): the kernel size of the convolution in the stem layer.
                temporal kernel size, height kernel size, width kernel size in
                order.
            stride (list): the stride size of the convolution in the stem layer.
                temporal kernel stride, height kernel size, width kernel size in
                order.
            padding (int): the padding size of the convolution in the stem
                layer, temporal padding size, height padding size, width
                padding size in order.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
        """
        super(HeadLayer, self).__init__()
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.inplace_relu = inplace_relu
        self.pool = pool
        # Construct the stem layer.
        self._make_layer(dim_in, dim_out)

    def _make_layer(self, dim_in, dim_out):
        self.conv = nn.Conv3d(
            dim_in,
            dim_out,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            bias=False,
        )
        self.bn = nn.BatchNorm3d(dim_out)
        self.relu = nn.ReLU(self.inplace_relu)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.pool:
            x = self.pool(x)
        return x
