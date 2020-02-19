import torch.nn as nn
import torch
from models.nonlocal_module import Nonlocal

def get_block_func(name):
    """
    Retrieves the block module by name.
    """
    block_funcs = {
        "bottleneck_block": BottleneckBlock,
        "basic_block": BasicBlock,
    }
    assert (
        name in block_funcs.keys()
    ), "Block function '{}' not supported".format(name)
    return block_funcs[name]

class BasicBlock(nn.Module):
    """
    Basic block: Tx3x3, 1x3x3, where T is the size of temporal kernel.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner=None,
        num_groups=1,
        stride_1x1=None,
        inplace_relu=True,
    ):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            dim_inner (None): the inner dimension would not be used in
                BasicBlock.
            num_groups (int): number of groups for the convolution. Number of
                group is als 1 for BasicBlock.
            stride_1x1 (None): stride_1x1 will not be used in BasicBlock.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
        """
        super(BasicBlock, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._make_block(dim_in, dim_out, stride)

    def _make_block(self, dim_in, dim_out, stride):
        # Tx3x3, BN, ReLU.
        self.a = nn.Conv3d(
            dim_in,
            dim_out,
            kernel_size=[self.temp_kernel_size, 3, 3],
            stride=[1, stride, stride],
            padding=[int(self.temp_kernel_size // 2), 1, 1],
            bias=False,
        )
        self.a_bn = nn.BatchNorm3d(dim_out)
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)
        # 1x3x3, BN.
        self.b = nn.Conv3d(
            dim_out,
            dim_out,
            kernel_size=[1, 3, 3],
            stride=[1, 1, 1],
            padding=[0, 1, 1],
            bias=False,
        )
        self.b_bn = nn.BatchNorm3d(dim_out)
        self.b_bn.block_final_bn = True

    def forward(self, x):
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)
        x = self.b(x)
        x = self.b_bn(x)
        return x


class BottleneckBlock(nn.Module):
    """
    Bottleneck block: Tx1x1, 1x3x3, 1x1x1, where T is the size of
        temporal kernel.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        dim_inner,
        num_groups,
        stride_1x1=False,
        inplace_relu=True,
        dilation=1,
    ):
        """
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            dilation (int): size of dilation.
        """
        super(BottleneckBlock, self).__init__()
        self.temp_kernel_size = temp_kernel_size
        self._inplace_relu = inplace_relu
        self._stride_1x1 = stride_1x1
        self._make_block(
            dim_in, dim_out, stride, dim_inner, num_groups, dilation
        )

    def _make_block(
        self, dim_in, dim_out, stride, dim_inner, num_groups, dilation
    ):
        (str1x1, str3x3) = (stride, 1) if self._stride_1x1 else (1, stride)

        # Tx1x1, BN, ReLU.
        self.a = nn.Conv3d(
            dim_in,
            dim_inner,
            kernel_size=[self.temp_kernel_size, 1, 1],
            stride=[1, str1x1, str1x1],
            padding=[int(self.temp_kernel_size // 2), 0, 0],
            bias=False,
        )
        self.a_bn = nn.BatchNorm3d(dim_inner)
        self.a_relu = nn.ReLU(inplace=self._inplace_relu)

        # 1x3x3, BN, ReLU.
        self.b = nn.Conv3d(
            dim_inner,
            dim_inner,
            [1, 3, 3],
            stride=[1, str3x3, str3x3],
            padding=[0, dilation, dilation],
            groups=num_groups,
            bias=False,
            dilation=[1, dilation, dilation],
        )
        self.b_bn = nn.BatchNorm3d(dim_inner)
        self.b_relu = nn.ReLU(inplace=self._inplace_relu)

        # 1x1x1, BN.
        self.c = nn.Conv3d(
            dim_inner,
            dim_out,
            kernel_size=[1, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.c_bn = nn.BatchNorm3d(dim_out)
        self.c_bn.block_final_bn = True

    def forward(self, x):
        # Explicitly forward every layer.
        # Branch2a.
        x = self.a(x)
        x = self.a_bn(x)
        x = self.a_relu(x)

        # Branch2b.
        x = self.b(x)
        x = self.b_bn(x)
        x = self.b_relu(x)

        # Branch2c
        x = self.c(x)
        x = self.c_bn(x)
        return x


class ResBlock(nn.Module):
    """
    Residual block.
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        block_func,
        dim_inner,
        num_groups=1,
        stride_1x1=False,
        inplace_relu=True,
        dilation=1,
    ):
        """
        ResBlock class constructs redisual blocks. More details can be found in:
            Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun.
            "Deep residual learning for image recognition."
            https://arxiv.org/abs/1512.03385
        Args:
            dim_in (int): the channel dimensions of the input.
            dim_out (int): the channel dimension of the output.
            temp_kernel_size (int): the temporal kernel sizes of the middle
                convolution in the bottleneck.
            stride (int): the stride of the bottleneck.
            block_func (string): block function to be used to construct the
                bottleneck.
            dim_inner (int): the inner dimension of the block.
            num_groups (int): number of groups for the convolution. num_groups=1
                is for standard ResNet like networks, and num_groups>1 is for
                ResNeXt like networks.
            stride_1x1 (bool): if True, apply stride to 1x1 conv, otherwise
                apply stride to the 3x3 conv.
            inplace_relu (bool): calculate the relu on the original input
                without allocating new memory.
            dilation (int): size of dilation.
        """
        super(ResBlock, self).__init__()
        self._inplace_relu = inplace_relu
        self._make_block(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            block_func,
            dim_inner,
            num_groups,
            stride_1x1,
            inplace_relu,
            dilation,
        )

    def _make_block(
        self,
        dim_in,
        dim_out,
        temp_kernel_size,
        stride,
        block_func,
        dim_inner,
        num_groups,
        stride_1x1,
        inplace_relu,
        dilation,
    ):
        # Use skip connection with projection if dim or res change.
        if (dim_in != dim_out) or (stride != 1):
            self.branch1 = nn.Conv3d(
                dim_in,
                dim_out,
                kernel_size=1,
                stride=[1, stride, stride],
                padding=0,
                bias=False,
                dilation=1,
            )
            self.branch1_bn = nn.BatchNorm3d(dim_out)
        self.branch2 = block_func(
            dim_in,
            dim_out,
            temp_kernel_size,
            stride,
            dim_inner,
            num_groups,
            stride_1x1=stride_1x1,
            inplace_relu=inplace_relu,
            dilation=dilation,
        )
        self.relu = nn.ReLU(self._inplace_relu)

    def forward(self, x):
        if hasattr(self, "branch1"):
            x = self.branch1_bn(self.branch1(x)) + self.branch2(x)
        else:
            x = x + self.branch2(x)
        x = self.relu(x)
        return x


class InternalLayer(nn.Module):
    """
    Stage of 3D ResNet. It expects to have one or more tensors as input for
        single path (C2D, I3D, SlowOnly), and multi-path (SlowFast) cases.
        More details can be found here:
        Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
        "Slowfast networks for video recognition."
        https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(
        self,
        dim_in,
        dim_out,
        stride,
        temp_kernel_sizes,
        num_blocks,
        dim_inner,
        num_groups,
        num_block_temp_kernel,
        nonlocal_inds,
        nonlocal_group,
        nonlocal_pool,
        dilation,
        instantiation="softmax",
        block_func_name="bottleneck_block",
        stride_1x1=False,
        inplace_relu=True,
    ):
        """
        The `__init__` method of any subclass should also contain these arguments.
        ResStage builds p streams, where p can be greater or equal to one.
        Args:
            dim_in (list): list of p the channel dimensions of the input.
                Different channel dimensions control the input dimension of
                different paths.
            dim_out (list): list of p the channel dimensions of the output.
                Different channel dimensions control the input dimension of
                different paths.
            temp_kernel_sizes (list): list of the p temporal kernel sizes of the
                convolution in the bottleneck. Different temp_kernel_sizes
                control different path.
            stride (list): list of the p strides of the bottleneck. Different
                stride control different path.
            num_blocks (list): list of p numbers of blocks for each of the
                path.
            dim_inner (list): list of the p inner channel dimensions of the
                input. Different channel dimensions control the input dimension
                of different paths.
            num_groups (list): list of number of p groups for the convolution.
                num_groups=1 is for standard ResNet like networks, and
                num_groups>1 is for ResNeXt like networks.
            num_block_temp_kernel (list): extent the temp_kernel_sizes to
                num_block_temp_kernel blocks, then fill temporal kernel size
                of 1 for the rest of the layers.
            nonlocal_inds (list): If the tuple is empty, no nonlocal layer will
                be added. If the tuple is not empty, add nonlocal layers after
                the index-th block.
            dilation (list): size of dilation for each path.
            nonlocal_group (list): list of number of p nonlocal groups. Each
                number controls how to fold temporal dimension to batch
                dimension before applying nonlocal block.
                https://github.com/facebookresearch/video-nonlocal-net.
            instantiation (string): different instantiation for nonlocal layer.
                Supports two different instantiation method:
                    "dot_product": normalizing correlation matrix with L2.
                    "softmax": normalizing correlation matrix with Softmax.
            block_func_name (string): name of the the block function apply
                on the network.
        """
        super(InternalLayer, self).__init__()
        assert all(
            (
                num_block_temp_kernel[i] <= num_blocks[i]
                for i in range(len(temp_kernel_sizes))
            )
        )
        self.num_blocks = num_blocks
        self.nonlocal_group = nonlocal_group
        self.temp_kernel_sizes = [
            (temp_kernel_sizes[i] * num_blocks[i])[: num_block_temp_kernel[i]]
            + [1] * (num_blocks[i] - num_block_temp_kernel[i])
            for i in range(len(temp_kernel_sizes))
        ]
        assert (
            len(
                {
                    len(dim_in),
                    len(dim_out),
                    len(temp_kernel_sizes),
                    len(stride),
                    len(num_blocks),
                    len(dim_inner),
                    len(num_groups),
                    len(num_block_temp_kernel),
                    len(nonlocal_inds),
                    len(nonlocal_group),
                }
            )
            == 1
        )
        self.num_paths = len(self.num_blocks)
        self._make_layer(
            dim_in,
            dim_out,
            stride,
            dim_inner,
            num_groups,
            block_func_name,
            stride_1x1,
            inplace_relu,
            nonlocal_inds,
            nonlocal_pool,
            instantiation,
            dilation,
        )

    def _make_layer(
        self,
        dim_in,
        dim_out,
        stride,
        dim_inner,
        num_groups,
        block_func_name,
        stride_1x1,
        inplace_relu,
        nonlocal_inds,
        nonlocal_pool,
        instantiation,
        dilation,
    ):
        for path in range(self.num_paths):
            for i in range(self.num_blocks[path]):
                # Retrieve the block function.
                block_func = get_block_func(block_func_name)
                # Construct the block.
                res_block = ResBlock(
                    dim_in[path] if i == 0 else dim_out[path],
                    dim_out[path],
                    self.temp_kernel_sizes[path][i],
                    stride[path] if i == 0 else 1,
                    block_func,
                    dim_inner[path],
                    num_groups[path],
                    stride_1x1=stride_1x1,
                    inplace_relu=inplace_relu,
                    dilation=dilation[path],
                )
                self.add_module("path{}_res{}".format(path, i), res_block)
                if i in nonlocal_inds[path]:
                    nln = Nonlocal(
                        dim_out[path],
                        dim_out[path] // 2,
                        nonlocal_pool[path],
                        instantiation=instantiation,
                    )
                    self.add_module(
                        "path{}_nonlocal{}".format(path, i), nln
                    )

    def forward(self, inputs):
        output = []
        for path in range(self.num_paths):
            x = inputs[path]
            for i in range(self.num_blocks[path]):
                m = getattr(self, "path{}_res{}".format(path, i))
                x = m(x)
                if hasattr(self, "path{}_nonlocal{}".format(path, i)):
                    nln = getattr(
                        self, "path{}_nonlocal{}".format(path, i)
                    )
                    b, c, t, h, w = x.shape
                    if self.nonlocal_group[path] > 1:
                        # Fold temporal dimension into batch dimension.
                        x = x.permute(0, 2, 1, 3, 4)
                        x = x.reshape(
                            b * self.nonlocal_group[path],
                            t // self.nonlocal_group[path],
                            c,
                            h,
                            w,
                        )
                        x = x.permute(0, 2, 1, 3, 4)
                    x = nln(x)
                    if self.nonlocal_group[path] > 1:
                        # Fold back to temporal dimension.
                        x = x.permute(0, 2, 1, 3, 4)
                        x = x.reshape(b, t, c, h, w)
                        x = x.permute(0, 2, 1, 3, 4)
            output.append(x)

        return output