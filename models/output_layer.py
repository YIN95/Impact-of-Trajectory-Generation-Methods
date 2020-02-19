import torch
import torch.nn as nn

class OutputLayer(nn.Module):
    """
    ResNe(X)t 3D OutputLayer.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1x1. If the inputs are from multiple
    different paths, the inputs will be concatenated after pooling.
    """
    
    def __init__(
        self, 
        dim_in,
        num_classes,
        pool='global_avg',
        dropout_rate=0.5,
        act_func="none",
        cfg=None
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p paths as input where p in [1, infty].
        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p spatial temporal
                poolings, temporal pool kernel size, spatial pool kernel size,
                spatial pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(OutputLayer, self).__init__()
        self.num_paths = len(dim_in)
        self.cfg = cfg
        for path in range(self.num_paths):
            if pool == 'global_avg':
                avg_pool = nn.AdaptiveAvgPool3d(1)
                self.add_module("path{}_avgpool".format(path), avg_pool)
        
        self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.

        self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)
  
        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=4)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        elif act_func == "none":
            self.act = None
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )
    
    def forward(self, inputs):
        assert (
            len(inputs) == self.num_paths
        ), "Input tensor does not contain {} path".format(self.num_paths)

        pool_out = []
        for path in range(self.num_paths):
            m = getattr(self, "path{}_avgpool".format(path))
            pool_out.append(m(inputs[path]))

        x = torch.cat(pool_out, 1)
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))

        x = self.dropout(x)
        x = self.projection(x)

        if not self.training:
            if self.act:
                x = self.act(x)
            x = x.mean([1, 2, 3])
        
        x = x.view(x.shape[0], -1)
        return x
        