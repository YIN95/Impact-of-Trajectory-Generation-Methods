from fvcore.common.config import CfgNode

# ---------------------------------------------------------------------------- #
# Config definition
# ---------------------------------------------------------------------------- #
_C = CfgNode()


# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #

_C.TRAIN = CfgNode()

_C.TRAIN.ENABLE = True

# Dataset
_C.TRAIN.DATASET = "data"

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 64

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 1

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 1

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

_C.TRAIN.RESUME = False

_C.TRAIN.LOAD_PATH = ""


# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #

_C.TEST = CfgNode()

_C.TEST.ENABLE = True

# Dataset for testing.
_C.TEST.DATASET = "data"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 1

_C.TEST.LOAD_PATH = "" 

# Number of clips to sample from a video uniformly for aggregating the
# prediction results.
_C.TEST.NUM_ENSEMBLE_VIEWS = 10

# Number of crops to sample from a frame spatially for aggregating the
# prediction results.
_C.TEST.NUM_SPATIAL_CROPS = 3

# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #

_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

_C.SOLVER.LR_POLICY = "cosine"

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.MULTI_STEP = [30,50,60]

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCH = 0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.001

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"


# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #

_C.MODEL = CfgNode()

# Model architecture.
_C.MODEL.ARCH = ""

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5

_C.MODEL.NUM_CLASSES = 400


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #

_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# The index of the current machine.
_C.SHARD_ID = 0

# Output basedir.
_C.CHECKPOINT_DIR = "."

# Log basedir
_C.LOG_DIR = "."

# Data path
_C.DATA_PATH = "."

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# Log period in iters.
_C.LOG_PERIOD = 10

# Distributed backend.
_C.DIST_BACKEND = "nccl"





# ---


_C.SLOWFAST = CfgNode()

# Corresponds to the inverse of the channel reduction ratio, $\beta$ between
# the Slow and Fast pathways.
_C.SLOWFAST.BETA_INV = 8

# Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
# Fast pathways.
_C.SLOWFAST.ALPHA = 8

# Ratio of channel dimensions between the Slow and Fast pathways.
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2

# Kernel dimension used for fusing information from Fast pathway to Slow
# pathway.
_C.SLOWFAST.FUSION_KERNEL_SZ = 5



# -----------------------------------------------------------------------------
# ResNet options
# -----------------------------------------------------------------------------
_C.RESNET = CfgNode()

# Transformation function.
_C.RESNET.BLOCK_FUNC = "bottleneck_block"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt).
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C.RESNET.STRIDE_1X1 = False

#  If true, initialize the gamma of the final BN of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_BN = False

# Number of weight layers.
_C.RESNET.DEPTH = 50

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]

# Size of stride on different res stages.
_C.RESNET.SPATIAL_STRIDES = [[1,1], [2,2], [2,2], [2,2]]

# Size of dilation on different res stages.
_C.RESNET.SPATIAL_DILATIONS = [[1,1], [1,1], [1,1], [1,1]]

# -----------------------------------------------------------------------------
# Nonlocal options
# -----------------------------------------------------------------------------
_C.NONLOCAL = CfgNode()

# Index of each stage and block to add nonlocal layers.
_C.NONLOCAL.LOCATION = [[[]], [[]], [[]], [[]]]

# Number of group for nonlocal for each stage.
_C.NONLOCAL.GROUP = [[1], [1], [1], [1]]

# instatiation to use for non-local layer.
_C.NONLOCAL.INSTANTIATION = "dot_product"

# Size of pooling layers used in Non-Local.
_C.NONLOCAL.POOL = [
    # Res2
    [[1, 2, 2], [1, 2, 2]],
    # Res3
    [[1, 2, 2], [1, 2, 2]],
    # Res4
    [[1, 2, 2], [1, 2, 2]],
    # Res5
    [[1, 2, 2], [1, 2, 2]],
]



_C.DATA = CfgNode()

_C.DATA.NUM_FRAMES = 32

_C.DATA.LEN_INPUT = 2

_C.DATA.SAMPLING_RATE = 4

_C.DATA.SQUARE_SCALE = True

# The mean value of the video raw pixels across the R G B channels.
_C.DATA.MEAN = [0.45, 0.45, 0.45]
# List of input frame channel dimensions.

_C.DATA.INPUT_CHANNEL_NUM = [3, 3]

# The std value of the video raw pixels across the R G B channels.
_C.DATA.STD = [0.225, 0.225, 0.225]

# The spatial augmentation jitter scales for training.
_C.DATA.TRAIN_JITTER_SCALES = [256, 320]

# The spatial crop size for training.
_C.DATA.TRAIN_CROP_SIZE = 224

# The spatial crop size for testing.
_C.DATA.TEST_CROP_SIZE = 256

_C.TASK = 1


_C.STGCN = CfgNode()
_C.STGCN.LAYOUT = 'congreg8-marker'
_C.STGCN.STRATEGY = 'spatial'

_C.STGCN.MAX_HOP = 1
_C.STGCN.DILATION = 1
_C.STGCN.EDGE_IMPORTANCE = True
_C.STGCN.HIDDEN_FEATURES = 64
_C.STGCN.OUT_FEATURES = 16
_C.STGCN.TEMPORAL_KERNEL_SIZE = 9

def check_config(cfg):
    
    # SOLVER assertions
    assert 0<=cfg.SOLVER.WARMUP_EPOCH
    assert 0<cfg.SOLVER.WARMUP_START_LR<cfg.SOLVER.BASE_LR, " 0< warm_lr < base_lr "
    assert cfg.SOLVER.LR_POLICY in ["multistep", "cosine"]

    if cfg.SOLVER.LR_POLICY == 'multistep':
        assert cfg.SOLVER.WARMUP_EPOCH < cfg.SOLVER.MULTI_STEP[0]
        assert cfg.SOLVER.MULTI_STEP[-1] < cfg.SOLVER.MAX_EPOCH
    
    assert cfg.NUM_GPUS >0
    
    # TRAIN assertions
    assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # TEST assertions
    assert cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0

    # MODEL assertions
    assert cfg.MODEL.LOSS_FUNC in ["bce","cross_entropy","mse"]
    

def get_config(cfg_path, opts):
    cfg = _C.clone()
    assert cfg_path, "You need a yaml config file!"
    cfg.merge_from_file(cfg_path)
    if opts:
        cfg.merge_from_list(opts)
    check_config(cfg)
    return cfg
