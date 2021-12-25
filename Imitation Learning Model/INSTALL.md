# Installation

## Requirements
- Python >= 3.6
- Numpy
- PyTorch >= 1.3
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- simplejson: `pip install simplejson`
- GCC >= 4.9
- PyAV: `conda install av -c conda-forge`
- ffmpeg (4.0 is prefereed, will be installed along with PyAV)
- PyYaml: (will be installed along with fvcore)
- OpenCV: `pip install opencv-python`
```
    pip install -U torch torchvision tensorboard
    pip install -U 'git+https://github.com/facebookresearch/fvcore.git'
    pip install -U scikit-learn
```