from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataset.utils import multiLabel_binarizer
import torchvision
import os.path as osp
import os
import numpy as np
import random
import torch
from glob import glob
import dataset.video_container as container
import dataset.video_decoder as decoder
import utils.logger as _logger

logger = _logger.get_logger(__name__)


class CATER(Dataset):
    def __init__(self, cfg, mode='train'):

        # Only support train, val, and test mode.
        assert mode in ["train", "val", "test", ], \
            "Split '{}' not supported for CATER".format(mode)

        assert cfg.TASK in [1, 2, 3], \
            "Task {} not supported for CATER".format(cfg.TASK)

        self.mode = mode
        self.cfg = cfg
        self.num_tries = 10
        self.ext = ["avi", "mp4"]
        self.video_meta = {}

        if self.mode in ["train", "val"]:
            self._num_clips = 1
        elif self.mode in ["test"]:
            self._num_clips = (
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS
            )

        logger.info("Load CATER {}...".format(mode))
        self._load_data()
        logger.info("Successfully load CATER {} (size:{}).".format(
            self.mode, len(self.video_paths)))

    def _load_data(self):

        task_name = ['actions_present', 'actions_order_uniq', 'localize']
        label_txt = "train" if self.mode in ["train"] else "val"
        # open label file
        label_path = osp.join(
            self.cfg.DATA_PATH,
            'lists',
            task_name[self.cfg.TASK-1],
            label_txt+'.txt'
        )
        assert os.path.exists(label_path), \
            "{} label file not found".format(label_path)

        with open(label_path) as f:

            # save paths for avi and strip
            path_dict = {}
            video_dir = osp.join(self.cfg.DATA_PATH, "videos")
            assert os.path.exists(video_dir), \
                "{} no such videos dir".format(video_dir)

            for path in os.listdir(video_dir):
                if path.split('.')[-1] in self.ext:
                    path_dict[path.split('.')[0]] = osp.join(
                        self.cfg.DATA_PATH, "videos", path)

            # save video paths and labels
            self.video_paths = []
            self.labels = []
            self.spatial_temporal_idx = []
            clip_idx = 0
            for line in f.read().strip().split("\n"):
                name, string_label = line.strip().split()
                if name.split(".")[0] in path_dict:
                    for idx in range(self._num_clips):
                        self.video_paths.append(path_dict[name.split(".")[0]])
                        self.spatial_temporal_idx.append(idx)
                        self.video_meta[clip_idx * self._num_clips + idx] = {}
                        if self.cfg.TASK in [1, 2]:
                            self.labels.append(
                                multiLabel_binarizer(
                                    self.cfg,
                                    [int(i) for i in string_label.split(",")]
                                )
                            )
                        else:
                            self.labels.append(int(string_label))
                    clip_idx+=1
                
        assert (
            len(self.video_paths) > 0
        ), "Failed to load CATER split {} from {}".format(
            self.mode, video_dir
        )

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, index):

        if self.mode in ["train", "val"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
        elif self.mode in ["test"]:
            temporal_sample_index = (
                self.spatial_temporal_idx[index]
                // self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            # spatial_sample_index is in [0, 1, 2]. Corresponding to left,
            # center, or right if width is larger than height, and top, middle,
            # or bottom if height is larger than width.
            spatial_sample_index = (
                self.spatial_temporal_idx[index]
                % self.cfg.TEST.NUM_SPATIAL_CROPS
            )
            min_scale, max_scale, crop_size = [
                self.cfg.DATA.TEST_CROP_SIZE] * 3
            # The testing is deterministic and no jitter should be performed.

        video_container = None
        try:
            video_container = container.get_video_container(
                self.video_paths[index]
            )
        except Exception as e:
            logger.info(
                "Failed to load video from {} with error {}".format(
                    self.video_paths[index], e
                )
            )
        try:
            # Select a random video if the current video was not able to access.
            for i in range(self.num_tries):
                if video_container is None:
                    index = random.randint(0, len(self.video_paths) - 1)
                    continue

                # Decode video. Meta info is used to perform selective decoding.
                decoder_config = {
                    'index': spatial_sample_index,
                    'min_scale': min_scale,
                    'max_scale': max_scale,
                    'crop_size': crop_size,
                    'square_scale': self.cfg.DATA.SQUARE_SCALE,
                    'enable_flip': True if self.cfg.TASK in [1, 2] else False
                }

                frames = decoder.decode(
                    video_container,
                    self.cfg.DATA.SAMPLING_RATE,
                    self.cfg.DATA.NUM_FRAMES,
                    temporal_sample_index,
                    self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
                    video_meta=self.video_meta[index],
                    target_fps=30, 
                    config=decoder_config,
                )

                # If decoding failed (wrong format, video is too short, and etc),
                # select another video.
                if frames is None:
                    index = random.randint(0, len(self.video_paths) - 1)
                    continue

                label = torch.from_numpy(np.array(self.labels[index])).float()
                return frames, label, index
        except:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self.num_tries
                )
            )

