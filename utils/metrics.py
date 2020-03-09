from fvcore.common.timer import Timer
import datetime
import numpy as np
from collections import deque
import torch
import utils.logger as _logger
from sklearn.metrics import average_precision_score, accuracy_score

logger = _logger.get_logger(__name__)


class TrainMeter(object):
    """
    Measures training stats.
    """
    
    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_acc= ScalarMeter(cfg.LOG_PERIOD)
        self.acc = ScalarMeter(cfg.LOG_PERIOD)

        self.num_top1_correct = 0
        self.num_samples = 0

        # Current mean average precision 
        self.map = ScalarMeter(cfg.LOG_PERIOD)
        self.total_acc = 0.0

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_acc.reset()

        self.num_top1_correct = 0
        self.num_samples = 0
        self.map.reset()
        self.acc.reset()
        self.total_acc = 0.0
    
    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
    
    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
    
    def update_stats_topk(self, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        # Current minibatch stats
        # self.mb_top1_acc.add_value(top1_acc)
        self.loss.add_value(loss)
        self.lr = lr
        # Aggregate stats
        # self.num_top1_correct += top1_acc * mb_size
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

    def update_stats_map(self, cur_map, cur_acc, loss, lr, mb_size):
        self.loss.add_value(loss)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.num_samples += mb_size
        self.map.add_value(cur_map)
        self.total_map += cur_map*mb_size
        self.acc.add_value(cur_acc)
        self.total_acc += cur_acc*mb_size
    
    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            # "top1_acc": self.mb_top1_acc.get_win_median(),
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
        }
        _logger.log_json_stats(stats)
    
    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
    
        # mAP = self.total_map / self.num_samples
        # acc = self.total_acc / self.num_samples
        # top1_acc = self.num_top1_correct / self.num_samples
        # top5_err = self.num_top5_mis / self.num_samples
        avg_loss = self.loss_total / self.num_samples
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "loss": avg_loss,
            "lr": self.lr,
        }
        _logger.log_json_stats(stats)


class ValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full val set).
        self.max_top1_acc = 0
        self.max_map = 0

        self.num_top1_correct = 0
        self.num_samples = 0
        # Current mean average precision 
        self.map = ScalarMeter(cfg.LOG_PERIOD)
        self.total_map = 0.0

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_top1_acc.reset()
        self.num_top1_correct = 0
        self.num_samples = 0
        self.map.reset()
        self.total_map = 0.0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()

    def update_stats_topk(self, loss, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        # Current minibatch stats
        # self.mb_top1_acc.add_value(top1_acc)
        self.loss.add_value(loss)
        # Aggregate stats
        # self.num_top1_correct += top1_acc * mb_size
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

    def update_stats_map(self, cur_map, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            mb_size (int): mini batch size.
        """
        self.map.add_value(cur_map)
        self.total_map += cur_map*mb_size
        self.num_samples += mb_size
          
    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))

        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "loss": self.loss.get_win_median(),
        }
        _logger.log_json_stats(stats)


    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        mAP = self.total_map / self.num_samples
        self.max_map = max(self.max_map, mAP)
        top1_acc = self.num_top1_correct / self.num_samples
        self.max_top1_acc = max(self.max_top1_acc, top1_acc)
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "top1_acc": top1_acc,
            "max_top1_acc": self.max_top1_acc,
        }
        _logger.log_json_stats(stats)


class TestMeter(object):
    """
    Perform the multi-view ensemble for testing: each video with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the video.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(self, num_videos, num_clips, num_cls, overall_iters):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each video, and calculate the metrics on
        num_videos videos.
        Args:
            num_videos (int): number of videos to test.
            num_clips (int): number of clips sampled from each video for
                aggregating the final prediction for the video.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
        """

        self.iter_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        # Initialize tensors.
        self.video_preds = torch.zeros((num_videos, num_cls))
        self.video_labels_class = torch.zeros((num_videos)).long()
        self.video_labels_multi_class = torch.zeros((num_videos, num_cls)).long()
        self.clip_count = torch.zeros((num_videos)).long()
        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.video_preds.zero_()
        self.video_labels_class.zero_()
        self.video_labels_multi_class.zero_()

    def update_stats(self, preds, labels, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(preds.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            self.video_labels_multi_class[vid_id] = labels[ind]
            self.video_preds[vid_id] += preds[ind]
            self.clip_count[vid_id] += 1
                
    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        _logger.log_json_stats(stats)

    def iter_tic(self):
        self.iter_timer.reset()

    def iter_toc(self):
        self.iter_timer.pause()

    def finalize_metrics_topk(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    self.clip_count, self.num_clips
                )
            )
            logger.warning(self.clip_count)

        num_topks_correct = topks_correct(
            self.video_preds, self.video_labels_class, ks
        )
        topks = [
            (x / self.video_preds.size(0)) * 100.0 for x in num_topks_correct
        ]
        assert len({len(ks), len(topks)}) == 1
        stats = {"split": "test_final"}
        for k, topk in zip(ks, topks):
            stats["top{}_acc".format(k)] = "{:.{prec}f}".format(topk, prec=2)
        _logger.log_json_stats(stats)

    def finalize_metrics_map(self):
        """
        Calculate and log the final ensembled metrics.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    self.clip_count, self.num_clips
                )
            )
            logger.warning(self.clip_count)

        np.save("gt.npy",self.video_labels_multi_class.numpy())
        np.save("pred.npy",self.video_preds.numpy())
        APs = compute_multiple_aps(self.video_labels_multi_class.numpy(),self.video_preds.sigmoid().numpy())
        mAP = np.mean([ap for ap in APs if ap >=0])
        stats = {"split": "test_final"}
        stats["mAP"] = "{:.{prec}f}".format(mAP, prec=4)
        _logger.log_json_stats(stats)

class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


def topks_correct(preds, labels, ks):
    """
    Given the predictions, labels, and a list of top-k values, compute the
    number of correct predictions for each top-k value.
    
    Args:
        preds (array): array of predictions. Dimension is batchsize
            N x ClassNum.
        labels (array): array of labels. Dimension is batchsize N.
        ks (list): list of top-k values. For example, ks = [1, 5] correspods
            to top-1 and top-5.
    
    Returns:
        topks_correct (list): list of numbers, where the `i`-th entry
            corresponds to the number of top-`ks[i]` correct predictions.
    """
    assert preds.size(0) == labels.size(
        0
    ), "Batch dim of predictions and labels must match"
    # Find the top max_k predictions for each sample
    _top_max_k_vals, top_max_k_inds = torch.topk(
        preds, max(ks), dim=1, largest=True, sorted=True
    )
    # (batch_size, max_k) -> (max_k, batch_size).
    top_max_k_inds = top_max_k_inds.t()
    # (batch_size, ) -> (max_k, batch_size).
    rep_max_k_labels = labels.view(1, -1).expand_as(top_max_k_inds)
    # (i, j) = 1 if top i-th prediction for the j-th sample is correct.
    top_max_k_correct = top_max_k_inds.eq(rep_max_k_labels)
    # Compute the number of topk correct predictions for each k.
    topks_correct = [
        top_max_k_correct[:k, :].view(-1).float().sum() for k in ks
    ]
    return topks_correct


def topk_errors(preds, labels, ks):
    """
    Computes the top-k error for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct]


def topk_accuracies(preds, labels, ks):
    """
    Computes the top-k accuracy for each k.
    Args:
        preds (array): array of predictions. Dimension is N.
        labels (array): array of labels. Dimension is N.
        ks (list): list of ks to calculate the top accuracies.
    """
    num_topks_correct = topks_correct(preds, labels, ks)
    return [(x / preds.size(0)) * 100.0 for x in num_topks_correct]

def compute_precision(groundtruth, predictions):
    predictions = np.asarray(predictions).squeeze()
    groundtruth = np.asarray(groundtruth, dtype=float).squeeze()
    if predictions.ndim != 1:
        raise ValueError('Predictions vector should be 1 dimensional.'
                         'For multiple labels, use `compute_multiple_aps`.')
    if groundtruth.ndim != 1:
        raise ValueError('Groundtruth vector should be 1 dimensional.'
                         'For multiple labels, use `compute_multiple_aps`.')

    sorted_indices = np.argsort(predictions)[::-1]
    predictions = predictions[sorted_indices]
    groundtruth = groundtruth[sorted_indices]
    # The false positives are all the negative groundtruth instances, since we
    # assume all instances were 'retrieved'. Ideally, these will be low scoring
    # and therefore in the end of the vector.
    false_positives = 1 - groundtruth

    tp = np.cumsum(groundtruth)      # tp[i] = # of positive examples up to i
    fp = np.cumsum(false_positives)  # fp[i] = # of false positives up to i

    num_positives = tp[-1]

    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    
    return precisions


def compute_multiple_precision(groundtruth, predictions):
    """Convenience function to compute APs for multiple labels.
    Args:
        groundtruth (np.array): Shape (num_samples, num_labels)
        predictions (np.array): Shape (num_samples, num_labels)
    Returns:
        aps_per_label (np.array, shape (num_labels,)): Contains APs for each
            label. NOTE: If a label does not have positive samples in the
            groundtruth, the AP is set to -1.
    """
    predictions = np.asarray(predictions)
    groundtruth = np.asarray(groundtruth)
    groundtruth = groundtruth.reshape((groundtruth.shape[0], -1))
    if predictions.ndim != 2:
        raise ValueError('Predictions should be 2-dimensional,'
                         ' but has shape %s' % (predictions.shape, ))
    if groundtruth.ndim != 2:
        raise ValueError('Groundtruth should be 2-dimensional,'
                         ' but has shape %s' % (predictions.shape, ))

    num_labels = groundtruth.shape[1]
    aps = np.zeros(groundtruth.shape[1])
    for i in range(num_labels):
        if not groundtruth[:, i].any():
            # print('WARNING: No groundtruth for label: %s' % i)
            aps[i] = 0
        else:
            aps[i] = np.average(compute_precision(groundtruth[:, i],
                                               predictions[:, i]))
    return aps


def compute_average_precision(groundtruth, predictions):
    """
    Computes average precision for a binary problem. This is based off of the
    PASCAL VOC implementation.
    Args:
        groundtruth (array-like): Binary vector indicating whether each sample
            is positive or negative.
        predictions (array-like): Contains scores for each sample.
    Returns:
        Average precision.
    """
    predictions = np.asarray(predictions).squeeze()
    groundtruth = np.asarray(groundtruth, dtype=float).squeeze()
    if predictions.ndim != 1:
        raise ValueError('Predictions vector should be 1 dimensional.'
                         'For multiple labels, use `compute_multiple_aps`.')
    if groundtruth.ndim != 1:
        raise ValueError('Groundtruth vector should be 1 dimensional.'
                         'For multiple labels, use `compute_multiple_aps`.')

    sorted_indices = np.argsort(predictions)[::-1]
    predictions = predictions[sorted_indices]
    groundtruth = groundtruth[sorted_indices]
    # The false positives are all the negative groundtruth instances, since we
    # assume all instances were 'retrieved'. Ideally, these will be low scoring
    # and therefore in the end of the vector.
    false_positives = 1 - groundtruth

    tp = np.cumsum(groundtruth)      # tp[i] = # of positive examples up to i
    fp = np.cumsum(false_positives)  # fp[i] = # of false positives up to i

    num_positives = tp[-1]

    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    recalls = tp / num_positives

    # Append end points of the precision recall curve.
    precisions = np.concatenate(([0.], precisions))
    recalls = np.concatenate(([0.], recalls))

    # Find points where prediction score changes.
    prediction_changes = set(
        np.where(predictions[1:] != predictions[:-1])[0] + 1)

    num_examples = predictions.shape[0]

    # Recall and scores always "change" at the first and last prediction.
    c = prediction_changes | set([0, num_examples])
    c = np.array(sorted(list(c)), dtype=np.int)

    precisions = precisions[c[1:]]

    # Set precisions[i] = max(precisions[j] for j >= i)
    # This is because (for j > i), recall[j] >= recall[i], so we can always use
    # a lower threshold to get the higher recall and higher precision at j.
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]

    ap = np.sum((recalls[c[1:]] - recalls[c[:-1]]) * precisions)

    return ap


def compute_multiple_aps(groundtruth, predictions):
    """Convenience function to compute APs for multiple labels.
    Args:
        groundtruth (np.array): Shape (num_samples, num_labels)
        predictions (np.array): Shape (num_samples, num_labels)
    Returns:
        aps_per_label (np.array, shape (num_labels,)): Contains APs for each
            label. NOTE: If a label does not have positive samples in the
            groundtruth, the AP is set to -1.
    """
    predictions = np.asarray(predictions)
    groundtruth = np.asarray(groundtruth)
    groundtruth = groundtruth.reshape((groundtruth.shape[0], -1))
    if predictions.ndim != 2:
        raise ValueError('Predictions should be 2-dimensional,'
                         ' but has shape %s' % (predictions.shape, ))
    if groundtruth.ndim != 2:
        raise ValueError('Groundtruth should be 2-dimensional,'
                         ' but has shape %s' % (predictions.shape, ))

    num_labels = groundtruth.shape[1]
    aps = np.zeros(groundtruth.shape[1])
    for i in range(num_labels):
        if not groundtruth[:, i].any():
            # print('WARNING: No groundtruth for label: %s' % i)
            aps[i] = 0
        else:
            aps[i] = compute_average_precision(groundtruth[:, i],
                                               predictions[:, i])
    return aps
