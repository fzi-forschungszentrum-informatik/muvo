"""
code is taken from https://github.com/astra-vision/MonoScene/blob/master/monoscene/loss/sscMetrics.py

Part of the code is taken from https://github.com/waterljwant/SSC/blob/master/sscMetrics.py
"""
# import numpy as np
import torch
from chamferdist import ChamferDistance
# from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from muvo.losses import SSIMLoss, CDLoss


# SSCMetrics code is modified from https://github.com/astra-vision/MonoScene/blob/master/monoscene/loss/sscMetrics.py
def get_iou(iou_sum, cnt_class):
    _C = iou_sum.shape[0]  # 12
    iou = torch.zeros(_C, dtype=torch.float32)  # iou for each class
    for idx in range(_C):
        iou[idx] = iou_sum[idx] / cnt_class[idx] if cnt_class[idx] else 0

    mean_iou = torch.sum(iou[1:]) / torch.count_nonzero(cnt_class[1:])
    return iou, mean_iou


def get_accuracy(predict, target, weight=None):  # 0.05s
    _bs = predict.shape[0]  # batch size
    _C = predict.shape[1]  # _C = 12
    target = target.int32()
    target = target.reshape(_bs, -1)  # (_bs, 60*36*60) 129600
    predict = predict.reshape(_bs, _C, -1)  # (_bs, _C, 60*36*60)
    predict = torch.argmax(
        predict, dim=1
    )  # one-hot: _bs x _C x 60*36*60 -->  label: _bs x 60*36*60.

    correct = predict == target  # (_bs, 129600)
    if weight:  # 0.04s, add class weights
        weight_k = torch.ones(target.shape)
        for i in range(_bs):
            for n in range(target.shape[1]):
                idx = 0 if target[i, n] == 255 else target[i, n]
                weight_k[i, n] = weight[idx]
        correct = correct * weight_k
    acc = correct.sum() / correct.size
    return acc


class SSCMetrics:
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.reset()

    def hist_info(self, n_cl, pred, gt):
        assert pred.shape == gt.shape
        k = (gt >= 0) & (gt < n_cl)  # exclude 255
        labeled = torch.sum(k)
        correct = torch.sum((pred[k] == gt[k]))

        return (
            torch.bincount(
                n_cl * gt[k].astype(int) + pred[k].astype(int), minlength=n_cl ** 2
            ).reshape(n_cl, n_cl),
            correct,
            labeled,
        )

    @staticmethod
    def compute_score(hist, correct, labeled):
        iu = torch.diag(hist) / (hist.sum(1) + hist.sum(0) - torch.diag(hist))
        mean_IU = torch.nanmean(iu)
        mean_IU_no_back = torch.nanmean(iu[1:])
        freq = hist.sum(1) / hist.sum()
        freq_IU = (iu[freq > 0] * freq[freq > 0]).sum()
        mean_pixel_acc = correct / labeled if labeled != 0 else 0

        return iu, mean_IU, mean_IU_no_back, mean_pixel_acc

    def add_batch(self, y_pred, y_true, nonempty=None, nonsurface=None):
        self.count += 1
        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        if nonsurface is not None:
            mask = mask & nonsurface
        tp, fp, fn = self.get_score_completion(y_pred, y_true, mask)

        self.completion_tp += tp
        self.completion_fp += fp
        self.completion_fn += fn

        mask = y_true != 255
        if nonempty is not None:
            mask = mask & nonempty
        tp_sum, fp_sum, fn_sum = self.get_score_semantic_and_completion(
            y_pred, y_true, mask
        )
        self.tps += tp_sum
        self.fps += fp_sum
        self.fns += fn_sum

        self.compute()

    def compute(self):
        if self.completion_tp != 0:
            self.precision = self.completion_tp / (self.completion_tp + self.completion_fp)
            self.recall = self.completion_tp / (self.completion_tp + self.completion_fn)
            self.iou = self.completion_tp / (
                    self.completion_tp + self.completion_fp + self.completion_fn
            )
        else:
            self.precision, self.recall, self.iou = 0, 0, 0

        self.iou_ssc = self.tps / (self.tps + self.fps + self.fns + 1e-5)

    def get_stats(self):
        return {
            "precision": self.precision,
            "recall": self.recall,
            "iou": self.iou,
            "iou_ssc": self.iou_ssc,
            "iou_ssc_mean": torch.mean(self.iou_ssc[1:]),
        }

    def reset(self):

        self.completion_tp = 0
        self.completion_fp = 0
        self.completion_fn = 0
        self.tps = torch.zeros(self.n_classes)
        self.fps = torch.zeros(self.n_classes)
        self.fns = torch.zeros(self.n_classes)

        self.hist_ssc = torch.zeros((self.n_classes, self.n_classes))
        self.labeled_ssc = 0
        self.correct_ssc = 0

        self.precision = 0
        self.recall = 0
        self.iou = 0
        self.count = 1e-8
        self.iou_ssc = torch.zeros(self.n_classes, dtype=torch.float32)
        self.cnt_class = torch.zeros(self.n_classes, dtype=torch.float32)

    def get_score_completion(self, predict, target, nonempty=None):
        predict = predict.clone().detach()
        target = target.clone().detach()

        """for scene completion, treat the task as two-classes problem, just empty or occupancy"""
        _bs = predict.shape[0]  # batch size
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.reshape(_bs, -1)  # (_bs, 129600)
        predict = predict.reshape(_bs, -1)  # (_bs, _C, 129600), 60*36*60=129600
        # ---- treat all non-empty object class as one category, set them to label 1
        b_pred = predict.new_zeros(predict.shape)
        b_true = target.new_zeros(target.shape)
        b_pred[predict > 0] = 1
        b_true[target > 0] = 1
        p, r, iou = 0.0, 0.0, 0.0
        tp_sum, fp_sum, fn_sum = 0, 0, 0
        for idx in range(_bs):
            y_true = b_true[idx, :]  # GT
            y_pred = b_pred[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                y_true = y_true[nonempty_idx == 1]
                y_pred = y_pred[nonempty_idx == 1]

            tp = torch.stack(torch.where(torch.logical_and(y_true == 1, y_pred == 1))).numel()
            fp = torch.stack(torch.where(torch.logical_and(y_true != 1, y_pred == 1))).numel()
            fn = torch.stack(torch.where(torch.logical_and(y_true == 1, y_pred != 1))).numel()
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
        return tp_sum, fp_sum, fn_sum

    def get_score_semantic_and_completion(self, predict, target, nonempty=None):
        target = target.clone().detach()
        predict = predict.clone().detach()
        _bs = predict.shape[0]  # batch size
        _C = self.n_classes  # _C = 12
        # ---- ignore
        predict[target == 255] = 0
        target[target == 255] = 0
        # ---- flatten
        target = target.reshape(_bs, -1)  # (_bs, 129600)
        predict = predict.reshape(_bs, -1)  # (_bs, 129600), 60*36*60=129600

        cnt_class = torch.zeros(_C, dtype=torch.int32)  # count for each class
        iou_sum = torch.zeros(_C, dtype=torch.float32)  # sum of iou for each class
        tp_sum = torch.zeros(_C, dtype=torch.int32)  # tp
        fp_sum = torch.zeros(_C, dtype=torch.int32)  # fp
        fn_sum = torch.zeros(_C, dtype=torch.int32)  # fn

        for idx in range(_bs):
            y_true = target[idx, :]  # GT
            y_pred = predict[idx, :]
            if nonempty is not None:
                nonempty_idx = nonempty[idx, :].reshape(-1)
                y_pred = y_pred[
                    torch.where(torch.logical_and(nonempty_idx == 1, y_true != 255))
                ]
                y_true = y_true[
                    torch.where(torch.logical_and(nonempty_idx == 1, y_true != 255))
                ]
            for j in range(_C):  # for each class
                tp = torch.stack(torch.where(torch.logical_and(y_true == j, y_pred == j))).numel()
                fp = torch.stack(torch.where(torch.logical_and(y_true != j, y_pred == j))).numel()
                fn = torch.stack(torch.where(torch.logical_and(y_true == j, y_pred != j))).numel()

                tp_sum[j] += tp
                fp_sum[j] += fp
                fn_sum[j] += fn

        return tp_sum, fp_sum, fn_sum


class SSIMMetric:
    def __init__(self, channel=3, window_size=11, sigma=1.5, L=1, non_negative=False):
        self.ssim = SSIMLoss(channel=channel, window_size=window_size, sigma=sigma, L=L, non_negative=non_negative)
        self.reset()

    def add_batch(self, prediction, target):
        self.count += 1
        self.ssim_score += self.ssim(prediction, target)
        self.ssim_avg = self.ssim_score / self.count

    def get_stat(self):
        return self.ssim_avg

    def reset(self):
        self.ssim_score = 0
        self.count = 1e-8
        self.ssim_avg = 0


class CDMetric:
    def __init__(self, reducer=torch.mean):
        self.reducer = reducer
        self.reset()

    def add_batch(self, prediction, target):
        self.count += 1
        # dist = CDLoss.batch_pairwise_dist(prediction.float(), target.float()).cpu().numpy()
        dist = torch.cdist(prediction.float(), target.float(), 2)
        dl, dr = dist.min(1)[0], dist.min(2)[0]
        cost = (self.reducer(dl, dim=1) + self.reducer(dr, dim=1)) / 2
        self.total_cost += cost.mean()
        self.avg_cost = self.total_cost / self.count

    def get_stat(self):
        return self.avg_cost

    def reset(self):
        self.total_cost = 0
        self.count = 1e-8
        self.avg_cost = 0


class CDMetric0:
    def __init__(self):
        self.chamferDist = ChamferDistance()
        self.reset()

    def add_batch(self, prediction, target, valid_pred, valid_target):
        self.count += 1
        b = prediction.shape[0]
        cdist = 0
        for i in range(b):
            # cdist += 0.5 * self.chamferDist(prediction[i][valid_pred[i]][None].float(),
            #                                 target[i][valid_target[i]][None].float(),
            #                                 bidirectional=True).detach().cpu().item()
            pred_pcd = prediction[i][valid_pred[i]]
            target_pcd = target[i][valid_target[i]]
            cd_forward = self.chamferDist(pred_pcd[None].float(),
                                          target_pcd[None].float(),
                                          point_reduction='mean').detach().cpu().item()
            cd_backward = self.chamferDist(target_pcd[None].float(),
                                           pred_pcd[None].float(),
                                           point_reduction='mean').detach().cpu().item()
            cdist += 0.5 * (cd_forward + cd_backward)
        self.total_cost += cdist / b
        self.avg_cost = self.total_cost / self.count

    def get_stat(self):
        return self.avg_cost

    def reset(self):
        self.total_cost = 0
        self.count = 1e-8
        self.avg_cost = 0


class PSNRMetric:
    def __init__(self, max_pixel_val=1.0):
        self.max_pixel_value = max_pixel_val
        self.reset()

    def add_batch(self, prediction, target):
        self.count += 1
        self.total_psnr += self.psnr(prediction, target).mean()
        self.avg_psnr = self.total_psnr / self.count

    def psnr(self, prediction, target):
        # b, s, c, h, w
        mse = torch.mean((prediction - target) ** 2, dim=(2, 3, 4))
        psnr = 20 * torch.log10(self.max_pixel_value / torch.sqrt(mse))
        return psnr

    def get_stat(self):
        return self.avg_psnr

    def reset(self):
        self.total_psnr = 0
        self.count = 1e-8
        self.avg_psnr = 0
