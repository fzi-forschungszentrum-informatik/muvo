import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from constants import SEMANTIC_SEG_WEIGHTS, VOXEL_SEG_WEIGHTS


class SegmentationLoss(nn.Module):
    def __init__(self, use_top_k=False, top_k_ratio=1.0, use_weights=False, poly_one=False, poly_one_coefficient=0.0,
                 is_bev=True):
        super().__init__()
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.use_weights = use_weights
        self.poly_one = poly_one
        self.poly_one_coefficient = poly_one_coefficient

        if self.use_weights:
            self.weights = SEMANTIC_SEG_WEIGHTS if is_bev else VOXEL_SEG_WEIGHTS

    def forward(self, prediction, target):
        b, s, c, h, w = prediction.shape

        prediction = prediction.view(b * s, c, h, w)
        target = target.view(b * s, h, w)

        if self.use_weights:
            weights = torch.tensor(self.weights, dtype=prediction.dtype, device=prediction.device)
        else:
            weights = None
        loss = F.cross_entropy(
            prediction,
            target,
            reduction='none',
            weight=weights,
        )

        if self.poly_one:
            prob = torch.exp(-loss)
            loss_poly_one = self.poly_one_coefficient * (1 - prob)
            loss = loss + loss_poly_one

        loss = loss.view(b, s, -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[2])
            loss = loss.topk(k, dim=-1)[0]

        return torch.mean(loss)


class RegressionLoss(nn.Module):
    def __init__(self, norm, channel_dim=-1):
        super().__init__()
        self.norm = norm
        self.channel_dim = channel_dim

        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, prediction, target):
        loss = self.loss_fn(prediction, target, reduction='none')

        # Sum channel dimension
        loss = torch.sum(loss, dim=self.channel_dim, keepdims=True)
        return loss.mean()


class SpatialRegressionLoss(nn.Module):
    def __init__(self, norm, ignore_index=255):
        super(SpatialRegressionLoss, self).__init__()
        self.norm = norm
        self.ignore_index = ignore_index

        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, prediction, target, instance_mask=None):
        assert len(prediction.shape) == 5, 'Must be a 5D tensor'
        # ignore_index is the same across all channels
        mask = instance_mask if instance_mask is not None else target[:, :, :1] != self.ignore_index
        if mask.sum() == 0:
            return prediction.new_zeros(1)[0].float()

        loss = self.loss_fn(prediction, target, reduction='none')

        # Sum channel dimension
        loss = torch.sum(loss, dim=-3, keepdims=True)

        return loss[mask].mean()


class ProbabilisticLoss(nn.Module):
    """ Given a prior distribution and a posterior distribution, this module computes KL(posterior, prior)"""

    def __init__(self, remove_first_timestamp=True):
        super().__init__()
        self.remove_first_timestamp = remove_first_timestamp

    def forward(self, prior_mu, prior_sigma, posterior_mu, posterior_sigma):
        posterior_var = posterior_sigma[:, 1:] ** 2
        prior_var = prior_sigma[:, 1:] ** 2

        posterior_log_sigma = torch.log(posterior_sigma[:, 1:])
        prior_log_sigma = torch.log(prior_sigma[:, 1:])

        kl_div = (
                prior_log_sigma - posterior_log_sigma - 0.5
                + (posterior_var + (posterior_mu[:, 1:] - prior_mu[:, 1:]) ** 2) / (2 * prior_var)
        )
        first_kl = - posterior_log_sigma[:, :1] - 0.5 + (posterior_var[:, :1] + posterior_mu[:, :1] ** 2) / 2
        kl_div = torch.cat([first_kl, kl_div], dim=1)

        # Sum across channel dimension
        # Average across batch dimension, keep time dimension for monitoring
        kl_loss = torch.mean(torch.sum(kl_div, dim=-1))
        return kl_loss


class KLLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.loss = ProbabilisticLoss(remove_first_timestamp=True)

    def forward(self, prior, posterior):
        prior_mu, prior_sigma = prior['mu'], prior['sigma']
        posterior_mu, posterior_sigma = posterior['mu'], posterior['sigma']
        prior_loss = self.loss(prior_mu, prior_sigma, posterior_mu.detach(), posterior_sigma.detach())
        posterior_loss = self.loss(prior_mu.detach(), prior_sigma.detach(), posterior_mu, posterior_sigma)

        return self.alpha * prior_loss + (1 - self.alpha) * posterior_loss


class VoxelLoss(nn.Module):
    """ 3D version of SegmentationLoss """

    def __init__(self, use_top_k=False, top_k_ratio=1.0, use_weights=False, poly_one=False, poly_one_coefficient=0.0):
        super().__init__()
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.use_weights = use_weights
        self.poly_one = poly_one
        self.poly_one_coefficient = poly_one_coefficient

        if self.use_weights:
            self.weights = VOXEL_SEG_WEIGHTS

    def forward(self, prediction, target):
        b, s, c, x, y, z = prediction.shape

        prediction = prediction.view(b * s, c, x, y, z)
        target = target.view(b * s, x, y, z)

        if self.use_weights:
            weights = torch.tensor(self.weights, dtype=prediction.dtype, device=prediction.device)
        else:
            weights = None
        loss = F.cross_entropy(
            prediction,
            target,
            reduction='none',
            weight=weights,
        )

        if self.poly_one:
            prob = torch.exp(-loss)
            loss_poly_one = self.poly_one_coefficient * (1 - prob)
            loss = loss + loss_poly_one

        loss = loss.view(b, s, -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[2])
            loss = loss.topk(k, dim=-1)[0]

        return torch.mean(loss)


# Scene-Class Affinity Loss proposed in MonoScene
# https://github.com/astra-vision/MonoScene/blob/master/monoscene/loss/ssc_loss.py
class SemScalLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, prediction, target):
        b, s, c, x, y, z = prediction.shape

        prediction = prediction.view(b * s, c, x, y, z)
        target = target.view(b * s, x, y, z)

        # Get softmax probabilities
        prediction = F.softmax(prediction, dim=1)
        loss = 0
        count = 0
        mask = target != self.ignore_index
        n_classes = prediction.shape[1]
        for i in range(0, n_classes):

            # Get probability of class i
            p = prediction[:, i, :, :, :]

            # Remove unknown voxels
            target_ori = target
            p = p[mask]
            target_mask = target[mask]

            completion_target = torch.ones_like(target_mask).float()
            completion_target[target_mask != i] = 0
            completion_target_ori = torch.ones_like(target_ori).float()
            completion_target_ori[target_ori != i] = 0
            if torch.sum(completion_target) > 0:
                count += 1.0
                nominator = torch.sum(p * completion_target)
                loss_class = 0
                with autocast(enabled=False):
                    if torch.sum(p) > 0:
                        precision = nominator / (torch.sum(p))
                        if 0 <= precision <= 1:
                            loss_precision = F.binary_cross_entropy(
                                precision, torch.ones_like(precision)
                            )
                            loss_class += loss_precision
                    if torch.sum(completion_target) > 0:
                        recall = nominator / (torch.sum(completion_target))
                        if 0 <= recall <= 1:
                            loss_recall = F.binary_cross_entropy(
                                recall, torch.ones_like(recall)
                            )
                            loss_class += loss_recall
                    if torch.sum(1 - completion_target) > 0:
                        specificity = torch.sum((1 - p) * (1 - completion_target)) / (
                            torch.sum(1 - completion_target)
                        )
                        if 0 <= specificity <= 1:
                            loss_specificity = F.binary_cross_entropy(
                                specificity, torch.ones_like(specificity)
                            )
                            loss_class += loss_specificity
                loss += loss_class
        return loss / count


class GeoScalLoss(nn.Module):
    def __init__(self, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, prediction, target):
        b, s, c, x, y, z = prediction.shape

        prediction = prediction.view(b * s, c, x, y, z)
        target = target.view(b * s, x, y, z)

        # Get softmax probabilities
        prediction = F.softmax(prediction, dim=1)

        # Compute empty and nonempty probabilities
        empty_probs = prediction[:, 0, :, :, :]
        nonempty_probs = 1 - empty_probs

        # Remove unknown voxels
        mask = target != self.ignore_index
        nonempty_target = target != 0
        nonempty_target = nonempty_target[mask].float()
        nonempty_probs = nonempty_probs[mask]
        empty_probs = empty_probs[mask]

        intersection = (nonempty_target * nonempty_probs).sum()
        precision = intersection / nonempty_probs.sum()
        recall = intersection / nonempty_target.sum()
        spec = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
        with autocast(enabled=False):
            loss = F.binary_cross_entropy(precision, torch.ones_like(precision)) + \
                   F.binary_cross_entropy(recall, torch.ones_like(recall)) + \
                   F.binary_cross_entropy(spec, torch.ones_like(spec))
        return loss


# Structure Similarity Index Measure,
# modified from https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
class SSIMLoss(nn.Module):
    def __init__(self, channel=1, window_size=11, sigma=1.5, L=1, non_negative=False):
        super().__init__()
        self.window_size = window_size
        # self.size_average = size_average
        self.channel = channel
        self.sigma = sigma
        self.C1 = (0.01 * L) ** 2
        self.C2 = (0.03 * L) ** 2
        self.window = self.create_window()
        self.non_negative = non_negative

    def gaussian(self, window_size, sigma):
        x = torch.arange(window_size)
        gauss = torch.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        return gauss / gauss.sum()

    # gaussian kernel
    def create_window(self):
        _1D_window = self.gaussian(self.window_size, self.sigma).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(self.channel, 1, self.window_size, self.window_size).contiguous()
        return window

    def _ssim(self, prediction, target):
        window = torch.as_tensor(self.window, dtype=prediction.dtype, device=prediction.device)

        padd = 0
        # padd = self.window_size // 2
        mu1 = F.conv2d(target, window, padding=padd, groups=self.channel)
        mu2 = F.conv2d(prediction, window, padding=padd, groups=self.channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(target * target, window, padding=padd, groups=self.channel) - mu1_sq
        sigma2_sq = F.conv2d(prediction * prediction, window, padding=padd, groups=self.channel) - mu2_sq
        sigma12 = F.conv2d(target * prediction, window, padding=padd, groups=self.channel) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)) / \
                   ((mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2))

        ssim_batch = ssim_map.mean([1, 2, 3])
        if self.non_negative:
            ssim_batch = F.relu(ssim_batch)

        return ssim_batch

    def forward(self, prediction, target):
        b, s, c, h, w = prediction.shape

        prediction = prediction.view(b * s, c, h, w)
        target = target.view(b * s, c, h, w)

        loss = self._ssim(prediction, target)
        return loss.mean()


# Chamfer Distance
class CDLoss(nn.Module):
    def __init__(self, reducer=torch.mean):
        super().__init__()
        self.reducer = reducer

    def forward(self, prediction, target):
        b, s, n, d = prediction.shape

        prediction = prediction.view(b * s, n, d)
        target = target.view(b * s, n, d)
        # dist = self.batch_pairwise_dist(prediction, target)
        # point-to-point distance
        dist = torch.cdist(prediction.float(), target.float(), 2)   # b*s, n, n
        dl, dr = dist.min(1)[0], dist.min(2)[0]
        loss = self.reducer(dl, dim=1) + self.reducer(dr, dim=1)
        return loss.mean()

    @staticmethod
    def batch_pairwise_dist(x: torch.Tensor, y: torch.Tensor):
        x_norm = torch.sum(x ** 2, dim=2, keepdim=True)
        y_norm = torch.sum(y ** 2, dim=2, keepdim=True)
        xy = torch.bmm(x, y.transpose(1, 2))
        dist = x_norm - 2 * xy + y_norm.transpose(1, 2)
        return dist
