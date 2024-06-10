import os

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from torchmetrics import JaccardIndex

from muvo.config import get_cfg
from muvo.models.mile import Mile
from muvo.losses import \
    SegmentationLoss, KLLoss, RegressionLoss, SpatialRegressionLoss, VoxelLoss, SSIMLoss, SemScalLoss, GeoScalLoss
from muvo.metrics import SSCMetrics, SSIMMetric, CDMetric, PSNRMetric
from muvo.models.preprocess import PreProcess
from muvo.utils.geometry_utils import PointCloud, compute_pcd_transformation
from constants import BIRDVIEW_COLOURS, VOXEL_COLOURS, VOXEL_LABEL

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')


class WorldModelTrainer(pl.LightningModule):
    def __init__(self, hparams, path_to_conf_file=None, pretrained_path=None):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = get_cfg(cfg_dict=hparams)
        if path_to_conf_file:
            self.cfg.merge_from_file(path_to_conf_file)
        if pretrained_path:
            self.cfg.PRETRAINED.PATH = pretrained_path
        # print(self.cfg)
        self.vis_step = -1
        self.rf = self.cfg.RECEPTIVE_FIELD
        self.fh = self.cfg.FUTURE_HORIZON

        self.cml_logger = None
        self.preprocess = PreProcess(self.cfg)

        # Model
        self.model = Mile(self.cfg)
        self.load_pretrained_weights()

        # self.metrics_vals = [dict() for _ in range(len(self.val_dataloader()))]
        # self.metrics_vals_imagine = [dict() for _ in range(len(self.val_dataloader()))]
        # self.metrics_tests = [dict() for _ in range(len(self.test_dataloader()))]
        # self.metrics_tests_imagine = [dict() for _ in range(len(self.test_dataloader()))]
        # self.metrics_train = dict()
        self.metrics_vals = [{}, {}, {}]
        self.metrics_vals_imagine = [{}, {}, {}]
        self.metrics_tests = [{}, {}, {}]
        self.metrics_tests_imagine = [{}, {}, {}]
        self.metrics_train = dict()

        # Losses
        self.action_loss = RegressionLoss(norm=1)
        if self.cfg.MODEL.TRANSITION.ENABLED:
            self.probabilistic_loss = KLLoss(alpha=self.cfg.LOSSES.KL_BALANCING_ALPHA)

        if self.cfg.SEMANTIC_SEG.ENABLED:
            self.segmentation_loss = SegmentationLoss(
                use_top_k=self.cfg.SEMANTIC_SEG.USE_TOP_K,
                top_k_ratio=self.cfg.SEMANTIC_SEG.TOP_K_RATIO,
                use_weights=self.cfg.SEMANTIC_SEG.USE_WEIGHTS,
                is_bev=True,
            )

            self.center_loss = SpatialRegressionLoss(norm=2)
            self.offset_loss = SpatialRegressionLoss(norm=1, ignore_index=self.cfg.INSTANCE_SEG.IGNORE_INDEX)

            for metrics_val, metrics_val_imagine in zip(self.metrics_vals, self.metrics_vals_imagine):
                metrics_val['iou'] = JaccardIndex(
                    task='multiclass', num_classes=self.cfg.SEMANTIC_SEG.N_CHANNELS, average='none',
                )
                metrics_val_imagine['iou'] = JaccardIndex(
                    task='multiclass', num_classes=self.cfg.SEMANTIC_SEG.N_CHANNELS, average='none',
                )

            for metrics_test, metrics_test_imagine in zip(self.metrics_tests, self.metrics_tests_imagine):
                metrics_test['iou'] = JaccardIndex(
                    task='multiclass', num_classes=self.cfg.SEMANTIC_SEG.N_CHANNELS, average='none',
                )
                metrics_test_imagine['iou'] = JaccardIndex(
                    task='multiclass', num_classes=self.cfg.SEMANTIC_SEG.N_CHANNELS, average='none',
                )

            # self.metrics_train['iou'] = JaccardIndex(
            #     task='multiclass', num_classes=self.cfg.SEMANTIC_SEG.N_CHANNELS, average='none',
            # )

        if self.cfg.EVAL.RGB_SUPERVISION:
            self.rgb_loss = SpatialRegressionLoss(norm=1)
            if self.cfg.LOSSES.RGB_INSTANCE:
                self.rgb_instance_loss = SpatialRegressionLoss(norm=1)
            if self.cfg.LOSSES.SSIM:
                self.ssim_loss = SSIMLoss(channel=3)

            for metrics_val, metrics_val_imagine in zip(self.metrics_vals, self.metrics_vals_imagine):
                metrics_val['ssim'] = SSIMMetric(channel=3)
                metrics_val_imagine['ssim'] = SSIMMetric(channel=3)
                metrics_val['psnr'] = PSNRMetric(max_pixel_val=1.0)
                metrics_val_imagine['psnr'] = PSNRMetric(max_pixel_val=1.0)
            for metrics_test, metrics_test_imagine in zip(self.metrics_tests, self.metrics_tests_imagine):
                metrics_test['ssim'] = SSIMMetric(channel=3)
                metrics_test_imagine['ssim'] = SSIMMetric(channel=3)
                metrics_test['psnr'] = PSNRMetric(max_pixel_val=1.0)
                metrics_test_imagine['psnr'] = PSNRMetric(max_pixel_val=1.0)
            # self.metrics_train['ssim'] = SSIMMetric(channel=3)
            # self.metrics_train['psnr'] = PSNRMetric(max_pixel_val=1.0)

        if self.cfg.LIDAR_RE.ENABLED:
            self.lidar_re_loss = SpatialRegressionLoss(norm=2)
            self.lidar_depth_loss = SpatialRegressionLoss(norm=1)
            # self.lidar_cd_loss = CDLoss()
            self.pcd = PointCloud(
                self.cfg.POINTS.CHANNELS,
                self.cfg.POINTS.HORIZON_RESOLUTION,
                *self.cfg.POINTS.FOV,
                self.cfg.POINTS.LIDAR_POSITION
            )

            for metrics_val, metrics_val_imagine in zip(self.metrics_vals, self.metrics_vals_imagine):
                metrics_val['cd'] = CDMetric()
                metrics_val_imagine['cd'] = CDMetric()
            for metrics_test, metrics_test_imagine in zip(self.metrics_tests, self.metrics_tests_imagine):
                metrics_test['cd'] = CDMetric()
                metrics_test_imagine['cd'] = CDMetric()
            # self.metrics_train['cd'] = CDMetric()

        if self.cfg.LIDAR_SEG.ENABLED:
            self.lidar_seg_loss = SegmentationLoss(
                use_top_k=self.cfg.LIDAR_SEG.USE_TOP_K,
                top_k_ratio=self.cfg.LIDAR_SEG.TOP_K_RATIO,
                use_weights=self.cfg.LIDAR_SEG.USE_WEIGHTS,
                is_bev=False,
            )

            for metrics_val, metrics_val_imagine in zip(self.metrics_vals, self.metrics_vals_imagine):
                metrics_val['pcd_iou'] = JaccardIndex(
                    task='multiclass', num_classes=self.cfg.LIDAR_SEG.N_CLASSES, average='none',
                )
                metrics_val_imagine['pcd_iou'] = JaccardIndex(
                    task='multiclass', num_classes=self.cfg.LIDAR_SEG.N_CLASSES, average='none',
                )

            for metrics_test, metrics_test_imagine in zip(self.metrics_tests, self.metrics_tests_imagine):
                metrics_test['pcd_iou'] = JaccardIndex(
                    task='multiclass', num_classes=self.cfg.LIDAR_SEG.N_CLASSES, average='none',
                )
                metrics_test_imagine['pcd_iou'] = JaccardIndex(
                    task='multiclass', num_classes=self.cfg.LIDAR_SEG.N_CLASSES, average='none',
                )

        if self.cfg.SEMANTIC_IMAGE.ENABLED:
            self.sem_image_loss = SegmentationLoss(
                use_top_k=self.cfg.SEMANTIC_IMAGE.USE_TOP_K,
                top_k_ratio=self.cfg.SEMANTIC_IMAGE.TOP_K_RATIO,
                use_weights=self.cfg.SEMANTIC_IMAGE.USE_WEIGHTS,
                is_bev=False,
            )

            for metrics_val, metrics_val_imagine in zip(self.metrics_vals, self.metrics_vals_imagine):
                metrics_val['image_iou'] = JaccardIndex(
                    task='multiclass', num_classes=self.cfg.SEMANTIC_IMAGE.N_CLASSES, average='none',
                )
                metrics_val_imagine['image_iou'] = JaccardIndex(
                    task='multiclass', num_classes=self.cfg.SEMANTIC_IMAGE.N_CLASSES, average='none',
                )

            for metrics_test, metrics_test_imagine in zip(self.metrics_tests, self.metrics_tests_imagine):
                metrics_test['image_iou'] = JaccardIndex(
                    task='multiclass', num_classes=self.cfg.SEMANTIC_IMAGE.N_CLASSES, average='none',
                )
                metrics_test_imagine['image_iou'] = JaccardIndex(
                    task='multiclass', num_classes=self.cfg.SEMANTIC_IMAGE.N_CLASSES, average='none',
                )

        if self.cfg.DEPTH.ENABLED:
            self.depth_image_loss = SpatialRegressionLoss(norm=1)

        if self.cfg.VOXEL_SEG.ENABLED:
            self.voxel_loss = VoxelLoss(
                use_top_k=self.cfg.VOXEL_SEG.USE_TOP_K,
                top_k_ratio=self.cfg.VOXEL_SEG.TOP_K_RATIO,
                use_weights=self.cfg.VOXEL_SEG.USE_WEIGHTS,
            )
            self.sem_scal_loss = SemScalLoss()
            self.geo_scal_loss = GeoScalLoss()
            for metrics_val, metrics_val_imagine in zip(self.metrics_vals, self.metrics_vals_imagine):
                metrics_val['ssc'] = SSCMetrics(self.cfg.VOXEL_SEG.N_CLASSES)
                metrics_val_imagine['ssc'] = SSCMetrics(self.cfg.VOXEL_SEG.N_CLASSES)
            for metrics_test, metrics_test_imagine in zip(self.metrics_tests, self.metrics_tests_imagine):
                metrics_test['ssc'] = SSCMetrics(self.cfg.VOXEL_SEG.N_CLASSES)
                metrics_test_imagine['ssc'] = SSCMetrics(self.cfg.VOXEL_SEG.N_CLASSES)
            # self.metrics_train['ssc'] = SSCMetrics(self.cfg.VOXEL_SEG.N_CLASSES)

    def get_cml_logger(self, cml_logger):
        self.cml_logger = cml_logger

    def load_pretrained_weights(self):
        if self.cfg.PRETRAINED.PATH:
            if os.path.isfile(self.cfg.PRETRAINED.PATH):
                checkpoint = torch.load(self.cfg.PRETRAINED.PATH, map_location='cpu')['state_dict']
                checkpoint = {key[6:]: value for key, value in checkpoint.items() if key[:5] == 'model'}

                self.model.load_state_dict(checkpoint, strict=True)
                print(f'Loaded weights from: {self.cfg.PRETRAINED.PATH}')
            else:
                raise FileExistsError(self.cfg.PRETRAINED.PATH)

    def forward(self, batch, deployment=False):
        batch = self.preprocess(batch)
        output, state_dict = self.model.forward(batch, deployment=deployment)
        return output, state_dict

    def deployment_forward(self, batch, is_dreaming):
        batch = self.preprocess(batch)
        output = self.model.deployment_forward(batch, is_dreaming)
        return output

    def shared_step(self, batch, mode='train', predict_action=False):
        n_prediction_samples = self.cfg.PREDICTION.N_SAMPLES
        output_imagines = []
        losses_imagines = []

        if mode == 'train':
            # in training, only reconstruction
            output, state_dict = self.forward(batch)
            losses = self.compute_loss(batch, output)
        else:
            batch = self.preprocess(batch)
            batch_rf = {key: value[:, :self.rf] for key, value in batch.items()}  # dim (b, s, 512)
            batch_fh = {key: value[:, self.rf:] for key, value in batch.items()}  # dim (b, s, 512)
            output, state_dict = self.model.forward(batch_rf, deployment=False)
            losses = self.compute_loss(batch_rf, output)

            # in evaluation, do imagination (prediction)
            state_imagine = {'hidden_state': state_dict['posterior']['hidden_state'][:, -1],
                             'sample': state_dict['posterior']['sample'][:, -1],
                             'throttle_brake': batch['throttle_brake'][:, self.rf:],
                             'steering': batch['steering'][:, self.rf:]}
            for _ in range(n_prediction_samples):
                output_imagine = self.model.imagine(state_imagine, predict_action=predict_action, future_horizon=self.fh)
                output_imagines.append(output_imagine)
                losses_imagines.append(self.compute_loss(batch_fh, output_imagine))

        return losses, output, losses_imagines, output_imagines

    def compute_loss(self, batch, output):
        losses = dict()

        action_weight = self.cfg.LOSSES.WEIGHT_ACTION
        if 'throttle_brake' in output.keys():
            losses['throttle_brake'] = action_weight * self.action_loss(output['throttle_brake'],
                                                                        batch['throttle_brake'])
        if 'steering' in output.keys():
            losses['steering'] = action_weight * self.action_loss(output['steering'], batch['steering'])

        if self.cfg.MODEL.TRANSITION.ENABLED:
            if 'prior' in output.keys() and 'posterior' in output.keys():
                probabilistic_loss = self.probabilistic_loss(output['prior'], output['posterior'])

                losses['probabilistic'] = self.cfg.LOSSES.WEIGHT_PROBABILISTIC * probabilistic_loss

        # compute losses in down-sampling scale 1, 2, 4, separately.
        if self.cfg.SEMANTIC_SEG.ENABLED:
            for downsampling_factor in [1, 2, 4]:
                bev_segmentation_loss = self.segmentation_loss(
                    prediction=output[f'bev_segmentation_{downsampling_factor}'],
                    target=batch[f'birdview_label_{downsampling_factor}'],
                )
                discount = 1 / downsampling_factor
                losses[f'bev_segmentation_{downsampling_factor}'] = discount * self.cfg.LOSSES.WEIGHT_SEGMENTATION * \
                                                                    bev_segmentation_loss

                center_loss = self.center_loss(
                    prediction=output[f'bev_instance_center_{downsampling_factor}'],
                    target=batch[f'center_label_{downsampling_factor}']
                )
                offset_loss = self.offset_loss(
                    prediction=output[f'bev_instance_offset_{downsampling_factor}'],
                    target=batch[f'offset_label_{downsampling_factor}']
                )

                center_loss = self.cfg.INSTANCE_SEG.CENTER_LOSS_WEIGHT * center_loss
                offset_loss = self.cfg.INSTANCE_SEG.OFFSET_LOSS_WEIGHT * offset_loss

                losses[f'bev_center_{downsampling_factor}'] = discount * self.cfg.LOSSES.WEIGHT_INSTANCE * center_loss
                # Offset are already discounted in the labels
                losses[f'bev_offset_{downsampling_factor}'] = self.cfg.LOSSES.WEIGHT_INSTANCE * offset_loss

        if self.cfg.EVAL.RGB_SUPERVISION:
            for downsampling_factor in [1, 2, 4]:
                rgb_weight = 0.1
                discount = 1 / downsampling_factor
                rgb_loss = self.rgb_loss(
                    prediction=output[f'rgb_{downsampling_factor}'],
                    target=batch[f'rgb_label_{downsampling_factor}'],
                )

                if self.cfg.LOSSES.RGB_INSTANCE:
                    rgb_instance_loss = self.rgb_instance_loss(
                        prediction=output[f'rgb_{downsampling_factor}'],
                        target=batch[f'rgb_label_{downsampling_factor}'],
                        instance_mask=batch[f'image_instance_mask_{downsampling_factor}']
                    )
                else:
                    rgb_instance_loss = 0

                if self.cfg.LOSSES.SSIM:
                    ssim_loss = 1 - self.ssim_loss(
                        prediction=output[f'rgb_{downsampling_factor}'],
                        target=batch[f'rgb_label_{downsampling_factor}'],
                    )
                    ssim_weight = 0.6
                    losses[f'ssim_{downsampling_factor}'] = rgb_weight * discount * ssim_loss * ssim_weight

                losses[f'rgb_{downsampling_factor}'] = \
                    rgb_weight * discount * (rgb_loss + 0.5 * rgb_instance_loss)

        if self.cfg.LIDAR_RE.ENABLED:
            for downsampling_factor in [1, 2, 4]:
                discount = 1 / downsampling_factor
                lidar_re_loss = self.lidar_re_loss(
                    prediction=output[f'lidar_reconstruction_{downsampling_factor}'][:, :, :3, :, :],
                    target=batch[f'range_view_label_{downsampling_factor}'][:, :, :3, :, :]
                )
                lidar_depth_loss = self.lidar_depth_loss(
                    prediction=output[f'lidar_reconstruction_{downsampling_factor}'][:, :, -1:, :, :],
                    target=batch[f'range_view_label_{downsampling_factor}'][:, :, -1:, :, :]
                )
                losses[f'lidar_re_{downsampling_factor}'] = lidar_re_loss * discount * self.cfg.LOSSES.WEIGHT_LIDAR_RE
                losses[
                    f'lidar_depth_{downsampling_factor}'] = lidar_depth_loss * discount * self.cfg.LOSSES.WEIGHT_LIDAR_RE

        if self.cfg.LIDAR_SEG.ENABLED:
            for downsampling_factor in [1, 2, 4]:
                discount = 1 / downsampling_factor
                lidar_seg_loss = self.lidar_seg_loss(
                    prediction=output[f'lidar_segmentation_{downsampling_factor}'],
                    target=batch[f'range_view_seg_label_{downsampling_factor}']
                )
                losses[f'lidar_seg_{downsampling_factor}'] = \
                    lidar_seg_loss * discount * self.cfg.LOSSES.WEIGHT_LIDAR_SEG

        if self.cfg.SEMANTIC_IMAGE.ENABLED:
            for downsampling_factor in [1, 2, 4]:
                discount = 1 / downsampling_factor
                sem_image_loss = self.sem_image_loss(
                    prediction=output[f'semantic_image_{downsampling_factor}'],
                    target=batch[f'semantic_image_label_{downsampling_factor}']
                )
                losses[f'semantic_image_{downsampling_factor}'] = \
                    sem_image_loss * discount * self.cfg.LOSSES.WEIGHT_SEM_IMAGE

        if self.cfg.DEPTH.ENABLED:
            for downsampling_factor in [1, 2, 4]:
                discount = 1 / downsampling_factor
                depth_image_loss = self.depth_image_loss(
                    prediction=output[f'depth_{downsampling_factor}'],
                    target=batch[f'depth_label_{downsampling_factor}']
                )
                losses[f'depth_{downsampling_factor}'] = \
                    depth_image_loss * discount * self.cfg.LOSSES.WEIGHT_DEPTH

        if self.cfg.VOXEL_SEG.ENABLED:
            for downsampling_factor in [1, 2, 4]:
                discount = 1 / downsampling_factor
                voxel_loss = self.voxel_loss(
                    prediction=output[f'voxel_{downsampling_factor}'],
                    target=batch[f'voxel_label_{downsampling_factor}'].type(torch.long)
                )
                sem_scal_loss = self.sem_scal_loss(
                    prediction=output[f'voxel_{downsampling_factor}'],
                    target=batch[f'voxel_label_{downsampling_factor}']
                )
                geo_scal_loss = self.geo_scal_loss(
                    prediction=output[f'voxel_{downsampling_factor}'],
                    target=batch[f'voxel_label_{downsampling_factor}']
                )
                losses[f'voxel_{downsampling_factor}'] = discount * self.cfg.LOSSES.WEIGHT_VOXEL * voxel_loss
                losses[f'sem_scal_{downsampling_factor}'] = discount * self.cfg.LOSSES.WEIGHT_VOXEL * sem_scal_loss
                losses[f'geo_scal_{downsampling_factor}'] = discount * self.cfg.LOSSES.WEIGHT_VOXEL * geo_scal_loss

        if self.cfg.MODEL.REWARD.ENABLED:
            reward_loss = self.action_loss(output['reward'], batch['reward'])
            losses['reward'] = self.cfg.LOSSES.WEIGHT_REWARD * reward_loss
        return losses

    def training_step(self, batch, batch_idx):
        if batch_idx == self.cfg.STEPS and self.cfg.MODEL.TRANSITION.ENABLED:
            print('!' * 50)
            print('ACTIVE INFERENCE ACTIVATED')
            print('!' * 50)
            self.model.rssm.active_inference = True
        losses, output, _, _ = self.shared_step(batch, mode='train')

        self.logging_and_visualisation(batch, output, [], losses, None, batch_idx, prefix='train')

        return self.loss_reducing(losses)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        self.train()
        for module in self.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        with torch.no_grad():
            loss, output, loss_imagines, output_imagines = self.shared_step(batch, mode='val', predict_action=False)
        self.eval()

        batch_rf = {key: value[:, :self.rf] for key, value in batch.items()}  # dim (b, s, 512)
        batch_fh = {key: value[:, self.rf:] for key, value in batch.items()}  # dim (b, s, 512)
        self.add_metrics(self.metrics_vals[dataloader_idx], batch_rf, output)
        for output_imagine in output_imagines:
            self.add_metrics(self.metrics_vals_imagine[dataloader_idx], batch_fh, output_imagine)

        self.logging_and_visualisation(batch, output, output_imagines, loss, loss_imagines,
                                       batch_idx, prefix=f'val{dataloader_idx}')

        return {f'val{dataloader_idx}_loss': self.loss_reducing(loss),
                f'val{dataloader_idx}_loss_imagine':
                    sum([self.loss_reducing(loss_imagine) for loss_imagine in loss_imagines]) / len(loss_imagines)}

    def add_metrics(self, metrics, batch, output):
        if self.cfg.SEMANTIC_SEG.ENABLED:
            seg_prediction = output['bev_segmentation_1'].detach()
            seg_prediction = torch.argmax(seg_prediction, dim=2)
            metrics['iou'](
                seg_prediction.view(-1).cpu(),
                batch['birdview_label'].view(-1).cpu()
            )

        if self.cfg.EVAL.RGB_SUPERVISION:
            metrics['ssim'].add_batch(
                prediction=output[f'rgb_1'].detach(),
                target=batch[f'rgb_label_1'],
            )
            metrics['psnr'].add_batch(
                prediction=output[f'rgb_1'].detach(),
                target=batch[f'rgb_label_1'],
            )

        if self.cfg.LIDAR_RE.ENABLED:
            lidar_target = batch['range_view_label_1']
            lidar_pred = output['lidar_reconstruction_1'].detach()

            pcd_target = lidar_target.detach().permute(0, 1, 3, 4, 2).flatten(2, 3).flatten(0, 1) \
                         * self.cfg.LIDAR_RE.SCALE
            pcd_pred = lidar_pred.detach().permute(0, 1, 3, 4, 2).flatten(2, 3).flatten(0, 1) \
                       * self.cfg.LIDAR_RE.SCALE
            index = np.random.randint(0, pcd_target.size(-2), 10000)
            metrics['cd'].add_batch(pcd_pred[:, index, :-1], pcd_target[:, index, :-1])

            # pcd_target = lidar_target.detach().permute(0, 1, 3, 4, 2).flatten(0, 1) \
            #              * self.cfg.LIDAR_RE.SCALE
            # valid_target = pcd_target[..., -1] > 0
            # pcd_pred = lidar_pred.detach().permute(0, 1, 3, 4, 2).flatten(0, 1) \
            #            * self.cfg.LIDAR_RE.SCALE
            # valid_pred = pcd_pred[..., -1] > 0
            # metrics['cd'].add_batch(pcd_pred[..., :-1], pcd_target[..., :-1], valid_pred, valid_target)

        if self.cfg.LIDAR_SEG.ENABLED:
            pcd_sem_prediction = output['lidar_segmentation_1'].detach()
            pcd_sem_prediction = torch.argmax(pcd_sem_prediction, dim=2)
            metrics['pcd_iou'](
                pcd_sem_prediction.view(-1).cpu(),
                batch['range_view_seg_label_1'].view(-1).cpu()
            )

        if self.cfg.SEMANTIC_IMAGE.ENABLED:
            image_sem_prediction = output['semantic_image_1'].detach()
            image_sem_prediction = torch.argmax(image_sem_prediction, dim=2)
            metrics['image_iou'](
                image_sem_prediction.view(-1).cpu(),
                batch['semantic_image_label_1'].reshape(-1).cpu()
            )

        if self.cfg.VOXEL_SEG.ENABLED:
            self.compute_ssc_metrics(batch, output, metrics['ssc'])

    def compute_ssc_metrics(self, batch, output, metric):
        y_true = batch['voxel_label_1']
        y_pred = output['voxel_1'].detach()
        b, s, c, x, y, z = y_pred.shape
        y_pred = y_pred.reshape(b * s, c, x, y, z)
        y_true = y_true.reshape(b * s, x, y, z)
        y_pred = torch.argmax(y_pred, dim=1)
        metric.add_batch(y_pred, y_true)

    def logging_and_visualisation(self, batch, output, output_imagine, loss, loss_imagines, batch_idx, prefix='train'):
        # Logging
        self.log('-global_step', torch.tensor(-self.global_step, dtype=torch.float32))
        for key, value in loss.items():
            self.log(f'{prefix}_{key}', value)
        if loss_imagines:
            for key, value in loss_imagines[0].items():
                self.log(f'{prefix}_{key}_imagine', value)

        #  Visualisation
        if prefix == 'train':
            visualisation_criteria = (self.global_step % self.cfg.LOG_VIDEO_INTERVAL == 0) \
                                     & (self.global_step != self.vis_step)
            self.vis_step = self.global_step
        else:
            visualisation_criteria = batch_idx == 0
        if visualisation_criteria:
            self.visualise(batch, output, output_imagine, batch_idx, prefix=prefix)

    def loss_reducing(self, loss):
        total_loss = sum([x for x in loss.values()])
        return total_loss

    def on_validation_epoch_end(self):
        self.log_metrics(self.metrics_vals, 'val')
        self.log_metrics(self.metrics_vals_imagine, 'val_imagine')

    def log_metrics(self, metrics_list, metrics_type):
        class_names = ['Background', 'Road', 'Lane marking', 'Vehicle', 'Pedestrian', 'Green light', 'Yellow light',
                       'Red light and stop sign']
        class_names_voxel = list(VOXEL_LABEL.values())
        for idx, metrics in enumerate(metrics_list):
            prefix = f'{metrics_type}{idx}'
            if self.cfg.SEMANTIC_SEG.ENABLED:
                scores = metrics['iou'].compute()
                for key, value in zip(class_names, scores):
                    self.logger.experiment.add_scalar(f'{prefix}_bev_iou_' + key, value, global_step=self.global_step)
                self.logger.experiment.add_scalar(f'{prefix}_bev_mean_iou', torch.mean(scores), global_step=self.global_step)
                metrics['iou'].reset()

            if self.cfg.EVAL.RGB_SUPERVISION:
                self.log(f'{prefix}_ssim', metrics['ssim'].get_stat())
                metrics['ssim'].reset()
                self.log(f'{prefix}_psnr', metrics['psnr'].get_stat())
                metrics['psnr'].reset()

            if self.cfg.LIDAR_RE.ENABLED:
                self.log(f'{prefix}_chamfer_distance', metrics['cd'].get_stat())
                metrics['cd'].reset()

            if self.cfg.LIDAR_SEG.ENABLED:
                scores_pcd = metrics['pcd_iou'].compute()
                for key, value in zip(class_names_voxel, scores_pcd):
                    self.logger.experiment.add_scalar(f'{prefix}_lidar_iou_' + key, value, global_step=self.global_step)
                self.logger.experiment.add_scalar(f'{prefix}_lidar_mean_iou', torch.mean(scores_pcd), global_step=self.global_step)
                metrics['pcd_iou'].reset()

            if self.cfg.SEMANTIC_IMAGE.ENABLED:
                scores_img = metrics['image_iou'].compute()
                for key, value in zip(class_names_voxel, scores_img):
                    self.logger.experiment.add_scalar(f'{prefix}_camera_iou_' + key, value, global_step=self.global_step)
                self.logger.experiment.add_scalar(f'{prefix}_camera_mean_iou', torch.mean(scores_img), global_step=self.global_step)
                metrics['image_iou'].reset()

            if self.cfg.VOXEL_SEG.ENABLED:
                # class_names_voxel = ['Background', 'Road', 'RoadLines', 'Sidewalk', 'Vehicle',
                #                      'Pedestrian', 'TrafficSign', 'TrafficLight', 'Others']

                stats = metrics['ssc'].get_stats()
                for i, class_name in enumerate(class_names_voxel):
                    self.log(f'{prefix}_Voxel_{class_name}_SemIoU', stats['iou_ssc'][i])
                self.log(f'{prefix}_Voxel_mIoU', stats["iou_ssc_mean"])
                self.log(f'{prefix}_Voxel_IoU', stats["iou"])
                self.log(f'{prefix}_Voxel_Precision', stats["precision"])
                self.log(f'{prefix}_Voxel_Recall', stats["recall"])
                metrics['ssc'].reset()

    def visualise(self, batch, output, output_imagines, batch_idx, prefix='train', writer=None):
        writer = writer if writer else self.logger.experiment
        s = list(batch.values())[0].shape[1]    # total sequence length
        rf = list(output.values())[-1].shape[1]  # receptive field

        name = f'{prefix}_outputs'
        if prefix != 'train':
            name = name + f'_{batch_idx}'
        # global_step = batch_idx if prefix == 'pred' else self.global_step
        global_step = self.global_step

        if self.cfg.SEMANTIC_SEG.ENABLED:

            # target = batch['birdview_label'][:, :, 0]
            # pred = torch.argmax(output['bev_segmentation_1'].detach(), dim=-3)

            # colours = torch.tensor(BIRDVIEW_COLOURS, dtype=torch.uint8, device=pred.device)

            # target = colours[target]
            # pred = colours[pred]

            # # Move channel to third position
            # target = target.permute(0, 1, 4, 2, 3)
            # pred = pred.permute(0, 1, 4, 2, 3)

            # visualisation_video = torch.cat([target, pred], dim=-1).detach()

            # # Rotate for visualisation
            # visualisation_video = torch.rot90(visualisation_video, k=1, dims=[3, 4])

            # name = f'{prefix}_outputs'
            # if prefix == 'val':
            #     name = name + f'_{batch_idx}'
            # self.logger.experiment.add_video(name, visualisation_video, global_step=self.global_step, fps=2)

            target = batch['birdview_label'][:, :, 0].cpu()
            pred = torch.argmax(output['bev_segmentation_1'].detach().cpu(), dim=-3)
            bev_imagines = []
            if output_imagines:
                # multi samples of future
                for imagine in output_imagines:
                    bev_imagines.append(torch.argmax(imagine['bev_segmentation_1'].detach().cpu(), dim=-3))
            else:
                bev_imagines.append(None)

            colours = torch.tensor(BIRDVIEW_COLOURS, dtype=torch.uint8, device=pred.device) / 255.0

            target = colours[target]
            # pred = colours[pred]

            # Move channel to third position and add white border
            target = F.pad(target.permute(0, 1, 4, 2, 3), [2, 2, 2, 2], 'constant', 0.8)
            # pred = F.pad(pred.permute(0, 1, 4, 2, 3), [2, 2, 2, 2], 'constant', 0.8)
            preds = []
            # put reconstruction and all imaginations together
            for i, bev_imagine in enumerate(bev_imagines):
                bev_receptive = pred if i == 0 else torch.zeros_like(pred)
                p_i = bev_receptive if bev_imagine is None else torch.cat([bev_receptive, bev_imagine], dim=1)
                p_i = colours[p_i]
                p_i = F.pad(p_i.permute(0, 1, 4, 2, 3), [2, 2, 2, 2], 'constant', 0.8)
                preds.append(p_i)

            bev = torch.cat([*preds[::-1], target], dim=-1).detach()
            # Rotation for Visualization
            bev = torch.rot90(bev, k=1, dims=[3, 4])

            b, _, c, h, w = bev.size()

            visualisation_bev = []
            for step in range(s):
                if step == rf:
                    # separate the receptive filed and future horizon
                    visualisation_bev.append(torch.ones(b, c, h, int(w / 4), device=pred.device))
                visualisation_bev.append(bev[:, step])
            visualisation_bev = torch.cat(visualisation_bev, dim=-1).detach()

            name_ = f'{name}_bev'
            writer.add_images(name_, visualisation_bev, global_step=global_step)

        if self.cfg.EVAL.RGB_SUPERVISION:
            # rgb_target = batch['rgb_label_1']
            # rgb_pred = output['rgb_1'].detach()

            # visualisation_rgb = torch.cat([rgb_pred, rgb_target], dim=-2).detach()
            # name_ = f'{name}_rgb'
            # writer.add_video(name_, visualisation_rgb, global_step=global_step, fps=2)

            rgb_target = batch['rgb_label_1'].cpu()
            rgb_pred = output['rgb_1'].detach().cpu()
            rgb_imagines = []
            if output_imagines:
                for imagine in output_imagines:
                    rgb_imagines.append(imagine['rgb_1'].detach().cpu())
            else:
                rgb_imagines.append(None)

            b, _, c, h, w = rgb_target.size()

            rgb_preds = []
            for i, rgb_imagine in enumerate(rgb_imagines):
                rgb_receptive = rgb_pred if i == 0 else torch.ones_like(rgb_pred)
                pred_imagine = rgb_receptive if rgb_imagine is None else torch.cat([rgb_receptive, rgb_imagine], dim=1)
                rgb_preds.append(F.pad(pred_imagine, [5, 5, 5, 5], 'constant', 0.8))

            rgb_target = F.pad(rgb_target, [5, 5, 5, 5], 'constant', 0.8)
            # rgb_pred = F.pad(rgb_pred, [5, 5, 5, 5], 'constant', 0.8)

            acc = batch['throttle_brake']
            steer = batch['steering']

            acc_bar = np.ones((b, s, int(h / 4), w + 10, c)).astype(np.uint8) * 255
            steer_bar = np.ones((b, s, int(h / 4), w + 10, c)).astype(np.uint8) * 255

            red = np.array([200, 0, 0])[None, None]
            green = np.array([0, 200, 0])[None, None]
            blue = np.array([0, 0, 200])[None, None]
            mid = int(w / 2) + 5

            # visualize accelerating and steering. green for throttle, red for brake, blue for steer.
            for b_idx in range(b):
                for step in range(s):
                    if acc[b_idx, step] >= 0:
                        acc_bar[b_idx, step, 5: -5, mid: mid + int(w / 2 * acc[b_idx, step]), :] = green
                        cv2.putText(acc_bar[b_idx, step], f'{acc[b_idx, step, 0]:.5f}', (mid - 220, int(h / 8) + 15),
                                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
                    else:
                        acc_bar[b_idx, step, 5: -5, mid + int(w / 2 * acc[b_idx, step]): mid, :] = red
                        cv2.putText(acc_bar[b_idx, step], f'{acc[b_idx, step, 0]:.5f}', (mid + 10, int(h / 8) + 15),
                                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
                    if steer[b_idx, step] >= 0:
                        steer_bar[b_idx, step, 5: -5, mid: mid + int(w / 2 * steer[b_idx, step]), :] = blue
                        cv2.putText(steer_bar[b_idx, step], f'{steer[b_idx, step, 0]:.5f}',
                                    (mid - 220, int(h / 8) + 15),
                                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
                    else:
                        steer_bar[b_idx, step, 5: -5, mid + int(w / 2 * steer[b_idx, step]): mid, :] = blue
                        cv2.putText(steer_bar[b_idx, step], f'{steer[b_idx, step, 0]:.5f}', (mid + 10, int(h / 8) + 15),
                                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)
            acc_bar = torch.tensor(acc_bar.transpose((0, 1, 4, 2, 3)),
                                   dtype=torch.float, device=rgb_pred.device) / 255.0
            steer_bar = torch.tensor(steer_bar.transpose((0, 1, 4, 2, 3)),
                                     dtype=torch.float, device=rgb_pred.device) / 255.0

            rgb = torch.cat([acc_bar, steer_bar, rgb_target, *rgb_preds], dim=-2)
            visualisation_rgb = []
            for step in range(s):
                if step == rf:
                    visualisation_rgb.append(torch.ones(b, c, rgb.size(-2), int(w / 4), device=rgb_pred.device))
                visualisation_rgb.append(rgb[:, step, ...])
            visualisation_rgb = torch.cat(visualisation_rgb, dim=-1).detach()

            name_ = f'{name}_rgb'
            writer.add_images(name_, visualisation_rgb, global_step=global_step)

            ###################################
            # visualize optical flow of rgb images.
            flows = []
            rgb_target_np = (rgb_target.detach().cpu().numpy().transpose(0, 1, 3, 4, 2) * 255).astype(np.uint8)
            rgb_preds_np = [(rgb_pred_.detach().cpu().numpy().transpose(0, 1, 3, 4, 2) * 255).astype(np.uint8)
                            for rgb_pred_ in rgb_preds]
            for bs in range(rgb.size(0)):
                flows.append(list())
                for step in range(1, rgb.size(1)):
                    img1_target = rgb_target_np[bs, step - 1][5: -5, 5: -5]
                    img2_target = rgb_target_np[bs, step][5: -5, 5: -5]
                    # use color to present flow
                    flow_target = self.get_color_coded_flow(img1_target, img2_target)
                    flow_target = F.pad(flow_target, [5, 5, 5, 5], 'constant', 0.8)

                    flow_preds = []
                    for i, rgb_pred_np in enumerate(rgb_preds_np):
                        img1_pred = rgb_pred_np[bs, step - 1][5: -5, 5: -5]
                        if i == rf:
                            img1_pred = rgb_preds_np[0][bs, step - 1][5: -5, 5: -5]
                        img2_pred = rgb_pred_np[bs, step][5: -5, 5: -5]
                        flow_pred = self.get_color_coded_flow(img1_pred, img2_pred)
                        flow_pred = F.pad(flow_pred, [5, 5, 5, 5], 'constant', 0.8)
                        flow_preds.append(flow_pred)

                    flows[bs].append(torch.cat([flow_target, *flow_preds], dim=1))

            visualisation_flow = torch.stack([torch.cat(flow, dim=-1) for flow in flows], dim=0)
            name_ = f'{name}_flow'
            writer.add_images(name_, visualisation_flow, global_step=global_step)

        if self.cfg.LIDAR_RE.ENABLED:
            lidar_target = batch['range_view_label_1'].cpu()
            lidar_pred = output['lidar_reconstruction_1'].detach().cpu()
            # lidar_imagine = output_imagine[0]['lidar_reconstruction_1'].detach()
            if output_imagines:
                lidar_imagines = [imagine['lidar_reconstruction_1'].detach().cpu() for imagine in output_imagines]
                lidar_pred_imagine = torch.cat([lidar_pred, lidar_imagines[0]], dim=1)
            else:
                lidar_imagines = [None]
                lidar_pred_imagine = lidar_pred

            visualisation_lidar = torch.cat(
                [lidar_target[:, :, -1, :, :], lidar_pred_imagine[:, :, -1, :, :]],
                dim=-2).detach().unsqueeze(-3)
            name_ = f'{name}_lidar'
            writer.add_video(name_, visualisation_lidar, global_step=global_step, fps=2)

            # get the bird-eye-view of point cloud
            pcd_image_target, pcd_target, valid_target = self.pcd_xy_image(lidar_target)
            pcd_image_target = F.pad(pcd_image_target, [2, 2, 2, 2], 'constant', 0.2)

            pcd_image_pred, pcd_pred, valid_pred = self.pcd_xy_image(lidar_pred)

            pcd_image_preds = []
            pcd_preds = []
            valid_preds = []
            for i, lidar_imagine in enumerate(lidar_imagines):
                pcd_image_receptive = pcd_image_pred if i == 0 else torch.ones_like(pcd_image_pred)
                if lidar_imagine is None:
                    pcd_image_pred_imagine = pcd_image_receptive
                    pcd_pred_imagine = pcd_pred
                    valid_pred_imagine = valid_pred
                else:
                    pcd_image_imagine, pcd_imagine, valid_imagine = self.pcd_xy_image(lidar_imagine)
                    pcd_image_pred_imagine = torch.cat([pcd_image_receptive, pcd_image_imagine], dim=1)
                    pcd_pred_imagine = np.concatenate([pcd_pred, pcd_imagine], axis=1)
                    valid_pred_imagine = np.concatenate([valid_pred, valid_imagine], axis=1)
                pcd_image_preds.append(F.pad(pcd_image_pred_imagine, [2, 2, 2, 2], 'constant', 0.2))
                pcd_preds.append(pcd_pred_imagine)
                valid_preds.append(valid_pred_imagine)

            pcd_image = torch.cat([pcd_image_target, *pcd_image_preds], dim=-2)
            b, _, c, h, w = pcd_image.size()

            visualisation_pcd = []
            for step in range(s):
                if step == rf:
                    visualisation_pcd.append(torch.ones(b, c, h, int(w / 4), device=pcd_image.device))
                visualisation_pcd.append(pcd_image[:, step])
            visualisation_pcd = torch.cat(visualisation_pcd, dim=-1).detach()

            name_ = f'{name}_pcd_xy'
            writer.add_images(name_, visualisation_pcd, global_step=global_step)

            # calculate the ego-vehicle trajectory from point cloud and visualize it
            visualisation_traj = []
            for bs in range(pcd_target.shape[0]):
                path_target = [{'Rot': np.eye(3), 'pos': np.zeros((3, 1))}]
                traj_target = np.pad(np.zeros((192, 192)), pad_width=2, mode='constant', constant_values=50)
                traj_target = np.tile(traj_target[..., None], (1, 1, 3))
                traj_target = self.plot_traj(path_target, traj_target)
                path_preds = []
                traj_preds = []
                for i in range(len(pcd_preds)):
                    path_pred = [{'Rot': np.eye(3), 'pos': np.zeros((3, 1))}]
                    # traj_pred = np.pad(np.zeros((192, 192)), pad_width=2, mode='constant', constant_values=50)
                    # traj_pred = np.tile(traj_pred[..., None], (1, 1, 3))
                    # traj_pred = self.plot_traj(path_pred, traj_pred)
                    path_preds.append(path_pred)
                    traj_preds.append(traj_target.copy())
                for step in range(1, pcd_target.shape[1]):
                    pcd1 = pcd_target[bs, step - 1][valid_target[bs, step - 1]][:, :3]
                    pcd2 = pcd_target[bs, step][valid_target[bs, step]][:, :3]
                    _, Rt_target = compute_pcd_transformation(pcd1, pcd2, path_target[-1], threshold=5)
                    path_target.append(Rt_target)
                    traj_target = self.plot_traj(path_target, traj_target)

                    for j in range(len(pcd_preds)):
                        pcd1 = pcd_preds[j][bs, step - 1][valid_preds[j][bs, step - 1]][:, :3]
                        pcd2 = pcd_preds[j][bs, step][valid_preds[j][bs, step]][:, :3]
                        _, Rt_pred = compute_pcd_transformation(pcd1, pcd2, path_preds[j][-1], threshold=5)
                        path_preds[j].append(Rt_pred)
                        traj_preds[j] = self.plot_traj(path_preds[j], traj_preds[j])

                traj = np.concatenate([traj_target, *traj_preds], axis=1).transpose((2, 0, 1))[None]
                visualisation_traj.append(torch.tensor(traj, device=lidar_target.device, dtype=torch.float))
            visualisation_traj = torch.cat(visualisation_traj, dim=0) / 255.0
            name_ = f'{name}_traj'
            writer.add_images(name_, visualisation_traj, global_step=global_step)

        if self.cfg.LIDAR_SEG.ENABLED:
            lidar_seg_target = batch['range_view_seg_label_1'][:, :, 0].cpu()
            lidar_seg_pred = torch.argmax(output['lidar_segmentation_1'].detach().cpu(), dim=-3)
            lidar_seg_imagines = []
            if output_imagines:
                for imagine in output_imagines:
                    lidar_seg_imagines.append(torch.argmax(imagine['lidar_segmentation_1'].detach().cpu(), dim=-3))
            else:
                lidar_seg_imagines.append(None)

            colours = torch.tensor(VOXEL_COLOURS, dtype=torch.uint8, device=lidar_seg_pred.device) / 255.0
            
            lidar_seg_target = colours[lidar_seg_target]
            lidar_seg_target = F.pad(lidar_seg_target.permute(0, 1, 4, 2, 3), [3, 3, 3, 3], 'constant', 0.8)

            lidar_seg_preds = []

            for i, lidar_seg_imagine in enumerate(lidar_seg_imagines):
                lidar_seg_receptive = lidar_seg_pred if i == 0 else torch.zeros_like(lidar_seg_pred)
                lidar_seg_i = lidar_seg_receptive if lidar_seg_imagine is None else torch.cat([lidar_seg_receptive, lidar_seg_imagine], dim=1)
                lidar_seg_i = colours[lidar_seg_i]
                lidar_seg_i = F.pad(lidar_seg_i.permute(0, 1, 4, 2, 3), [3, 3, 3, 3], 'constant', 0.8)
                lidar_seg_preds.append(lidar_seg_i)

            lidar_seg = torch.cat([lidar_seg_target, torch.ones_like(lidar_seg_target[:, -1:, ...]), *lidar_seg_preds], dim=1).detach()
            visualisation_lidar_seg = lidar_seg.transpose(1, 2).flatten(2, 3)

            name_ = f'{name}_lidar_seg'
            writer.add_images(name_, visualisation_lidar_seg, global_step=global_step)

        if self.cfg.SEMANTIC_IMAGE.ENABLED:
            sem_target = batch['semantic_image_label_1'][:, :, 0].cpu()
            sem_pred = torch.argmax(output['semantic_image_1'].detach().cpu(), dim=-3)
            sem_imagines = []
            if output_imagines:
                for imagine in output_imagines:
                    sem_imagines.append(torch.argmax(imagine['semantic_image_1'].detach().cpu(), dim=-3))
            else:
                sem_imagines.append(None)

            colours = torch.tensor(VOXEL_COLOURS, dtype=torch.uint8, device=sem_pred.device) / 255.0

            sem_target = colours[sem_target]
            sem_target = F.pad(sem_target.permute(0, 1, 4, 2, 3), [5, 5, 5, 5], 'constant', 0.8)

            sem_preds = []

            for i, sem_imagine in enumerate(sem_imagines):
                sem_receptive = sem_pred if i == 0 else torch.zeros_like(sem_pred)
                sem_i = sem_receptive if sem_imagine is None else torch.cat([sem_receptive, sem_imagine], dim=1)
                sem_i = colours[sem_i]
                sem_i = F.pad(sem_i.permute(0, 1, 4, 2, 3), [5, 5, 5, 5], 'constant', 0.8)
                sem_preds.append(sem_i)

            sem_image = torch.cat([sem_target, *sem_preds], dim=-2).detach()

            b, _, c, h, w = sem_image.size()

            visualisation_sem_image = []
            for step in range(s):
                if step == rf:
                    visualisation_sem_image.append(torch.ones(b, c, h, int(w / 4), device=sem_pred.device))
                visualisation_sem_image.append(sem_image[:, step])
            visualisation_sem_image = torch.cat(visualisation_sem_image, dim=-1).detach()

            name_ = f'{name}_sem_image'
            writer.add_images(name_, visualisation_sem_image, global_step=global_step)

        if self.cfg.DEPTH.ENABLED:
            depth_target = batch['depth_label_1'].cpu()
            depth_pred = output['depth_1'].detach().cpu()
            if output_imagines:
                depth_imagine = output_imagines[0]['depth_1'].detach().cpu()
                depth_pred = torch.cat([depth_pred, depth_imagine], dim=1)

            visualisation_depth = torch.cat([depth_pred, depth_target], dim=-2).detach()
            name_ = f'{name}_depth'
            writer.add_video(name_, visualisation_depth, global_step=global_step, fps=2)

        if self.cfg.VOXEL_SEG.ENABLED:
            voxel_target = batch['voxel_label_1'][0, 0, 0].cpu().numpy()
            voxel_pred = torch.argmax(output['voxel_1'].detach(), dim=-4).cpu().numpy()[0, 0]
            colours = np.asarray(VOXEL_COLOURS, dtype=float) / 255.0
            voxel_color_target = colours[voxel_target]
            voxel_color_pred = colours[voxel_pred]
            name_ = f'{name}_voxel'
            self.write_voxel_figure(voxel_target, voxel_color_target, f'{name_}_target', global_step, writer)
            self.write_voxel_figure(voxel_pred, voxel_color_pred, f'{name_}_pred', global_step, writer)
            if output_imagines:
                voxel_imagine_target = batch['voxel_label_1'][0, self.rf, 0].cpu().numpy()
                voxel_imagine_pred = torch.argmax(output_imagines[0]['voxel_1'].detach(), dim=-4).cpu().numpy()[0, 0]
                voxel_color_imagine_target = colours[voxel_imagine_target]
                voxel_color_imagine_pred = colours[voxel_imagine_pred]
                self.write_voxel_figure(
                    voxel_imagine_target, voxel_color_imagine_target, f'{name_}_imagine_target', global_step, writer)
                self.write_voxel_figure(
                    voxel_imagine_pred, voxel_color_imagine_pred, f'{name_}_imagine_pred', global_step, writer)

        # visualize route map
        if self.cfg.MODEL.ROUTE.ENABLED:
            route_map = batch['route_map'].cpu()
            route_map = F.pad(route_map, [2, 2, 2, 2], 'constant', 0.8)

            b, _, c, h, w = route_map.size()

            visualisation_route = []
            for step in range(s):
                if step == rf:
                    visualisation_route.append(torch.ones(b, c, h, int(w / 4), device=route_map.device))
                visualisation_route.append(route_map[:, step])
            visualisation_route = torch.cat(visualisation_route, dim=-1).detach()

            name_ = f'{name}_input_route_map'
            writer.add_images(name_, visualisation_route, global_step=global_step)

    # render 1 frame voxel
    def write_voxel_figure(self, voxel, voxel_color, name, global_step, writer):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.voxels(voxel, facecolors=voxel_color, shade=False)
        ax.view_init(elev=60, azim=165)
        ax.set_axis_off()
        writer.add_figure(name, fig, global_step=global_step)

    # render trajectory
    def plot_traj(self, traj, img):
        x, y, z = traj[-1]['pos']
        plot_x = int(96 - 5 * y)
        plot_y = int(96 - 5 * x)
        x_, y_, _ = traj[-2]['pos'] if len(traj) > 1 else traj[-1]['pos']
        plot_x_ = int(96 - 5 * y_)
        plot_y_ = int(96 - 5 * x_)
        cv2.line(img, (plot_x, plot_y), (plot_x_, plot_y_), (20, 150, 20), 1)
        cv2.circle(img, (plot_x, plot_y), 2, [150, 20, 20], -2, cv2.LINE_AA)
        return img

    def pcd_xy_image(self, lidar):
        image_size = np.array([256, 256])
        lidar_range = 50

        pcd = lidar.cpu().detach().numpy().transpose(0, 1, 3, 4, 2) * self.cfg.LIDAR_RE.SCALE
        # pcd_target = pcd_target[..., :-1].flatten(1, 2)
        # pcd_target = pcd_target[pcd_target[..., -1] > 0][..., :-1]
        # pcd0 = self.pcd.restore_pcd_coor(lidar[:, :, -1].cpu().numpy() * self.cfg.LIDAR_RE.SCALE)
        pcd_xy = -pcd[..., :2]
        pcd_xy *= min(image_size) / (2 * lidar_range)
        pcd_xy += 0.5 * image_size.reshape((1, 1, 1, 1, -1))
        # only the point which range > 0 is valid
        valid = pcd[..., -1] > 0

        b, s, _, _, _ = pcd.shape
        pcd_xy_image = np.zeros((b, s, *image_size, 3))

        # projection point cloud to xy coordinate (bird-eye-view)
        for i in range(b):
            for j in range(s):
                hw = pcd_xy[i, j][valid[i, j]]
                hw = hw[(0 < hw[:, 0]) & (hw[:, 0] < image_size[0]) &
                        (0 < hw[:, 1]) & (hw[:, 1] < image_size[1])]
                hw = np.fabs(hw)
                hw = hw.astype(np.int32)
                pcd_xy_image[i, j][tuple(hw.T)] = (1.0, 1.0, 1.0)

        return torch.tensor(pcd_xy_image.transpose((0, 1, 4, 2, 3)), device=lidar.device), pcd, valid

    def get_color_coded_flow(self, img1, img2):
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
        hsv[..., 2] = 255
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * (180 / np.pi / 2)
        hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        color_coded_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return torch.tensor(color_coded_flow.transpose(2, 0, 1), dtype=torch.float) / 255.0

    def configure_optimizers(self):
        # frozen the layer that not in train list
        def frozen_params(model, no_frozen_list=[]):
            for name, param in model.named_parameters():
                if not any(name.startswith(layer) for layer in no_frozen_list):
                    param.requires_grad = False

        #  Do not decay batch norm parameters and biases
        # https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/2
        def add_weight_decay(model, weight_decay=0.01, skip_list=[]):
            no_decay = []
            decay = []
            train_list = []
            frozen_list = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    frozen_list.append(name)
                    continue
                train_list.append(name)
                if len(param.shape) == 1 or any(x in name for x in skip_list):
                    no_decay.append(param)
                else:
                    decay.append(param)
            print(f'train_layers: {train_list}\nfrozen_layers: {frozen_list}')
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay},
            ]

        if self.cfg.OPTIMIZER.FROZEN.ENABLED:
            frozen_params(self.model, self.cfg.OPTIMIZER.FROZEN.TRAIN_LIST)

        parameters = add_weight_decay(
            self.model,
            self.cfg.OPTIMIZER.WEIGHT_DECAY,
            skip_list=['relative_position_bias_table'],
        )
        weight_decay = 0.
        optimizer = torch.optim.AdamW(parameters, lr=self.cfg.OPTIMIZER.LR, weight_decay=weight_decay)

        # scheduler
        if self.cfg.SCHEDULER.NAME == 'none':
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda lr: 1)
        elif self.cfg.SCHEDULER.NAME == 'OneCycleLR':
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.cfg.OPTIMIZER.LR,
                total_steps=self.cfg.STEPS,
                pct_start=self.cfg.SCHEDULER.PCT_START,
            )

        return [optimizer], [{'scheduler': lr_scheduler, 'interval': 'step'}]

    def on_test_epoch_end(self):
        self.log_metrics(self.metrics_tests, 'test')
        self.log_metrics(self.metrics_tests_imagine, 'test_imagine')

    def test_step(self, batch, batch_idx, dataloader_idx):
        self.train()
        for module in self.modules():
            if isinstance(module, torch.nn.Dropout):
                module.eval()
        with torch.no_grad():
            loss, output, loss_imagines, output_imagines = self.shared_step(batch, mode='test', predict_action=False)
        self.eval()

        batch_rf = {key: value[:, :self.rf] for key, value in batch.items()}  # dim (b, s, 512)
        batch_fh = {key: value[:, self.rf:] for key, value in batch.items()}  # dim (b, s, 512)
        self.add_metrics(self.metrics_tests[dataloader_idx], batch_rf, output)
        for output_imagine in output_imagines:
            self.add_metrics(self.metrics_tests_imagine[dataloader_idx], batch_fh, output_imagine)

        self.visualise(batch, output, output_imagines, batch_idx, prefix=f'pred{dataloader_idx}')
        return output, output_imagines
