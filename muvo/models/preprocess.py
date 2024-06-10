import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as tvf
# import skimage.transform as skt
from typing import Dict, Tuple

from muvo.utils.geometry_utils import get_out_of_view_mask
from muvo.utils.instance_utils import convert_instance_mask_to_center_and_offset_label


class PreProcess(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.crop = tuple(cfg.IMAGE.CROP)
        self.route_map_size = cfg.ROUTE.SIZE

        if self.cfg.EVAL.MASK_VIEW:
            self.bev_out_of_view_mask = get_out_of_view_mask(cfg)

        # Instance label parameters
        self.center_sigma = cfg.INSTANCE_SEG.CENTER_LABEL_SIGMA_PX
        self.ignore_index = cfg.INSTANCE_SEG.IGNORE_INDEX

        self.min_depth = cfg.BEV.FRUSTUM_POOL.D_BOUND[0]
        self.max_depth = cfg.BEV.FRUSTUM_POOL.D_BOUND[1]

        self.pixel_augmentation = PixelAugmentation(cfg)
        self.route_augmentation = RouteAugmentation(
                cfg.ROUTE.AUGMENTATION_DROPOUT,
                cfg.ROUTE.AUGMENTATION_END_OF_ROUTE,
                cfg.ROUTE.AUGMENTATION_SMALL_ROTATION,
                cfg.ROUTE.AUGMENTATION_LARGE_ROTATION,
                cfg.ROUTE.AUGMENTATION_DEGREES,
                cfg.ROUTE.AUGMENTATION_TRANSLATE,
                cfg.ROUTE.AUGMENTATION_SCALE,
                cfg.ROUTE.AUGMENTATION_SHEAR,
            )

        self.register_buffer('image_mean', torch.tensor(cfg.IMAGE.IMAGENET_MEAN).unsqueeze(1).unsqueeze(1))
        self.register_buffer('image_std', torch.tensor(cfg.IMAGE.IMAGENET_STD).unsqueeze(1).unsqueeze(1))

    def augmentation(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        batch = self.pixel_augmentation(batch)
        batch = self.route_augmentation(batch)
        return batch

    def prepare_bev_labels(self, batch):
        if 'birdview_label' in batch:
            # Mask bird's-eye view label pixels that are not visible from the input image
            if self.cfg.EVAL.MASK_VIEW:
                batch['birdview_label'][:, :, :, self.bev_out_of_view_mask] = 0

            # Currently the frustum pooling is set up such that the bev features are rotated by 90 degrees clockwise
            batch['birdview_label'] = torch.rot90(batch['birdview_label'], k=-1, dims=[3, 4]).contiguous()

            # Compute labels at half, quarter, and 1/8th resolution
            batch['birdview_label_1'] = batch['birdview_label']
            h, w = batch['birdview_label'].shape[-2:]
            for downsample_factor in [2, 4]:
                size = h // downsample_factor, w // downsample_factor
                previous_label_factor = downsample_factor // 2
                batch[f'birdview_label_{downsample_factor}'] = functional_resize(
                    batch[f'birdview_label_{previous_label_factor}'], size, mode=tvf.InterpolationMode.NEAREST
                )

        if 'instance_label' in batch:
            # Mask elements not visible from the input image
            if self.cfg.EVAL.MASK_VIEW:
                batch['instance_label'][:, :, :, self.bev_out_of_view_mask] = 0
            #  Currently the frustum pooling is set up such that the bev features are rotated by 90 degrees clockwise
            batch['instance_label'] = torch.rot90(batch['instance_label'], k=-1, dims=[3, 4]).contiguous()

            center_label, offset_label = convert_instance_mask_to_center_and_offset_label(
                batch['instance_label'], ignore_index=self.ignore_index, sigma=self.center_sigma,
            )
            batch['center_label'] = center_label
            batch['offset_label'] = offset_label

            # Compute labels at half, quarter, and 1/8th resolution
            batch['instance_label_1'] = batch['instance_label']
            batch['center_label_1'] = batch['center_label']
            batch['offset_label_1'] = batch['offset_label']

            h, w = batch['instance_label'].shape[-2:]
            for downsample_factor in [2, 4]:
                size = h // downsample_factor, w // downsample_factor
                previous_label_factor = downsample_factor // 2
                batch[f'instance_label_{downsample_factor}'] = functional_resize(
                    batch[f'instance_label_{previous_label_factor}'], size, mode=tvf.InterpolationMode.NEAREST
                )

                center_label, offset_label = convert_instance_mask_to_center_and_offset_label(
                    batch[f'instance_label_{downsample_factor}'], ignore_index=self.ignore_index,
                    sigma=self.center_sigma/downsample_factor,
                )
                batch[f'center_label_{downsample_factor}'] = center_label
                batch[f'offset_label_{downsample_factor}'] = offset_label

        if self.cfg.EVAL.RGB_SUPERVISION:
            # Compute labels at half, quarter, and 1/8th resolution
            batch['rgb_label_1'] = batch['image']
            h, w = batch['rgb_label_1'].shape[-2:]
            for downsample_factor in [2, 4]:
                size = h // downsample_factor, w // downsample_factor
                previous_label_factor = downsample_factor // 2
                batch[f'rgb_label_{downsample_factor}'] = functional_resize(
                    batch[f'rgb_label_{previous_label_factor}'],
                    size,
                    mode=tvf.InterpolationMode.BILINEAR,
                )

            if self.cfg.LOSSES.RGB_INSTANCE:
                batch['image_instance_mask_1'] = batch['image_instance_mask']
                h, w = batch['image_instance_mask_1'].shape[-2:]
                for downsample_factor in [2, 4]:
                    size = h // downsample_factor, w // downsample_factor
                    previous_label_factor = downsample_factor // 2
                    batch[f'image_instance_mask_{downsample_factor}'] = functional_resize(
                        batch[f'image_instance_mask_{previous_label_factor}'],
                        size,
                        mode=tvf.InterpolationMode.NEAREST,
                    )

        if self.cfg.SEMANTIC_IMAGE.ENABLED:
            batch['semantic_image_label_1'] = batch['semantic_image']
            h, w = batch['semantic_image_label_1'].shape[-2:]
            for downsample_factor in [2, 4]:
                size = h // downsample_factor, w // downsample_factor
                previous_label_factor = downsample_factor // 2
                batch[f'semantic_image_label_{downsample_factor}'] = functional_resize(
                    batch[f'semantic_image_label_{previous_label_factor}'],
                    size,
                    mode=tvf.InterpolationMode.NEAREST,
                )

        if self.cfg.DEPTH.ENABLED:
            batch['depth_label_1'] = batch['depth']
            h, w = batch['depth_label_1'].shape[-2:]
            for downsample_factor in [2, 4]:
                size = h // downsample_factor, w // downsample_factor
                previous_label_factor = downsample_factor // 2
                batch[f'depth_label_{downsample_factor}'] = functional_resize(
                    batch[f'depth_label_{previous_label_factor}'],
                    size,
                    mode=tvf.InterpolationMode.BILINEAR,
                )

        if self.cfg.LIDAR_RE.ENABLED:
            batch['range_view_pcd_xyzd'] = batch['range_view_pcd_xyzd'].float() / self.cfg.LIDAR_RE.SCALE
            batch['range_view_label_1'] = batch['range_view_pcd_xyzd']
            h, w = batch['range_view_label_1'].shape[-2:]
            for downsample_factor in [2, 4]:
                size = h // downsample_factor, w // downsample_factor
                previous_label_factor = downsample_factor // 2
                batch[f'range_view_label_{downsample_factor}'] = functional_resize(
                    batch[f'range_view_label_{previous_label_factor}'],
                    size,
                    mode=tvf.InterpolationMode.NEAREST,
                )

        if self.cfg.LIDAR_SEG.ENABLED:
            batch['range_view_seg_label_1'] = batch['range_view_pcd_seg']
            h, w = batch['range_view_seg_label_1'].shape[-2:]
            for downsample_factor in [2, 4]:
                size = h // downsample_factor, w // downsample_factor
                previous_label_factor = downsample_factor // 2
                batch[f'range_view_seg_label_{downsample_factor}'] = functional_resize(
                    batch[f'range_view_seg_label_{previous_label_factor}'],
                    size,
                    mode=tvf.InterpolationMode.NEAREST,
                )

        if self.cfg.VOXEL_SEG.ENABLED:
            batch['voxel_label_1'] = batch['voxel']
            x, y, z = batch['voxel_label_1'].shape[-3:]
            for downsample_factor in [2, 4]:
                size = (x // downsample_factor, y // downsample_factor, z // downsample_factor)
                previous_label_factor = downsample_factor // 2
                batch[f'voxel_label_{downsample_factor}'] = functional_resize_voxel(
                    batch[f'voxel_label_{previous_label_factor}'],
                    size,
                    mode='nearest',
                )

        # if 'points_histogram' in batch:
        #     # mask histogram the same as bev.
        #     if self.cfg.EVAL.MASK_VIEW:
        #         scale = self.cfg.POINTS.HISTOGRAM.RESOLUTION * self.cfg.BEV.RESOLUTION
        #         bev_shape = self.bev_out_of_view_mask.shape
        #         out_shape = [int(scale * bev_shape[0]), int(scale * bev_shape[1])]
        #         view_mask = skt.resize(self.bev_out_of_view_mask, out_shape)
        #         batch['points_histogram'] = tvf.center_crop(batch['points_histogram'], out_shape)
        #         batch['points_histogram'][:, :, :, view_mask[::-1, ::-1]] = 0
            # batch['points_histogram'] = torch.rot90(batch['points_histogram'], k=-1, dims=[3, 4]).contiguous()

        return batch

    def forward(self, batch: Dict[str, torch.Tensor]):
        # Normalise from [0, 255] to [0, 1]
        batch['image'] = batch['image'].float() / 255

        if 'route_map' in batch:
            batch['route_map'] = batch['route_map'].float() / 255
            batch['route_map'] = functional_resize(batch['route_map'], size=(self.route_map_size, self.route_map_size))
        batch = functional_crop(batch, self.crop)
        if self.cfg.EVAL.RESOLUTION.ENABLED:
            batch = functional_resize_batch(batch, scale=1/self.cfg.EVAL.RESOLUTION.FACTOR)

        batch = self.prepare_bev_labels(batch)

        if self.training:
            batch = self.augmentation(batch)

        # Use imagenet mean and std normalisation, because we're loading pretrained backbones
        batch['image'] = (batch['image'] - self.image_mean) / self.image_std
        if 'route_map' in batch:
            batch['route_map'] = (batch['route_map'] - self.image_mean) / self.image_std

        if 'depth' in batch:
            batch['depth_mask'] = (batch['depth'] > self.min_depth) & (batch['depth'] < self.max_depth)

        return batch


def functional_crop(batch: Dict[str, torch.Tensor], crop: Tuple[int, int, int, int]):
    left, top, right, bottom = crop
    height = bottom - top
    width = right - left
    if 'image' in batch:
        batch['image'] = tvf.crop(batch['image'], top, left, height, width)
    if 'depth' in batch:
        batch['depth'] = tvf.crop(batch['depth'], top, left, height, width)
    if 'depth_color' in batch:
        batch['depth_color'] = tvf.crop(batch['depth'], top, left, height, width)
    if 'semseg' in batch:
        batch['semseg'] = tvf.crop(batch['semseg'], top, left, height, width)
    if 'semantic_image' in batch:
        batch['semantic_image'] = tvf.crop(batch['semantic_image'], top, left, height, width)
    if 'image_instance_mask' in batch:
        batch['image_instance_mask'] = tvf.crop(batch['image_instance_mask'], top, left, height, width)
    if 'intrinsics' in batch:
        intrinsics = batch['intrinsics'].clone()
        intrinsics[..., 0, 2] -= left
        intrinsics[..., 1, 2] -= top
        batch['intrinsics'] = intrinsics

    return batch


def functional_resize_batch(batch, scale):
    b, s, c, h, w = batch['image'].shape
    h1, w1 = int(round(h * scale)), int(round(w * scale))
    size = (h1, w1)
    if 'image' in batch:
        image = batch['image'].view(b*s, c, h, w)
        image = tvf.resize(image, size, antialias=True)
        batch['image'] = image.view(b, s, c, h1, w1)
    if 'intrinsics' in batch:
        intrinsics = batch['intrinsics'].clone()
        intrinsics[..., :2, :] *= scale
        batch['intrinsics'] = intrinsics
    if 'image_instance_mask' in batch:
        image = batch['image_instance_mask'].view(b*s, c, h, w)
        image = tvf.resize(image, size, antialias=True)
        batch['image_instance_mask'] = image.view(b, s, c, h1, w1)
    if 'semantic_image' in batch:
        image = batch['semantic_image'].view(b*s, c, h, w)
        image = tvf.resize(image, size, antialias=True)
        batch['semantic_image'] = image.view(b, s, c, h1, w1)

    return batch


def functional_resize(x, size, mode=tvf.InterpolationMode.NEAREST):
    b, s, c, h, w = x.shape
    x = x.view(b * s, c, h, w)
    x = tvf.resize(x, size, interpolation=mode)
    x = x.view(b, s, c, *size)

    return x


def functional_resize_voxel(voxel, size, mode='nearst'):
    b, s, c, x, y, z = voxel.shape
    voxel = voxel.view(b * s, c, x, y, z)
    voxel = F.interpolate(voxel, size, mode=mode)
    voxel = voxel.view(b, s, c, *size)

    return voxel


class PixelAugmentation(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # TODO replace with ImageApply([RandomBlurSharpen(), RandomColorJitter(), ...])
        self.blur_prob = cfg.IMAGE.AUGMENTATION.BLUR_PROB
        self.sharpen_prob = cfg.IMAGE.AUGMENTATION.SHARPEN_PROB
        self.blur_window = cfg.IMAGE.AUGMENTATION.BLUR_WINDOW
        self.blur_std = cfg.IMAGE.AUGMENTATION.BLUR_STD
        self.sharpen_factor = cfg.IMAGE.AUGMENTATION.SHARPEN_FACTOR
        assert self.blur_prob + self.sharpen_prob <= 1

        self.color_jitter = transforms.RandomApply(nn.ModuleList([
            transforms.ColorJitter(
                cfg.IMAGE.AUGMENTATION.COLOR_JITTER_BRIGHTNESS,
                cfg.IMAGE.AUGMENTATION.COLOR_JITTER_CONTRAST,
                cfg.IMAGE.AUGMENTATION.COLOR_JITTER_SATURATION,
                cfg.IMAGE.AUGMENTATION.COLOR_JITTER_HUE
            )
        ]), cfg.IMAGE.AUGMENTATION.COLOR_PROB)

    def forward(self, batch: Dict[str, torch.Tensor]):
        image = batch['image']
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # random blur
                rand_value = torch.rand(1)
                if rand_value < self.blur_prob:
                    std = torch.empty(1).uniform_(self.blur_std[0], self.blur_std[1]).item()
                    image[i, j] = tvf.gaussian_blur(image[i, j], self.blur_window, std)
                # random sharpen
                elif rand_value < self.blur_prob + self.sharpen_prob:
                    factor = torch.empty(1).uniform_(self.sharpen_factor[0], self.sharpen_factor[1]).item()
                    image[i, j] = tvf.adjust_sharpness(image[i, j], factor)

                # random color jitter
                image[i, j] = self.color_jitter(image[i, j])

        batch['image'] = image
        return batch


class RouteAugmentation(nn.Module):
    def __init__(self, drop=0.025, end_of_route=0.025, small_rotation=0.025, large_rotation=0.025, degrees=8.0,
                 translate=(.1, .1), scale=(.95, 1.05), shear=(.1, .1)):
        super().__init__()
        assert drop + end_of_route + small_rotation + large_rotation <= 1
        self.drop = drop  # random dropout of map
        self.end_of_route = end_of_route  # probability of end of route augmentation
        self.small_rotation = small_rotation  # probability of doing small rotation
        self.large_rotation = large_rotation  # probability of doing large rotation (arbitrary orientation)
        self.small_perturbation = transforms.RandomAffine(degrees, translate, scale, shear)  # small rotation
        self.large_perturbation = transforms.RandomAffine(180, translate, scale, shear)  # arbitrary orientation

    def forward(self, batch):
        if 'route_map' in batch:
            route_map = batch['route_map']

            # TODO: make augmentation independent of the sequence dimension?
            for i in range(route_map.shape[0]):
                rand_value = torch.rand(1)
                if rand_value < self.drop:
                    route_map[i] = torch.zeros_like(route_map[i])
                elif rand_value < self.drop + self.end_of_route:
                    height = torch.randint(route_map[i].shape[-2], (1,))
                    route_map[i][:, :, :height] = 0
                elif rand_value < self.drop + self.end_of_route + self.small_rotation:
                    route_map[i] = self.small_perturbation(route_map[i])
                elif rand_value < self.drop + self.end_of_route + self.small_rotation + self.large_rotation:
                    route_map[i] = self.large_perturbation(route_map[i])

            batch['route_map'] = route_map

        return batch
