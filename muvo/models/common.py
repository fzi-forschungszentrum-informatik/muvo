from typing import List

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_scatter import scatter_mean, scatter_max
import math


class RouteEncode(nn.Module):
    def __init__(self, out_channels, backbone='resnet18'):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, features_only=True, out_indices=[4])
        self.out_channels = out_channels
        feature_info = self.backbone.feature_info.get_dicts(keys=['num_chs', 'reduction'])
        self.fc = nn.Linear(feature_info[-1]['num_chs'], out_channels)

    def forward(self, route):
        x = self.backbone(route)[0]
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return self.fc(x)


class GRUCellLayerNorm(nn.Module):
    def __init__(self, input_size, hidden_size, reset_bias=1.0):
        super().__init__()
        self.reset_bias = reset_bias

        self.update_layer = nn.Linear(input_size + hidden_size, hidden_size, bias=False)
        self.update_norm = nn.LayerNorm(hidden_size)

        self.reset_layer = nn.Linear(input_size + hidden_size, hidden_size, bias=False)
        self.reset_norm = nn.LayerNorm(hidden_size)

        self.proposal_layer = nn.Linear(input_size + hidden_size, hidden_size, bias=False)
        self.proposal_norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs, state):
        update = self.update_layer(torch.cat([inputs, state], -1))
        update = torch.sigmoid(self.update_norm(update))

        reset = self.reset_layer(torch.cat([inputs, state], -1))
        reset = torch.sigmoid(self.reset_norm(reset) + self.reset_bias)

        h_n = self.proposal_layer(torch.cat([inputs, reset * state], -1))
        h_n = torch.tanh(self.proposal_norm(h_n))
        output = (1 - update) * h_n + update * state
        return output


class Policy(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            nn.Linear(in_channels, in_channels),
            nn.ReLU(True),
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(True),
            nn.Linear(in_channels // 2, 2),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, feature_info, out_channels):
        super().__init__()
        n_upsample_skip_convs = len(feature_info) - 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(feature_info[-1]['num_chs'], out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.upsample_skip_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(feature_info[-i]['num_chs'], out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
            for i in range(2, n_upsample_skip_convs + 2)
        )

        self.out_channels = out_channels

    def forward(self, xs: List[Tensor]) -> Tensor:
        x = self.conv1(xs[-1])

        for i, conv in enumerate(self.upsample_skip_convs):
            size = xs[-(i + 2)].shape[-2:]
            x = conv(xs[-(i + 2)]) + F.interpolate(x, size=size, mode='bilinear', align_corners=False)

        return x


class DecoderDS(nn.Module):
    def __init__(self, feature_info, out_channels):
        super().__init__()
        n_downsample_skip_convs = len(feature_info) - 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(feature_info[0]['num_chs'], out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.downsample_skip_convs = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(feature_info[i]['num_chs'], out_channels, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
            for i in range(1, n_downsample_skip_convs + 1)
        )

        self.out_channels = out_channels

    def forward(self, xs: List[Tensor]) -> Tensor:
        x = self.conv1(xs[0])

        for i, conv in enumerate(self.downsample_skip_convs):
            stride = xs[i].shape[-1] // xs[i + 1].shape[-1]
            x = conv(xs[i + 1]) + F.max_pool2d(x, stride)   # avg_pool?

        return x


class DownSampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, latent_n_channels, down_sample_scale=None):
        super().__init__()
        self.down_sample_scale = down_sample_scale
        self.conv1 = ConvInstanceNorm(in_channels, out_channels, latent_n_channels)
        self.conv2 = ConvInstanceNorm(out_channels, out_channels, latent_n_channels)

    def forward(self, x, w):
        if self.down_sample_scale:
            x = F.avg_pool2d(x, self.down_sample_scale)
        x = self.conv1(x, w)
        return self.conv2(x, w)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, latent_n_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.conv1 = ConvInstanceNorm(in_channels, out_channels, latent_n_channels)
        self.conv2 = ConvInstanceNorm(out_channels, out_channels, latent_n_channels)

    def forward(self, x, w):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode='bilinear', align_corners=False)
        x = self.conv1(x, w)
        return self.conv2(x, w)


class DecoderBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, latent_n_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.conv1 = ConvInstanceNorm3d(in_channels, out_channels, latent_n_channels)
        self.conv2 = ConvInstanceNorm3d(out_channels, out_channels, latent_n_channels)

    def forward(self, x, w):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2.0, mode='trilinear', align_corners=False)
        x = self.conv1(x, w)
        return self.conv2(x, w)


class ConvInstanceNorm(nn.Module):
    def __init__(self, in_channels, out_channels, latent_n_channels):
        super().__init__()
        self.conv_act = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.adaptive_norm = AdaptiveInstanceNorm(latent_n_channels, out_channels)

    def forward(self, x, w):
        x = self.conv_act(x)
        return self.adaptive_norm(x, w)


class ConvInstanceNorm3d(nn.Module):
    def __init__(self, in_channels, out_channels, latent_n_channels):
        super().__init__()
        self.conv_act = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.adaptive_norm = AdaptiveInstanceNorm3d(latent_n_channels, out_channels)

    def forward(self, x, w):
        x = self.conv_act(x)
        return self.adaptive_norm(x, w)


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, latent_n_channels, out_channels, epsilon=1e-8):
        super().__init__()
        self.out_channels = out_channels
        self.epsilon = epsilon

        self.latent_affine = nn.Linear(latent_n_channels, 2 * out_channels)

    def forward(self, x, style):
        #  Instance norm
        mean = x.mean(dim=(-1, -2), keepdim=True)
        x = x - mean
        std = torch.sqrt(torch.mean(x ** 2, dim=(-1, -2), keepdim=True) + self.epsilon)
        x = x / std

        # Normalising with the style vector
        style = self.latent_affine(style).unsqueeze(-1).unsqueeze(-1)
        scale, bias = torch.split(style, split_size_or_sections=self.out_channels, dim=1)
        out = scale * x + bias
        return out


class AdaptiveInstanceNorm3d(nn.Module):
    def __init__(self, latent_n_channels, out_channels, epsilon=1e-8):
        super().__init__()
        self.out_channels = out_channels
        self.epsilon = epsilon

        self.latent_affine = nn.Linear(latent_n_channels, 2 * out_channels)

    def forward(self, x, style):
        #  Instance norm
        mean = x.mean(dim=(-1, -2, -3), keepdim=True)
        x = x - mean
        std = torch.sqrt(torch.mean(x ** 2, dim=(-1, -2, -3), keepdim=True) + self.epsilon)
        x = x / std

        # Normalising with the style vector
        style = self.latent_affine(style).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        scale, bias = torch.split(style, split_size_or_sections=self.out_channels, dim=1)
        out = scale * x + bias
        return out


class SegmentationHead(nn.Module):
    def __init__(self, in_channels, n_classes, downsample_factor):
        super().__init__()
        self.downsample_factor = downsample_factor

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(in_channels, n_classes, kernel_size=1, padding=0),
        )
        self.instance_offset_head = nn.Sequential(
            nn.Conv2d(in_channels, 2, kernel_size=1, padding=0),
        )
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = {
            f'bev_segmentation_{self.downsample_factor}': self.segmentation_head(x),
            f'bev_instance_offset_{self.downsample_factor}': self.instance_offset_head(x),
            f'bev_instance_center_{self.downsample_factor}': self.instance_center_head(x),
        }
        return output


class RGBHead(nn.Module):
    def __init__(self, in_channels, n_classes, downsample_factor):
        super().__init__()
        self.downsample_factor = downsample_factor

        self.rgb_head = nn.Sequential(
            nn.Conv2d(in_channels, n_classes, kernel_size=1, padding=0),
        )

    def forward(self, x):
        output = {
            f'rgb_{self.downsample_factor}': self.rgb_head(x),
        }
        return output


class LidarReHead(nn.Module):
    def __init__(self, in_channels, n_classes, downsample_factor):
        super().__init__()
        self.downsample_factor = downsample_factor

        self.lidar_re_head = nn.Sequential(
            nn.Conv2d(in_channels, n_classes, kernel_size=1, padding=0),
        )

    def forward(self, x):
        output = {
            f'lidar_reconstruction_{self.downsample_factor}': self.lidar_re_head(x),
        }
        return output


class LidarSegHead(nn.Module):
    def __init__(self, in_channels, n_classes, downsample_factor):
        super().__init__()
        self.downsample_factor = downsample_factor

        self.seg_head = nn.Sequential(
            nn.Conv2d(in_channels, n_classes, kernel_size=1, padding=0),
        )

    def forward(self, x):
        output = {
            f'lidar_segmentation_{self.downsample_factor}': self.seg_head(x),
        }
        return output


class SemHead(nn.Module):
    def __init__(self, in_channels, n_classes, downsample_factor):
        super().__init__()
        self.downsample_factor = downsample_factor

        self.sem_head = nn.Sequential(
            nn.Conv2d(in_channels, n_classes, kernel_size=1, padding=0),
        )

    def forward(self, x):
        output = {
            f'semantic_image_{self.downsample_factor}': self.sem_head(x),
        }
        return output


class DepthHead(nn.Module):
    def __init__(self, in_channels, n_classes, downsample_factor):
        super().__init__()
        self.downsample_factor = downsample_factor

        self.depth_head = nn.Sequential(
            nn.Conv2d(in_channels, n_classes, kernel_size=1, padding=0),
        )

    def forward(self, x):
        output = {
            f'depth_{self.downsample_factor}': self.depth_head(x),
        }
        return output


class VoxelSemHead(nn.Module):
    def __init__(self, in_channels, n_classes, downsample_factor):
        super().__init__()
        self.downsample_factor = downsample_factor

        self.segmentation_head = nn.Sequential(
            nn.Conv3d(in_channels, n_classes, kernel_size=1, padding=0),
        )

    def forward(self, x):
        output = {
            f'voxel_{self.downsample_factor}': self.segmentation_head(x),
        }
        return output


class BevDecoder(nn.Module):
    def __init__(self, latent_n_channels, semantic_n_channels, constant_size=(3, 3), head='bev'):
        super().__init__()
        n_channels = 512

        self.constant_tensor = nn.Parameter(torch.randn((n_channels, *constant_size), dtype=torch.float32))

        # Input 512 x 3 x 3
        self.first_norm = AdaptiveInstanceNorm(latent_n_channels, out_channels=n_channels)
        self.first_conv = ConvInstanceNorm(n_channels, n_channels, latent_n_channels)
        # 512 x 3 x 3

        self.middle_conv = nn.ModuleList(
            [DecoderBlock(n_channels, n_channels, latent_n_channels, upsample=True) for _ in range(3)]
        )

        head_modules = {'rgb': RGBHead,
                        'bev': SegmentationHead,
                        'depth': DepthHead,
                        'sem_image': SemHead,
                        'lidar_re': LidarReHead,
                        'lidar_seg': LidarSegHead}
        head_module = head_modules[head] if head in head_modules else RGBHead
        # 512 x 24 x 24
        self.conv1 = DecoderBlock(n_channels, 256, latent_n_channels, upsample=True)
        self.head_4 = head_module(256, semantic_n_channels, downsample_factor=4)
        # 256 x 48 x 48

        self.conv2 = DecoderBlock(256, 128, latent_n_channels, upsample=True)
        self.head_2 = head_module(128, semantic_n_channels, downsample_factor=2)
        # 128 x 96 x 96

        self.conv3 = DecoderBlock(128, 64, latent_n_channels, upsample=True)
        self.head_1 = head_module(64, semantic_n_channels, downsample_factor=1)
        # 64 x 192 x 192

    def forward(self, w: Tensor) -> Tensor:
        b = w.shape[0]
        x = self.constant_tensor.unsqueeze(0).repeat([b, 1, 1, 1])

        x = self.first_norm(x, w)
        x = self.first_conv(x, w)

        for module in self.middle_conv:
            x = module(x, w)

        x = self.conv1(x, w)
        output_4 = self.head_4(x)
        x = self.conv2(x, w)
        output_2 = self.head_2(x)
        x = self.conv3(x, w)
        output_1 = self.head_1(x)

        output = {**output_4, **output_2, **output_1}
        return output


class VoxelDecoderScale(nn.Module):
    def __init__(self, input_channels, n_classes, kernel_size=1, feature_channels=512):
        super().__init__()

        # weight of xy,xz,yz view features.
        self.weight_xy_decoder = nn.Conv2d(input_channels, 1, kernel_size, 1)
        self.weight_xz_decoder = nn.Conv2d(input_channels, 1, kernel_size, 1)
        self.weight_yz_decoder = nn.Conv2d(input_channels, 1, kernel_size, 1)

        # self.classifier = nn.Sequential(
        #     nn.Linear(feature_channels, feature_channels),
        #     nn.Softplus(),
        #     nn.Linear(feature_channels, n_classes)
        # )
        self.classifier = nn.Sequential(
            nn.Conv3d(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1),
            nn.Softplus(),
            nn.Conv3d(feature_channels, n_classes, kernel_size=1, stride=1, padding=0)
        )

    def attention_fusion(self, t1, w1, t2, w2):
        norm_weight = torch.softmax(torch.cat([w1, w2], dim=1), dim=1)
        feat = t1 * norm_weight[:, 0:1] + t2 * norm_weight[:, 1:2]
        return feat

    def expand_to_XYZ(self, xy_feat, xz_feat, yz_feat):
        B, C, X, Y, Z = *xy_feat.size(), xz_feat.size(3)
        xy_feat = xy_feat.view(B, C, X, Y, 1)
        xz_feat = xz_feat.view(B, C, X, 1, Z)
        yz_feat = yz_feat.view(B, C, 1, Y, Z)
        return torch.broadcast_tensors(xy_feat, xz_feat, yz_feat)

    def forward(self, x):
        feature_xy, feature_xz, feature_yz = x

        weights_xy = self.weight_xy_decoder(feature_xy)
        weights_xz = self.weight_xz_decoder(feature_xz)
        weights_yz = self.weight_yz_decoder(feature_yz)

        feature_xy, feature_xz, feature_yz = self.expand_to_XYZ(feature_xy, feature_xz, feature_yz)
        weights_xy, weights_xz, weights_yz = self.expand_to_XYZ(weights_xy, weights_xz, weights_yz)

        # fuse xy, xz, yz features in xyz.
        features_xyz = self.attention_fusion(feature_xy, weights_xy, feature_xz, weights_xz) + \
                       self.attention_fusion(feature_xy, weights_xy, feature_yz, weights_yz)

        # B, C, X, Y, Z = features_xyz.size()
        # logits = self.classifier(features_xyz.view(B, C, -1).transpose(1, 2))
        # logits = logits.permute(0, 2, 1).reshape(B, -1, X, Y, Z)
        logits = self.classifier(features_xyz)

        return logits


class VoxelDecoder0(nn.Module):
    def __init__(self, input_channels, n_classes, kernel_size=1, feature_channels=512):
        super().__init__()

        self.decoder_1 = VoxelDecoderScale(input_channels, n_classes, kernel_size, feature_channels)
        self.decoder_2 = VoxelDecoderScale(input_channels, n_classes, kernel_size, feature_channels)
        self.decoder_4 = VoxelDecoderScale(input_channels, n_classes, kernel_size, feature_channels)

    def forward(self, xy, xz, yz):
        output_1 = self.decoder_1((xy['rgb_1'], xz['rgb_1'], yz['rgb_1']))
        output_2 = self.decoder_2((xy['rgb_2'], xz['rgb_2'], yz['rgb_2']))
        output_4 = self.decoder_4((xy['rgb_4'], xz['rgb_4'], yz['rgb_4']))
        return {'voxel_1': output_1,
                'voxel_2': output_2,
                'voxel_4': output_4}


class VoxelDecoder1(nn.Module):
    def __init__(self, latent_n_channels, semantic_n_channels, feature_channels=512, constant_size=(3, 3, 1)):
        super().__init__()
        n_channels = feature_channels

        self.constant_tensor = nn.Parameter(torch.randn((2 * n_channels, *constant_size), dtype=torch.float32))

        # Input 512 x 3 x 3 x 1
        self.first_norm = AdaptiveInstanceNorm3d(latent_n_channels, out_channels=2 * n_channels)
        self.first_conv = ConvInstanceNorm3d(2 * n_channels, n_channels, latent_n_channels)
        # 512 x 3 x 3 x 1

        self.middle_conv = nn.ModuleList(
            [DecoderBlock3d(n_channels, n_channels, latent_n_channels, upsample=True) for _ in range(3)]
        )

        head_module = VoxelSemHead
        # 512 x 24 x 24 x 8
        self.conv1 = DecoderBlock3d(n_channels, n_channels // 2, latent_n_channels, upsample=True)
        self.head_4 = head_module(n_channels // 2, semantic_n_channels, downsample_factor=4)
        # 256 x 48 x 48 x 16

        self.conv2 = DecoderBlock3d(n_channels // 2, n_channels // 4, latent_n_channels, upsample=True)
        self.head_2 = head_module(n_channels // 4, semantic_n_channels, downsample_factor=2)
        # 128 x 96 x 96 x 32

        self.conv3 = DecoderBlock3d(n_channels // 4, n_channels // 8, latent_n_channels, upsample=True)
        self.head_1 = head_module(n_channels // 8, semantic_n_channels, downsample_factor=1)
        # 64 x 192 x 192 x 64

    def forward(self, w: Tensor) -> Tensor:
        b = w.shape[0]
        x = self.constant_tensor.unsqueeze(0).repeat([b, 1, 1, 1, 1])

        x = self.first_norm(x, w)
        x = self.first_conv(x, w)

        for module in self.middle_conv:
            x = module(x, w)

        x = self.conv1(x, w)
        output_4 = self.head_4(x)
        x = self.conv2(x, w)
        output_2 = self.head_2(x)
        x = self.conv3(x, w)
        output_1 = self.head_1(x)

        output = {**output_4, **output_2, **output_1}
        return output


class ConvDecoder(nn.Module):
    def __init__(self, latent_n_channels, out_channels, constant_size=(5, 13), mlp_layers=0, layer_norm=True,
                 activation=nn.ELU, head='rgb'):
        super().__init__()
        n_channels = 512
        if mlp_layers == 0:
            layers = [
                nn.Linear(latent_n_channels, n_channels),  # no activation here in dreamer v2
            ]
        else:
            hidden_dim = n_channels
            norm = nn.LayerNorm if layer_norm else nn.Identity
            layers = [
                nn.Linear(latent_n_channels, hidden_dim),
                norm(hidden_dim, eps=1e-3),
                activation(),
            ]
            for _ in range(mlp_layers - 1):
                layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    norm(hidden_dim, eps=1e-3),
                    activation()
                ]
        self.linear = nn.Sequential(*layers, nn.Unflatten(-1, (n_channels, 1, 1)))  # N x n_channels

        self.pre_transpose_conv = nn.Sequential(
            # *layers,
            # nn.Unflatten(-1, (n_channels, 1, 5)),
            # nn.ConvTranspose2d(n_channels, n_channels, kernel_size=5, stride=2),  # 5 x 13
            nn.ConvTranspose2d(n_channels, n_channels, kernel_size=constant_size),  # 5 x 13
            activation(),
            nn.ConvTranspose2d(n_channels, n_channels, kernel_size=5, stride=2, padding=2, output_padding=1),  # 10 x 26
            activation(),
            nn.ConvTranspose2d(n_channels, n_channels, kernel_size=5, stride=2, padding=2, output_padding=1),  # 20 x 52
            activation(),
            nn.ConvTranspose2d(n_channels, n_channels, kernel_size=6, stride=2, padding=2),  # 40 x 104
            activation(),
        )

        head_modules = {'rgb': RGBHead,
                        'bev': SegmentationHead,
                        'depth': DepthHead,
                        'sem_image': SemHead,
                        'lidar_re': LidarReHead,
                        'lidar_seg': LidarSegHead}
        head_module = head_modules[head] if head in head_modules else RGBHead

        self.trans_conv1 = nn.Sequential(
            nn.ConvTranspose2d(n_channels, 256, kernel_size=6, stride=2, padding=2),
            activation(),
        )
        self.head_4 = head_module(in_channels=256, n_classes=out_channels, downsample_factor=4)
        # 256 x 80 x 208

        self.trans_conv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=6, stride=2, padding=2),
            activation(),
        )
        self.head_2 = head_module(in_channels=128, n_classes=out_channels, downsample_factor=2)
        # 128 x 160 x 416

        self.trans_conv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=6, stride=2, padding=2),
            activation()
        )
        self.head_1 = head_module(in_channels=64, n_classes=out_channels, downsample_factor=1)
        # 64 x 320 x 832

    def forward(self, x):
        x = self.linear(x)  # N x n_channels x 1 x 1

        # x = x.repeat(1, 1, 1, 5)
        # N x n_channels x 1 x 5
        x = self.pre_transpose_conv(x)

        x = self.trans_conv1(x)
        output_4 = self.head_4(x)
        x = self.trans_conv2(x)
        output_2 = self.head_2(x)
        x = self.trans_conv3(x)
        output_1 = self.head_1(x)

        output = {**output_4, **output_2, **output_1}
        return output


# https://github.com/opendilab/InterFuser/blob/e0682c350892a243cf40bf448622743f4b26d0f3/interfuser/timm/models/interfuser.py#L66
class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        x = tensor
        _, _, h, w = x.shape
        not_mask = torch.ones((1, h, w), device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


# https://github.com/dotchen/LAV/blob/main/lav/models/point_pillar.py#L38
class DynamicPointNet(nn.Module):
    def __init__(self, num_input=9, num_features=[32, 32]):
        super().__init__()

        L = []
        for num_feature in num_features:
            L += [
                nn.Linear(num_input, num_feature),
                nn.BatchNorm1d(num_feature),
                nn.ReLU(inplace=True),
            ]

            num_input = num_feature

        self.net = nn.Sequential(*L)

    def forward(self, points, inverse_indices):
        """
        TODO: multiple layers
        """
        feat = self.net(points)
        feat_max = scatter_max(feat, inverse_indices, dim=0)[0]
        # feat_max = scatter_max(points, inverse_indices, dim=0)[0]
        return feat_max


class PointPillarNet(nn.Module):
    def __init__(self, num_input=9, num_features=[32, 32],
                 min_x=-10, max_x=70,
                 min_y=-40, max_y=40,
                 pixels_per_meter=4):
        super().__init__()
        self.point_net = DynamicPointNet(num_input, num_features)

        self.nx = (max_x - min_x) * pixels_per_meter
        self.ny = (max_y - min_y) * pixels_per_meter
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.pixels_per_meter = pixels_per_meter

    def decorate(self, points, unique_coords, inverse_indices):
        dtype = points.dtype
        x_centers = unique_coords[inverse_indices][:, 2:3].to(dtype) / self.pixels_per_meter + self.min_x
        y_centers = unique_coords[inverse_indices][:, 1:2].to(dtype) / self.pixels_per_meter + self.min_y

        xyz = points[:, :3]

        points_cluster = xyz - scatter_mean(xyz, inverse_indices, dim=0)[inverse_indices]

        points_xp = xyz[:, :1] - x_centers
        points_yp = xyz[:, 1:2] - y_centers

        features = torch.cat([points, points_cluster, points_xp, points_yp], dim=-1)
        return features

    def grid_locations(self, points):
        keep = (points[:, 0] >= self.min_x) & (points[:, 0] < self.max_x) & \
               (points[:, 1] >= self.min_y) & (points[:, 1] < self.max_y)
        points = points[keep, :]

        coords = (points[:, [0, 1]] - torch.tensor([self.min_x, self.min_y],
                                                   device=points.device)) * self.pixels_per_meter
        coords = coords.long()

        return points, coords

    def pillar_generation(self, points, coords):
        unique_coords, inverse_indices = coords.unique(return_inverse=True, dim=0)
        decorated_points = self.decorate(points, unique_coords, inverse_indices)

        return decorated_points, unique_coords, inverse_indices

    def scatter_points(self, features, coords, batch_size):
        canvas = torch.zeros(batch_size, features.shape[1], self.ny, self.nx, dtype=features.dtype,
                             device=features.device)
        canvas[coords[:, 0], :, torch.clamp(
            self.ny - 1 - coords[:, 1], 0, self.ny - 1), torch.clamp(coords[:, 2], 0, self.nx - 1)] = features
        return canvas

    def forward(self, lidar_list, num_points):
        batch_size = len(lidar_list)
        with torch.no_grad():
            coords = []
            filtered_points = []
            for batch_id, points in enumerate(lidar_list):
                points = points[:num_points[batch_id]]
                points, grid_yx = self.grid_locations(points)

                # batch indices
                grid_byx = torch.nn.functional.pad(grid_yx,
                                                   (1, 0), mode='constant', value=batch_id)

                coords.append(grid_byx)
                filtered_points.append(points)

            # batch_size, grid_y, grid_x
            coords = torch.cat(coords, dim=0)
            filtered_points = torch.cat(filtered_points, dim=0)

            decorated_points, unique_coords, inverse_indices = self.pillar_generation(filtered_points, coords)

        features = self.point_net(decorated_points, inverse_indices)

        return self.scatter_points(features, unique_coords, batch_size)