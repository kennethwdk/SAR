from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer, build_upsample_layer
from mmengine.structures import InstanceData
from torch import Tensor, nn
import numpy as np

from mmpose.registry import KEYPOINT_CODECS, MODELS
from mmpose.utils.typing import (ConfigType, Features, OptConfigType,
                                 OptSampleList, Predictions)

from mmpose.models.heads import BaseHead

OptIntSeq = Optional[Sequence[int]]


@MODELS.register_module()
class SARHead(BaseHead):
    _version = 2

    def __init__(self,
                 in_channels: Union[int, Sequence[int]],
                 out_channels: int,
                 deconv_out_channels: OptIntSeq = (256, 256, 256),
                 deconv_kernel_sizes: OptIntSeq = (4, 4, 4),
                 with_heatmap = True,
                 codec: OptConfigType = None,
                 init_cfg: OptConfigType = None):

        if init_cfg is None:
            init_cfg = self.default_init_cfg
        import math
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        override_dict = [dict(
                type='Normal',
                layer=['Conv2d'],
                std=0.001,
                override=dict(
                    name='logit_conv',
                    type='Normal',
                    std=0.001,
                    bias=bias_value))]
        if with_heatmap:
            override_dict += [dict(
                type='Normal',
                layer=['Conv2d'],
                std=0.001,
                override=dict(
                    name='heatmap_conv',
                    type='Normal',
                    std=0.001,
                    bias=bias_value))]
        init_cfg = init_cfg + override_dict
        super().__init__(init_cfg)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.codec = codec

        if deconv_out_channels:
            if deconv_kernel_sizes is None or len(deconv_out_channels) != len(
                    deconv_kernel_sizes):
                raise ValueError(
                    '"deconv_out_channels" and "deconv_kernel_sizes" should '
                    'be integer sequences with the same length. Got '
                    f'mismatched lengths {deconv_out_channels} and '
                    f'{deconv_kernel_sizes}')

            self.deconv_layers = self._make_deconv_layers(
                in_channels=in_channels,
                layer_out_channels=deconv_out_channels,
                layer_kernel_sizes=deconv_kernel_sizes,
            )
            in_channels = deconv_out_channels[-1]
        else:
            self.deconv_layers = nn.Identity()
        
        self.with_heatmap = with_heatmap
        if self.with_heatmap:
            heatmap_cfg = dict(
            type='Conv2d',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1)
            self.heatmap_conv = build_conv_layer(heatmap_cfg)

        logit_cfg = dict(
            type='Conv2d',
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1)
        offset_cfg = dict(
            type='Conv2d',
            in_channels=in_channels,
            out_channels=out_channels*2,
            kernel_size=1)
        self.logit_conv = build_conv_layer(logit_cfg)
        self.offset_conv = build_conv_layer(offset_cfg)

        self.heatmap_loss = FocalLoss()
        self.num_keypoints = out_channels

        # Register the hook to automatically convert old version state dicts
        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def _make_deconv_layers(self, in_channels: int,
                            layer_out_channels: Sequence[int],
                            layer_kernel_sizes: Sequence[int]) -> nn.Module:
        """Create deconvolutional layers by given parameters."""

        layers = []
        for out_channels, kernel_size in zip(layer_out_channels,
                                             layer_kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise ValueError(f'Unsupported kernel size {kernel_size} for'
                                 'deconvlutional layers in '
                                 f'{self.__class__.__name__}')
            cfg = dict(
                type='deconv',
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=False)
            layers.append(build_upsample_layer(cfg))
            layers.append(nn.BatchNorm2d(num_features=out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        return nn.Sequential(*layers)

    @property
    def default_init_cfg(self):
        init_cfg = [
            dict(
                type='Normal', layer=['Conv2d', 'ConvTranspose2d'], std=0.001),
            dict(type='Constant', layer='BatchNorm2d', val=1)
        ]
        return init_cfg

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
        return y
    
    @torch.no_grad()
    def locations(self, features):
        h, w = features.size()[-2:]
        device = features.device
        shifts_x = torch.arange(0, w, dtype=torch.float32, device=device)
        shifts_y = torch.arange(0, h, dtype=torch.float32, device=device)
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1) / w
        shift_y = shift_y.reshape(-1) / h
        locations = torch.stack((shift_x, shift_y), dim=1)
        locations = locations.reshape(h, w, 2).permute(2, 0, 1)
        return locations
    
    def forward(self, feats: Tuple[Tensor]) -> Tensor:
        """Forward the network. The input is multi scale feature maps and the
        output is the heatmap.

        Args:
            feats (Tuple[Tensor]): Multi scale feature maps.

        Returns:
            Tensor: output heatmap.
        """
        x = feats[-1]

        x = self.deconv_layers(x)
        bs, c, h, w = x.size()
        logit = self.logit_conv(x).sigmoid()
        offset = self.offset_conv(x).reshape(bs, self.num_keypoints, 2, h, w)
        location = self.locations(offset)[None, None]
        keypoint = location - offset

        ret = [logit, keypoint]
        if self.with_heatmap:
            heatmap = self.heatmap_conv(x)
            ret.append(heatmap)

        return ret

    def predict(self,
                feats: Features,
                batch_data_samples: OptSampleList,
                test_cfg: ConfigType = {}) -> Predictions:

        assert test_cfg.get('flip_test', False)
        if test_cfg.get('flip_test', False):
            # assert False, 'flip test is not support!'

            # TTA: flip test -> feats = [orig, flipped]
            assert isinstance(feats, list) and len(feats) == 2
            flip_indices = batch_data_samples[0].metainfo['flip_indices']
            input_size = batch_data_samples[0].metainfo['input_size']

            _feats, _feats_flip = feats
            batch_rets = self.forward(_feats)
            flip_batch_rets = self.forward(_feats_flip)
            preds, vals = self.flip_decode(batch_rets, flip_batch_rets, flip_indices, input_size)
        else:
            batch_rets = self.forward(feats)
            preds, vals = self.decode(batch_rets)
        
        preds = [
            InstanceData(keypoints=keypoints[None, :, :], keypoint_scores=scores[None, :])
            for keypoints, scores in zip(preds, vals)
        ]

        return preds
    
    def flip_decode(self, batch_rets, flip_batch_rets, flip_indices, input_size):
        logit, keypoint = batch_rets[:2]
        bs, k, h, w = logit.shape
        logits = logit.reshape(bs*k, h*w)
        logits = logits / logits.sum(dim=1, keepdim=True)
        keypoints = keypoint.reshape(bs*k, 2, h*w).permute(0, 2, 1)
        maxvals, maxinds = logits.max(dim=1)
        coords = keypoints[torch.arange(bs*k, dtype=torch.long).to(keypoints.device), maxinds]

        logit, keypoint = flip_batch_rets[:2]
        bs, k, h, w = logit.shape
        logits = logit.reshape(bs*k, h*w)
        logits = logits / logits.sum(dim=1, keepdim=True)
        keypoints = keypoint.reshape(bs*k, 2, h*w).permute(0, 2, 1)
        maxvals_flip, maxinds = logits.max(dim=1)
        coords_flip = keypoints[torch.arange(bs*k, dtype=torch.long).to(keypoints.device), maxinds]

        coords_flip[:, 0] = 1 - coords_flip[:, 0] - 1.0 / (w * 4)
        coords_flip = coords_flip.reshape(bs, k, 2)[:, flip_indices, :].reshape(bs*k, 2)
        maxvals_flip = maxvals_flip.reshape(bs, k)[:, flip_indices].reshape(-1)
        coords = (coords + coords_flip) / 2.0
        coords[..., 0] *= w
        coords[..., 1] *= h
        maxvals = (maxvals + maxvals_flip) / 2.0

        # hmvals
        # heatmap = batch_rets[2]
        # heatmap_flip = flip_batch_rets[2]
        # heatmap_flip = heatmap_flip.flip(3)[:, flip_indices, :, :]
        # heatmap = (heatmap + heatmap_flip) / 2.0
        # bs, k, h, w = heatmap.shape
        # heatmap = heatmap.reshape(bs*k, 1, h, w).sigmoid()
        # coord_inds = torch.stack((
        #     coords[:, 0] / (w - 1) * 2 - 1,
        #     coords[:, 1] / (h - 1) * 2 - 1,
        # ), dim=-1)
        # coord_inds = coord_inds[:, None, None, :]
        # keypoint_scores = torch.nn.functional.grid_sample(
        #     heatmap, coord_inds,
        #     padding_mode='border').reshape(bs*k, -1)
        # maxvals = keypoint_scores

        preds = coords.reshape(bs, k, 2).cpu().numpy()
        maxvals = maxvals.reshape(bs, k).cpu().numpy()
        if self.codec.get('type', 'MSRAHeatmap') == 'UDPHeatmap':
            preds = preds / [w-1, h-1]
            preds = preds * self.codec['input_size']
        else:
            stride = self.codec['input_size'][0] / self.codec['heatmap_size'][0]
            preds = preds * stride

        return preds, maxvals

    def decode(self, batch_rets):
        logit, keypoint = batch_rets[:2]
        bs, k, h, w = logit.shape
        logits = logit.reshape(bs*k, h*w)
        logits = logits / logits.sum(dim=1, keepdim=True)
        keypoints = keypoint.reshape(bs*k, 2, h*w).permute(0, 2, 1)
        maxvals, maxinds = logits.max(dim=1)
        coords = keypoints[torch.arange(bs*k, dtype=torch.long).to(keypoints.device), maxinds]
        coords[..., 0] *= w
        coords[..., 1] *= h

        # hmvals
        # heatmap = batch_rets[2]
        # bs, k, h, w = heatmap.shape
        # heatmap = heatmap.reshape(bs*k, 1, h, w).sigmoid()
        # coord_inds = torch.stack((
        #     coords[:, 0] / (w - 1) * 2 - 1,
        #     coords[:, 1] / (h - 1) * 2 - 1,
        # ), dim=-1)
        # coord_inds = coord_inds[:, None, None, :]
        # keypoint_scores = torch.nn.functional.grid_sample(
        #     heatmap, coord_inds,
        #     padding_mode='border').reshape(bs*k, -1)
        # maxvals = keypoint_scores

        preds = coords.reshape(bs, k, 2).cpu().numpy()
        maxvals = maxvals.reshape(bs, k).cpu().numpy()
        if self.codec.get('type', 'MSRAHeatmap') == 'UDPHeatmap':
            preds = preds / [w-1, h-1]
            preds = preds * self.codec['input_size']
        else:
            stride = self.codec['input_size'][0] / self.codec['heatmap_size'][0]
            preds = preds * stride

        return preds, maxvals

    def loss(self,
             feats: Tuple[Tensor],
             batch_data_samples: OptSampleList,
             train_cfg: ConfigType = {}) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            feats (Tuple[Tensor]): The multi-stage features
            batch_data_samples (List[:obj:`PoseDataSample`]): The batch
                data samples
            train_cfg (dict): The runtime config for training process.
                Defaults to {}

        Returns:
            dict: A dictionary of losses.
        """
        pred_fields = self.forward(feats)
        gt_heatmaps = torch.stack(
            [d.gt_fields.heatmaps for d in batch_data_samples])
        gt_weights = torch.cat([
            d.gt_instance_labels.keypoint_weights for d in batch_data_samples
        ])
        gt_keypoints = torch.cat([
            d.gt_instance_labels.keypoint_labels for d in batch_data_samples
        ])

        logit, keypoint = pred_fields[:2]
        if self.with_heatmap: heatmap = pred_fields[2]

        bs, k, h, w = logit.size()
        assert k == keypoint.size(1) and keypoint.size(2) == 2

        valid_mask = gt_weights.reshape(bs*k) > 0
        # get cls score
        cls_score = logit.reshape(bs*k, h*w)[valid_mask]
        # get reg score
        pred_keypoints = keypoint.reshape(bs*k, 2, h*w).permute(0, 2, 1)
        gt_keypoints = gt_keypoints.reshape(bs*k, 2)[:, None, :]
        hh, ww = gt_heatmaps.shape[2:]
        gt_keypoints[..., 0] /= ww
        gt_keypoints[..., 1] /= hh

        dist_mat = torch.abs(pred_keypoints - gt_keypoints)
        dist_mat = dist_mat * 16.0
        reg_score = torch.exp(-dist_mat.sum(dim=2))[valid_mask]
        
        norm_cls_score = cls_score / cls_score.sum(dim=1, keepdim=True)
        normcls2reg_loss = torch.sum(norm_cls_score * reg_score, dim=1)
        normcls2reg_loss = -torch.log(normcls2reg_loss + 1e-6)
        loss = normcls2reg_loss.mean()

        if self.with_heatmap:
            bs, k, h, w = gt_heatmaps.size()
            heatmap = heatmap.reshape(bs*k, h, w)[valid_mask]
            gt_heatmap = gt_heatmaps.reshape(bs*k, h, w)[valid_mask]
            pos_label = gt_heatmap > 0
            num_pos = torch.sum(gt_heatmap > 0.7).item()
            heatmap_loss = gfl_loss(heatmap, gt_heatmap, pos_label) / num_pos
            loss = loss + heatmap_loss

        # calculate losses
        losses = dict()

        losses.update(loss_kpt=loss)

        return losses

    def _load_state_dict_pre_hook(self, state_dict, prefix, local_meta, *args,
                                  **kwargs):
        """A hook function to convert old-version state dict of
        :class:`DeepposeRegressionHead` (before MMPose v1.0.0) to a
        compatible format of :class:`RegressionHead`.

        The hook will be automatically registered during initialization.
        """
        version = local_meta.get('version', None)
        if version and version >= self._version:
            return

        # convert old-version state dict
        keys = list(state_dict.keys())
        for _k in keys:
            if not _k.startswith(prefix):
                continue
            v = state_dict.pop(_k)
            k = _k[len(prefix):]
            # In old version, "final_layer" includes both intermediate
            # conv layers (new "conv_layers") and final conv layers (new
            # "final_layer").
            #
            # If there is no intermediate conv layer, old "final_layer" will
            # have keys like "final_layer.xxx", which should be still
            # named "final_layer.xxx";
            #
            # If there are intermediate conv layers, old "final_layer"  will
            # have keys like "final_layer.n.xxx", where the weights of the last
            # one should be renamed "final_layer.xxx", and others should be
            # renamed "conv_layers.n.xxx"
            k_parts = k.split('.')
            if k_parts[0] == 'final_layer':
                if len(k_parts) == 3:
                    assert isinstance(self.conv_layers, nn.Sequential)
                    idx = int(k_parts[1])
                    if idx < len(self.conv_layers):
                        # final_layer.n.xxx -> conv_layers.n.xxx
                        k_new = 'conv_layers.' + '.'.join(k_parts[1:])
                    else:
                        # final_layer.n.xxx -> final_layer.xxx
                        k_new = 'final_layer.' + k_parts[2]
                else:
                    # final_layer.xxx remains final_layer.xxx
                    k_new = k
            else:
                k_new = k

            state_dict[prefix + k_new] = v

class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.alpha = 2
        self.beta = 4

    def forward(self, pred, gt, mask=None):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        if mask is not None:
            pos_inds = pos_inds * mask
            neg_inds = neg_inds * mask

        neg_weights = torch.pow(1 - gt, self.beta)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss

def gfl_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    pos_label: torch.Tensor,
    gamma: float = 2,
):
    score = targets
    pred = inputs
    # negatives are supervised by 0 quality score
    pred_sigmoid = pred.sigmoid()
    scale_factor = pred_sigmoid
    zerolabel = scale_factor.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
        pred, zerolabel, reduction='none') * scale_factor.pow(gamma)

    # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    # positives are supervised by bbox quality (IoU) score
    scale_factor = score[pos_label] - pred_sigmoid[pos_label]
    loss[pos_label] = F.binary_cross_entropy_with_logits(
        pred[pos_label], score[pos_label],
        reduction='none') * scale_factor.abs().pow(gamma)

    loss = loss.sum()
    return loss