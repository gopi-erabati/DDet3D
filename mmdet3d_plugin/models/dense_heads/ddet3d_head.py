import copy
import warnings
import math
import random

import mmcv
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from mmcv.runner import force_fp32, BaseModule, ModuleList
from mmcv.cnn import build_activation_layer
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.ops import MultiScaleDeformableAttention
from mmengine.structures import InstanceData

from mmdet.core import build_assigner, bbox2roi, multi_apply, build_sampler
from mmdet.core.utils import reduce_mean
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet3d.core import box3d_multiclass_nms, xywhr2xyxyr
from mmdet3d.models import HEADS, build_loss, build_head, build_roi_extractor
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from ...core.bbox.util import (normalize_bbox, denormalize_bbox,
                               normalize_0to1_bbox,
                               denormalize_0to1_bbox,
                               boxes3d_to_corners3d)

_DEFAULT_SCALE_CLAMP = math.log(100000.0 / 16)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in
    https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 1  # 1001
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)  # (1001, )
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2  # (1001, )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # (1001, )
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])  # (1000, )
    return torch.clip(betas, 0, 0.999)  # (1000, )


def extract(a, t, x_shape):
    """
     a (1000, ) ; t (1, )   ; x_shape (n_p, 10)       --- training
     a (1000, ) ; t (bs, )  ; x_shape (bs, n_p, 10)   --- testing
     """
    """extract the appropriate t index for a batch of indices."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)  # (bs, )
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    # (1, 1)    -- training
    # (bs, 1, 1) -- testing


class SinusoidalPositionEmbeddings(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2  # 128
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings)  # (128, )
        embeddings = time[:, None] * embeddings[None, :]  # (b_s, 128)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings  # (b_s, 256)


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        """Forward
        Args:
            xyz (tensor): shape (BS, n_q, 2)
        """
        xyz = xyz.transpose(1, 2).contiguous()  # (BS, 2, n_q)
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding  # (BS, 128, n_q)


@HEADS.register_module()
class DynamicDDet3DHead(BaseDenseHead):
    """
    This is the head for DDet3D (Diffusion based 3D Object Detection)
    """

    def __init__(self,
                 num_classes=10,
                 feat_channels=256,
                 num_proposals=900,
                 num_heads=6,
                 deep_supervision=True,
                 prior_prob=0.01,
                 snr_scale=2.0,
                 timesteps=1000,
                 sampling_timesteps=1,
                 ddim_sampling_eta=1.0,
                 box_renewal=True,
                 use_ensemble=True,
                 sync_cls_avg_factor=True,
                 code_weights=None,
                 size_norm=None,
                 with_lidar_encoder=False,
                 grid_size=None,
                 out_size_factor=8,
                 lidar_encoder_cfg=None,
                 single_head=None,
                 roi_extractor=None,
                 roi_extractor_img=None,
                 loss_cls=None,
                 loss_bbox=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None
                 ):
        super(DynamicDDet3DHead, self).__init__(init_cfg)

        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.num_proposals = num_proposals
        self.num_heads = num_heads
        self.deep_supervision = deep_supervision
        self.prior_prob = prior_prob
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.code_weights = code_weights
        self.size_norm = size_norm
        self.pc_range = single_head['pc_range']
        self.test_cfg = test_cfg
        self.with_lidar_encoder = with_lidar_encoder

        # Build Diffusion
        assert isinstance(timesteps, int), 'The type of `timesteps` should ' \
                                           f'be int but got {type(timesteps)}'
        assert sampling_timesteps <= timesteps
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.snr_scale = snr_scale
        self.ddim_sampling = self.sampling_timesteps < self.timesteps
        self.ddim_sampling_eta = ddim_sampling_eta
        self.box_renewal = box_renewal
        self.use_ensemble = use_ensemble
        self._build_diffusion()

        # Build Encoder for LiDAR BEV
        if self.with_lidar_encoder:
            self.encoder_lidar = build_transformer_layer_sequence(
                lidar_encoder_cfg)

            # BEV position Embeddings
            self.bev_pos_encoder_mlvl_embed = nn.ModuleList()
            lidar_feat_lvls = lidar_encoder_cfg.transformerlayers.attn_cfgs.num_levels
            for _ in range(lidar_feat_lvls):
                self.bev_pos_encoder_mlvl_embed.append(
                    PositionEmbeddingLearned(
                        2, feat_channels))

            # BEV Level Embeddings
            self.bev_level_embeds = nn.Parameter(torch.Tensor(
                lidar_feat_lvls, feat_channels))

            # BEV Pos for Multi-levels
            x_size = grid_size[0] // out_size_factor
            y_size = grid_size[1] // out_size_factor
            self.bev_pos_mlvl = []
            for lvl in range(lidar_feat_lvls):
                self.bev_pos_mlvl.append(
                    self.create_2D_grid(int(x_size / (2 ** lvl)),
                                        int(y_size / (2 ** lvl))))

        # Build Dynamic Head
        single_head_ = single_head.copy()
        single_head_num_classes = single_head_.get('num_classes', None)
        if single_head_num_classes is None:
            single_head_.update(num_classes=num_classes)
        else:
            if single_head_num_classes != num_classes:
                warnings.warn(
                    'The `num_classes` of `DynamicDDet3DHead` and '
                    '`SingleDDet3DHead` should be same, changing '
                    f'`single_head.num_classes` to {num_classes}')
                single_head_.update(num_classes=num_classes)

        single_head_feat_channels = single_head_.get('feat_channels', None)
        if single_head_feat_channels is None:
            single_head_.update(feat_channels=feat_channels)
        else:
            if single_head_feat_channels != feat_channels:
                warnings.warn(
                    'The `feat_channels` of `DynamicDDet3DHead` and '
                    '`SingleDDet3DHead` should be same, changing '
                    f'`single_head.feat_channels` to {feat_channels}')
                single_head_.update(feat_channels=feat_channels)

        default_pooler_resolution = roi_extractor['roi_layer'].get(
            'output_size')
        assert default_pooler_resolution is not None
        single_head_pooler_resolution = single_head_.get('pooler_resolution')
        if single_head_pooler_resolution is None:
            single_head_.update(pooler_resolution=default_pooler_resolution)
        else:
            if single_head_pooler_resolution != default_pooler_resolution:
                warnings.warn(
                    'The `pooler_resolution` of `DynamicDDet3DHead` '
                    'and `SingleDDet3DHead` should be same, changing '
                    f'`single_head.pooler_resolution` to {num_classes}')
                single_head_.update(
                    pooler_resolution=default_pooler_resolution)

        self.use_fed_loss = False
        self.use_focal_loss = True

        single_head_.update(
            use_focal_loss=self.use_focal_loss, use_fed_loss=self.use_fed_loss)
        single_head_module = build_head(single_head_)

        self.head_series = ModuleList(
            [copy.deepcopy(single_head_module) for _ in range(num_heads)])

        # Build ROI Extractor
        self.roi_extractor = build_roi_extractor(roi_extractor)
        if roi_extractor_img is not None:
            self.roi_extractor_img = build_roi_extractor(roi_extractor_img)
        else:
            self.roi_extractor_img = None

        # Build Losses
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)

        # Build Assigner
        if train_cfg:
            self.assigner = build_assigner(train_cfg.assigner)
            # for Hungarian
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        # for Hungarian
        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)

        # for Hungarian
        # Background Class weights and positive class weights
        self.bg_cls_weight = 0
        if loss_cls:
            class_weight = loss_cls.get('class_weight', None)
        else:
            class_weight = None
        if class_weight is not None:
            assert isinstance(class_weight, float), 'Expected ' \
                                                    'class_weight to have ' \
                                                    'type float. Found ' \
                                                    f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                                                     'bg_cls_weight to have ' \
                                                     'type float. Found ' \
                                                     f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        # for Hungarian
        # classes
        if loss_cls:
            if loss_cls.use_sigmoid:
                self.cls_out_channels = num_classes
            else:
                self.cls_out_channels = num_classes + 1

        # Gaussian random feature embedding layer for time
        time_dim = feat_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(feat_channels),
            nn.Linear(feat_channels, time_dim), nn.GELU(),
            nn.Linear(time_dim, time_dim))

        self.use_nms = self.test_cfg.get('use_nms', True)

        # srcn3d boxes
        # self.init_proposal_boxes = nn.Embedding(self.num_proposals, 10)
        # self.init_feats = nn.Embedding(self.num_proposals, feat_channels)

        self._init_weights()  # No for srcn3d

    def _init_weights(self):
        # init all parameters.
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        for name, p in self.named_parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

            # initialize the bias for focal loss and fed loss.
            if self.use_focal_loss or self.use_fed_loss:
                if (p.shape[-1] == self.num_classes or \
                    p.shape[-1] == self.num_classes + 1) and (
                        not name == 'code_weights'):
                    nn.init.constant_(p, bias_value)
        if self.with_lidar_encoder:
            nn.init.normal_(self.bev_level_embeds)
            for m in self.modules():
                if isinstance(m, MultiScaleDeformableAttention):
                    m.init_weights()

    def _build_diffusion(self):
        betas = cosine_beta_schedule(self.timesteps)  # (1000, )
        alphas = 1. - betas  # (1000, )
        alphas_cumprod = torch.cumprod(alphas, dim=0)  # (1000, )
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(posterior_variance.clamp(min=1e-20)))
        self.register_buffer(
            'posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * torch.sqrt(alphas) /
                             (1. - alphas_cumprod))

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base  # (1, x_size * y_size, 2)

    def forward_train(self,
                      img_feats, point_feats,
                      gt_bboxes, gt_labels,
                      gt_bboxes_ignore=None,
                      img_metas=None, feats_init=None,
                      points=None, epoch=None):
        """Forward function for training mode.

        Args:
            img_feats (list[Tensor] | None): Image feats list of stride 4, 8,
                16, 32 of shape (bs, n_cam, C, H, W)
            point_feats (list[Tensor]): Point feat list [(B, 128, H, W)...]
                strides 8, 16, 32, 64 of 1472
            gt_bboxes (list[Tensor]): Ground truth bboxes ,
                shape (num_gts, 9).
            gt_labels (list[Tensor]): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 9).
            img_metas (list[dict]): list of img_metas for all batch samples
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # prepare the training targets
        prepare_targets = self.prepare_training_targets(gt_bboxes, gt_labels)
        (batch_gt_instances, batch_pred_instances) = prepare_targets
        # ([gt_inst, ...]: bboxes (n_gt, 9), labels (n_gt, ),
        # [pred_inst,...]: time, diff_bboxes (n_p, 10),
        #                   diff_bboxes_abs (n_p, 10), noise (n_p, 10) all xyz)

        batch_diff_bboxes = torch.stack([
            pred_instances.diff_bboxes_abs
            for pred_instances in batch_pred_instances
        ])  # (bs, n_p, 10)
        batch_time = torch.stack(
            [pred_instances.time for pred_instances in batch_pred_instances])
        # (bs, )

        # # code to get gt boxes to pass to forward to test
        # batch_diff_bboxes = torch.stack([torch.cat([gt_instance.bboxes[..., :6
        #                                             ], gt_instance.bboxes[
        #     ..., 6:7].sin(), gt_instance.bboxes[..., 6:7].cos(),
        #                                             gt_instance.bboxes[...,
        #                                             7:9]], dim=1)
        #                                  for
        #                                  gt_instance
        #                                  in batch_gt_instances])

        # # bboxes init in srcn3d style
        # bs = img_feats[0].shape[0]  # batch size
        # batch_diff_bboxes = self.init_proposal_boxes.weight
        # batch_diff_bboxes = batch_diff_bboxes.repeat(bs, 1).view(bs,
        #                                                          batch_diff_bboxes.shape[0], batch_diff_bboxes.shape[1])
        # # (bs, n_p, 10)
        # feats_init = self.init_feats.weight
        # feats_init = feats_init.repeat(bs, 1).view(bs, feats_init.shape[0],
        #                                            feats_init.shape[1])

        pred_logits, pred_bboxes = self(img_feats, point_feats,
                                        batch_diff_bboxes, batch_time,
                                        img_metas, feats_init=feats_init)
        # (#lay, bs, n_p, #cls), (#lay, bs, n_p, 10)
        # pred boxes  center:abs and size:log
        # [cx, cy, cz, w, l, h, sin, cos, vx, vy]

        output = {
            'pred_logits': pred_logits[-1],
            'pred_boxes': pred_bboxes[-1]
        }
        if self.deep_supervision:
            output['aux_outputs'] = [{
                'pred_logits': a,
                'pred_boxes': b
            } for a, b in zip(pred_logits[:-1], pred_bboxes[:-1])]
        # output = {'pred_logits':(bs, n_p, #cls),
        #           'pred_boxes':(bs, n_p, 10),
        #           'aux_outputs': [{'pred_logits':(bs, n_p, #cls),
        #                            'pred_boxes':(bs, n_p, 10)},
        #                            { }, { }, ...]}
        # pred boxes  center:abs and size:log
        # [cx, cy, cz, w, l, h, sin, cos, vx, vy]

        losses = self.loss(output, batch_gt_instances)

        return losses

    def forward(self, img_feats, point_feats, bboxes_init, time_init,
                img_metas, feats_init=None):
        """
        img_feats (list[Tensor]): shape (bs, n_cam, C, H, W)
        point_feats (list[Tensor]): shape (bs, C, H, W)
        bboxes_init (Tensor): (bs, n_p, 10)
        time_init (Tensor): (bs, )
        feats_init (Tensor): (bs, n_p, 256)
        """
        # time embedding
        time = self.time_mlp(time_init)  # (bs, 256*4)

        inter_class_logits = []
        inter_pred_bboxes = []

        if img_feats is not None:
            batch_size = len(img_feats[0])
        else:
            batch_size = len(point_feats[0])
        bboxes = bboxes_init

        if feats_init is not None:
            # feats_init = feats_init[None].repeat(1, bs, 1)
            prop_feats = feats_init.clone()
        else:
            prop_feats = None

        # Encoder for LiDAR BEV feats if with_lidar_encoder=True
        if self.with_lidar_encoder:
            point_feats = self._get_lidar_encoder_feats(point_feats)
            # (list[Tensor]): shape (bs, C, H, W)

        # apply sigmoid for center
        # bboxes[..., :3] = bboxes[..., :3].sigmoid()
        bboxes[..., 3:6] = bboxes[..., 3:6].log()  # not for srcn3d

        for head_idx, single_head in enumerate(self.head_series):
            class_logits, pred_bboxes, prop_feats = single_head(
                img_feats, point_feats, bboxes, prop_feats,
                self.roi_extractor, time, img_metas, self.roi_extractor_img)
            # (bs, n_p, #cls), (bs, n_p, 10), (bs*n_p, 256)
            # pred boxes  center:abs and size:log
            # [cx, cy, cz, w, l, h, sin, cos, vx, vy]
            # pred_bboxes[..., 3:6] = torch.exp(pred_bboxes[..., 3:6])
            if self.deep_supervision:
                inter_class_logits.append(class_logits)
                inter_pred_bboxes.append(pred_bboxes.clone())
            bboxes = pred_bboxes.clone().detach()

        if self.deep_supervision:
            inter_class_logits = torch.stack(inter_class_logits)
            inter_pred_bboxes = torch.stack(inter_pred_bboxes)
            # # center denormalize
            # pc_range_ = bboxes.new_tensor(
            #     [[self.pc_range[3] - self.pc_range[0],
            #       self.pc_range[4] - self.pc_range[1],
            #       self.pc_range[5] - self.pc_range[
            #           2]]])  # (1, 3)
            # pc_start_ = bboxes.new_tensor(
            #     [[self.pc_range[0], self.pc_range[1],
            #       self.pc_range[2]]])  # (1, 3)
            # inter_pred_bboxes[..., :3] = (inter_pred_bboxes[..., :3] * pc_range_) + \
            #                        pc_start_
            # # (n_p, 3)
            return inter_class_logits, inter_pred_bboxes
            # (#lay, bs, n_p, #cls), (#lay, bs, n_p, 10)
            # pred boxes  center:abs and size:log
            # [cx, cy, cz, w, l, h, sin, cos, vx, vy]
        else:
            # center denormalize
            pc_range_ = bboxes.new_tensor(
                [[self.pc_range[3] - self.pc_range[0],
                  self.pc_range[4] - self.pc_range[1],
                  self.pc_range[5] - self.pc_range[
                      2]]])  # (1, 3)
            pc_start_ = bboxes.new_tensor(
                [[self.pc_range[0], self.pc_range[1],
                  self.pc_range[2]]])  # (1, 3)
            pred_bboxes[..., :3] = (pred_bboxes[..., :3] * pc_range_) + \
                                   pc_start_
            # (n_p, 3)
            return class_logits[None, ...], pred_bboxes[None, ...]

    def prepare_training_targets(self, gt_bboxes, gt_labels):
        """ This function is to prepare training targets from the
        groundtruths

        Args:
            gt_bboxes (list[Tensor]): Ground truth bboxes ,
                shape (num_gts, 9). format :obj:`LiDARInstance3DBoxes`
            gt_labels (list[Tensor]): Ground truth labels of each box,
                shape (num_gts,).

        Returns:

        """
        # hard-setting seed to keep results same (if necessary)
        # random.seed(0)
        # torch.manual_seed(0)
        # torch.cuda.manual_seed_all(0)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

        batch_gt_instances = []
        batch_pred_instances = []
        for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
            # put gt_bboxes in gt_istances of mmengine for nice data structure
            gt_instances = InstanceData()
            # convert gt_boxes from bottom_center to gravity_center
            device = gt_label.device
            gt_bbox = torch.cat((gt_bbox.gravity_center, gt_bbox.tensor[:,
                                                         3:]), dim=1).to(
                device)
            gt_instances.bboxes = gt_bbox  # (n_gt, 9)
            gt_instances.labels = gt_label  # (n_gt, )

            # Normalize 0 to 1 gt_bbox
            norm_gt_bbox = normalize_0to1_bbox(gt_bbox,
                                               self.pc_range,
                                               self.size_norm)  # (
            # n_gt, 10)
            # [cx, cy, cz, w, l, h, sin, cos, vx, vy]
            pred_instances = self.prepare_diffusion(norm_gt_bbox)
            gt_instances.norm_bbox = norm_gt_bbox

            batch_gt_instances.append(gt_instances)
            batch_pred_instances.append(pred_instances)

        return batch_gt_instances, batch_pred_instances
        # ([gt_inst, ...]: bboxes (n_gt, 9), labels (n_gt, ),
        # [pred_inst,...]: time, diff_bboxes (n_p, 10),
        #                   diff_bboxes_abs (n_p, 10), noise (n_p, 10) all xyz)

    def prepare_diffusion(self, gt_boxes):
        """
        Prepare diffusion (noisy) gt boxes as initial bboxes
        gt_bboxes: (n_gt, 10)
        """
        device = gt_boxes.device
        time = torch.randint(
            0, self.timesteps, (1,), dtype=torch.long, device=device)  # (1,)
        noise = torch.randn(self.num_proposals, len(self.code_weights),
                            device=device)
        # (n_p, 10)

        num_gt = gt_boxes.shape[0]
        if num_gt < self.num_proposals:
            # 3 * sigma = 1/2 --> sigma: 1/6
            box_placeholder = torch.randn(
                self.num_proposals - num_gt, len(self.code_weights),
                device=device) / 6. + 0.5
            # to make the values lie between 0 to 1 ; 3 sigma rule
            box_placeholder[:, 3:6] = torch.clip(
                box_placeholder[:, 3:6], min=1e-4)
            x_start = torch.cat((gt_boxes, box_placeholder), dim=0)
        else:
            select_mask = [True] * self.num_proposals + \
                          [False] * (num_gt - self.num_proposals)
            random.shuffle(select_mask)
            x_start = gt_boxes[select_mask]  # (n_p, 10)

        x_start = (x_start * 2. - 1.) * self.snr_scale  # (n_p, 10)

        # noise sample
        x = self.q_sample(x_start=x_start, time=time, noise=noise)  # (n_p, 10)

        x = torch.clamp(x, min=-1 * self.snr_scale, max=self.snr_scale)
        x = ((x / self.snr_scale) + 1) / 2.  # (n_p, 10)

        # change any size=0 to min value so that log wont get -inf
        size = x[:, 3:6].clone()  # (n_p, 3)
        size[size == 0.0] = 1e-4
        x[:, 3:6] = size

        diff_bboxes = x
        # convert to abs bboxes
        diff_bboxes_abs = denormalize_0to1_bbox(diff_bboxes, self.pc_range,
                                                self.size_norm)
        # (n_p, 10)

        metainfo = dict(time=time.squeeze(-1))
        pred_instances = InstanceData(metainfo=metainfo)
        pred_instances.diff_bboxes = diff_bboxes
        pred_instances.diff_bboxes_abs = diff_bboxes_abs
        pred_instances.noise = noise
        return pred_instances

    # forward diffusion
    def q_sample(self, x_start, time, noise=None):
        """ sampling the q
        Args:
            x_start (Tensor): (n_p, 10)
            time (Tensor): (1,)
            noise (Tensor): (n_p, 10)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        x_start_shape = x_start.shape  # (n_p, 10)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, time,
                                        x_start_shape)  # (1, 1)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, time, x_start_shape)  # (1, 1)

        return sqrt_alphas_cumprod_t * x_start + \
            sqrt_one_minus_alphas_cumprod_t * noise  # (n_p, 10)

    def _get_lidar_encoder_feats(self, lidar_feats):
        """
        This function is to get LiDAR BEV Encoder
        features with MultiScaleDeformAttn
        """
        batch_size = lidar_feats[0].shape[0]
        # repeat the BEV positions for all batches
        bev_pos_mlvl_bs = []
        for bev_pos_lvl in self.bev_pos_mlvl:
            bev_pos_lvl = bev_pos_lvl.repeat(batch_size, 1, 1).to(
                lidar_feats[0].device)  # (bs, H*W, 2)
            bev_pos_mlvl_bs.append(bev_pos_lvl)

        # Encoder: MS Deformable Attention for LiDAR features
        # get the BEV positions and embeddings of all levels with level
        # embed and also feats of all levels flatten
        bev_pos_encoder_mlvl_norm = []
        bev_pos_encoder_mlvl_embed = []
        bev_spatial_shape_mlvl = []
        lidar_feat_mlvl = []
        for idx, (bev_pos_lvl, lidar_feat) in enumerate(
                zip(bev_pos_mlvl_bs, lidar_feats)):
            bev_pos_encoder_lvl_embed = self.bev_pos_encoder_mlvl_embed[
                idx](bev_pos_lvl)  # (bs, h_dim, H*W)
            bev_pos_encoder_lvl_embed = \
                bev_pos_encoder_lvl_embed.permute(0, 2, 1)
            # (bs, H*W, h_dim)
            bev_pos_encoder_lvl_embed = bev_pos_encoder_lvl_embed + \
                                        self.bev_level_embeds[idx].view(
                                            1, 1, -1)  # (bs, H*W, h_dim)
            bev_pos_encoder_mlvl_embed.append(bev_pos_encoder_lvl_embed)

            # LiDAR feats
            lidar_feat_bs, lidar_feat_dim, lidar_feat_h, lidar_feat_w = \
                lidar_feat.shape
            bev_spatial_shape = (lidar_feat_h, lidar_feat_w)
            bev_spatial_shape_mlvl.append(bev_spatial_shape)
            lidar_feat = lidar_feat.flatten(2).permute(0, 2, 1)
            # (bs, H*W, h_dim)
            lidar_feat_mlvl.append(lidar_feat)

            # normalize bev_pos_encoder_lvl with lidar_feat_h and
            # lidar_feat_w to make them lie in [0, 1] for reference points
            bev_pos_encoder_lvl_norm = bev_pos_lvl.float()
            bev_pos_encoder_lvl_norm[..., 0] /= lidar_feat_h
            bev_pos_encoder_lvl_norm[..., 1] /= lidar_feat_w
            bev_pos_encoder_mlvl_norm.append(bev_pos_encoder_lvl_norm)

        # concatenate all levels
        lidar_feat_mlvl = torch.cat(lidar_feat_mlvl, dim=1)
        # (bs, lvl*H*W, h_dim)
        bev_pos_encoder_mlvl_norm = torch.cat(bev_pos_encoder_mlvl_norm,
                                              dim=1)
        # (bs, lvl*H*W, 2) normalized
        # repeat the bev_pos_encoder_mlvl (reference points) for all levels
        bev_pos_encoder_mlvl_norm = \
            bev_pos_encoder_mlvl_norm.unsqueeze(2).repeat(1, 1,
                                                          len(lidar_feats),
                                                          1)
        # (bs, lvl*H*W, lvls, 2)  normalized for reference points
        bev_pos_encoder_mlvl_embed = torch.cat(
            bev_pos_encoder_mlvl_embed, dim=1)  # (bs, lvl*H*W, h_dim)
        bev_spatial_shape_mlvl_tensor = torch.as_tensor(
            bev_spatial_shape_mlvl, dtype=torch.long,
            device=lidar_feat_mlvl.device)  # (lvl, 2)
        bev_level_start_index = torch.cat(
            (bev_spatial_shape_mlvl_tensor.new_zeros(
                (1,)),
             bev_spatial_shape_mlvl_tensor.prod(1).cumsum(0)[
             :-1]))  # (lvl, )

        # reshape according to encoder expectation
        lidar_feat_mlvl = lidar_feat_mlvl.permute(1, 0, 2)
        # (lvl*H*W, bs, h_dim)
        bev_pos_encoder_mlvl_embed = bev_pos_encoder_mlvl_embed.permute(
            1, 0, 2)
        # (lvl*H*W, bs, h_dim)
        lidar_feat_mlvl_encoder = self.encoder_lidar(
            query=lidar_feat_mlvl,
            key=None,
            value=None,
            query_pos=bev_pos_encoder_mlvl_embed,
            spatial_shapes=bev_spatial_shape_mlvl_tensor,
            reference_points=bev_pos_encoder_mlvl_norm,
            level_start_index=bev_level_start_index
        )
        # (lvl*H*W, bs, h_dim)

        # bring back the shape of feature maps
        lidar_feat_mlvl_encoder_list = lidar_feat_mlvl_encoder.split(
            [H_ * W_ for H_, W_ in bev_spatial_shape_mlvl],
            dim=0)
        # [(H*W, bs, h_dim), (H*W, bs, h_dim), ...]
        lidar_feats_enc = []
        for level, (H_, W_) in enumerate(bev_spatial_shape_mlvl):
            memory_point_fmap = lidar_feat_mlvl_encoder_list[
                level].permute(
                1, 2, 0).reshape(lidar_feat_bs, lidar_feat_dim, H_, W_)
            lidar_feats_enc.append(memory_point_fmap)
            # this contains list [(bs, c, h, w), ... for levels]

        return lidar_feats_enc

    @force_fp32(apply_to='outputs')
    def loss(self, outputs, batch_gt_instances):
        """
        This is the loss for DDet3D

        Args:
            outputs (dict): {'pred_logits':(bs, n_p, #cls),
                             'pred_boxes':(bs, n_p, 10),
                             'aux_outputs': [{'pred_logits':(bs, n_p, #cls),
                                              'pred_boxes':(bs, n_p, 10)},
                                               { }, { }, ...]}
                    pred boxes absolute [cx, cy, cz, w, l, h, sin, cos, vx, vy]
            batch_gt_instances (list[gt_instance]): bboxes (n_gt, 9),
                                                    labels (n_gt, ),

        Returns:
            (dict) A dict of loss with 'loss_cls', 'loss_bbox', 'd0.loss_cls',
            'd0.loss_bbox', ...
        """
        # convert outputs to the format this loss expects
        all_sem_cls_logits = [outputs['pred_logits']]
        all_bbox_pred = [outputs['pred_boxes']]
        for aux_output in outputs['aux_outputs']:
            all_sem_cls_logits.append(aux_output['pred_logits'])
            all_bbox_pred.append(aux_output['pred_boxes'])
        all_sem_cls_logits = torch.stack(all_sem_cls_logits)
        # (#lay, bs, n_p, #cls)
        all_bbox_pred = torch.stack(all_bbox_pred)  # (#lay, bs, n_p, 10)

        # convert ground truth to this format this loss expects
        gt_bboxes_list = []
        gt_labels_list = []
        for gt_instance in batch_gt_instances:
            gt_bboxes_list.append(gt_instance.bboxes)
            gt_labels_list.append(gt_instance.labels)

        num_dec_layers = len(all_sem_cls_logits)
        device = gt_labels_list[0].device

        # this is already done in prepare_training_targets
        # gt_bboxes_list = [torch.cat(
        #     (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
        #     dim=1).to(device) for gt_bboxes in gt_bboxes_list]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            all_sem_cls_logits,
            all_bbox_pred,
            all_gt_bboxes_list,
            all_gt_labels_list)

        loss_dict = dict()

        # loss from last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for (loss_cls_i, loss_bbox_i) in zip(
                losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    def loss_single(self,
                    sem_cls_logits,
                    bbox_pred,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """Loss function for each decoder layer
            Args:
                sem_cls_logits (Tensor): class logits (B, n_q, #cls)
                bbox_pred (Tensor): Bboxes  (B, n_q, 10)
                gt_bboxes_list (list[Tensor]): Ground truth bboxes for each
                    image with shape (num_gts, 7) in [cx, cy, cz, l, w, h,
                    theta] format. LiDARInstance3DBoxes
                gt_labels_list (list[Tensor]): Ground truth class indices
                    for each image with shape (num_gts, ).
                gt_bboxes_ignore_list (list[Tensor], optional): Bounding boxes
                    which can be ignored for each image. Default None.
            Returns:
                (tuple): loss_cls, loss_bbox
        """

        num_imgs = sem_cls_logits.size(0)

        # prepare scores and bboxes list for all images to get targets
        sem_cls_logits_list = [sem_cls_logits[i] for i in range(num_imgs)]
        # [(n_q, #cls+1), ... #images]
        bbox_pred_list = [bbox_pred[i] for i in range(num_imgs)]
        # [(n_q, 10), ... #images]

        cls_reg_targets = self.get_targets(sem_cls_logits_list,
                                           bbox_pred_list,
                                           gt_bboxes_list,
                                           gt_labels_list,
                                           gt_bboxes_ignore_list)

        (labels_list, label_weights_list,
         bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)  # (bs*num_q, )
        label_weights = torch.cat(label_weights_list, 0)  # (bs*num_q, )
        bbox_targets = torch.cat(bbox_targets_list, 0)  # (bs * num_q, 8)
        bbox_weights = torch.cat(bbox_weights_list, 0)  # (bs * num_q, 8)

        # classification loss
        pred_logits = sem_cls_logits.reshape(-1, self.cls_out_channels)
        # (bs * num_q, #cls)
        # construct weighted avg_factor
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                pred_logits.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        loss_cls = self.loss_cls(
            pred_logits, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_pred.reshape(-1, bbox_pred.size(-1))
        # (bs * num_q, 10)
        normalized_bbox_targets = normalize_bbox(bbox_targets, None)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :len(self.code_weights)],
            normalized_bbox_targets[isnotnan, :len(self.code_weights)],
            bbox_weights[isnotnan, :len(self.code_weights)],
            avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)

        return loss_cls, loss_bbox

    def get_targets(self,
                    sem_cls_logits_list,
                    bbox_pred_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """
        Compute Classification and Regression targets for all batch elements.

        Args:
            sem_cls_logits_list (list[Tensor]): Box score logits for each
                batch element (n_q, #cls)
            bbox_pred_list (list[Tensor]): Bbox predictions for each
                batch element (n_q, 10)
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 7) in [cx, cy, cz, l, w, h, theta] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.

        Returns:
            tuple: a tuple containing the following targets.

                - labels_list (list[Tensor]): Labels for all batch elements of
                    shape (n_q, ).
                - label_weights_list (list[Tensor]): Label weights for all
                    batch elements of shape (n_q, ).
                - bbox_targets_list (list[Tensor]): BBox targets for all
                    batch elements of shape (n_q, 10).
                - bbox_weights_list (list[Tensor]): BBox weights for all
                    batch elements of shape (n_q, 10).
                - num_total_pos (int): Number of positive samples in all \
                    batch elements.
                - num_total_neg (int): Number of negative samples in all \
                    batch elements.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'

        num_imgs = len(sem_cls_logits_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list,
         bbox_targets_list, bbox_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single,
            sem_cls_logits_list,
            bbox_pred_list,
            gt_bboxes_list, gt_labels_list,
            gt_bboxes_ignore_list)
        # multi_apply retunrs tuple of lists
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list,
                num_total_pos, num_total_neg)

    def _get_target_single(self,
                           sem_cls_logits,
                           bbox_pred,
                           gt_bboxes, gt_labels,
                           gt_bboxes_ignore=None):
        """Compute regression and classification targets for one
            image.
        Args:
            sem_cls_logits (Tensor): Box score logits for each
                batch element (n_q, #cls)
            bbox_pred (Tensor): Bbox predictions for each
                batch element (n_q, 10)
            gt_bboxes (Tensor): Ground truth bboxes for one batch with
                shape (num_gts, 7) in [cx, cy, cz, l, w, h, theta] format.
            gt_labels (Tensor): Ground truth class indices for one batch
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.

        Returns:
            tuple[Tensor]: a tuple containing the following for one image.

                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - dir_targets (Tensor): Direction targets for each image.
                - dir_weights (Tensor): Direction weights for each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_bboxes = bbox_pred.size(0)

        assign_result = self.assigner.assign(bbox_pred,
                                             sem_cls_logits.reshape(-1,
                                                                    self.cls_out_channels),
                                             gt_bboxes,
                                             gt_labels,
                                             gt_bboxes_ignore)
        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        pos_inds = sampling_result.pos_inds  # (num_pos, ) number of
        # predicted bounding boxes with matched ground truth box, the index
        # of those matched predicted bounding boxes
        neg_inds = sampling_result.neg_inds  # (num_neg, ) indices of
        # negative predicted bounding boxes

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),  # (num_q, ) filled w/ cls
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        # here only the matched positive boxes are assigned with labels of
        # matched boxes from ground truth
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        # Here, we assign predicted boxes to gt boxes and therefore we want
        # labels and bbox_targets both in predicted box shape but with gt
        # labels and boxes in it!!!
        bbox_targets = torch.zeros_like(bbox_pred)[...,
                       :len(self.code_weights) - 1]
        # because bbox_pred is 10 values but targets is 9 values
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (labels, label_weights,
                bbox_targets, bbox_weights,
                pos_inds, neg_inds)
        # labels (num_q, )
        # label_weights (num_q, )
        # bbox_targets (num_q, 7)
        # bbox_weights (num_q, 7)
        # pos_inds (num_q, )
        # neg_inds (num_q, )

    @force_fp32(apply_to='outputs')
    def loss1(self, outputs, batch_gt_instances):
        """
        This is the loss for DDet3D

        Args:
            outputs (dict): {'pred_logits':(bs, n_p, #cls),
                             'pred_boxes':(bs, n_p, 10),
                             'aux_outputs': [{'pred_logits':(bs, n_p, #cls),
                                              'pred_boxes':(bs, n_p, 10)},
                                               { }, { }, ...]}
                    pred boxes absolute [cx, cy, cz, w, l, h, sin, cos, vx, vy]
            batch_gt_instances (list[gt_instance]): bboxes (n_gt, 9),
                                                    labels (n_gt, ),
        """

        #

        batch_indices = self.assigner(outputs, batch_gt_instances)
        # list[(n_p, )(n_p_gt), ()(), ...bs]
        # fg_mask_inboxes: gives pred indices where it has one matched gt
        # matched_gt_inds: gives the indices of matched gt for each pred box

        # Compute all losses
        loss_cls = self.loss_classification(outputs, batch_gt_instances,
                                            batch_indices)
        loss_bbox = self.loss_boxes(outputs, batch_gt_instances, batch_indices)

        losses = dict(
            loss_cls=loss_cls, loss_bbox=loss_bbox)

        if self.deep_supervision:
            assert 'aux_outputs' in outputs
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                batch_indices = self.assigner(aux_outputs, batch_gt_instances)
                loss_cls = self.loss_classification(aux_outputs,
                                                    batch_gt_instances,
                                                    batch_indices)
                loss_bbox = self.loss_boxes(aux_outputs, batch_gt_instances,
                                            batch_indices)
                tmp_losses = dict(
                    loss_cls=loss_cls,
                    loss_bbox=loss_bbox)
                for name, value in tmp_losses.items():
                    losses[f's.{i}.{name}'] = value
        return losses

    def loss_classification(self, outputs, batch_gt_instances, indices):
        """ Classification Loss
        Args:
            outputs (dict): {'pred_logits':(bs, n_p, #cls),
                             'pred_boxes':(bs, n_p, 10),
                             'aux_outputs': [{'pred_logits':(bs, n_p, #cls),
                                              'pred_boxes':(bs, n_p, 10)},
                                               { }, { }, ...]}
                    pred boxes [cx, cy, cz, w, l, h, sin, cos, vx, vy]
            batch_gt_instances (list[gt_instance]): bboxes (n_gt, 9),
                                                    labels (n_gt, ),
            indices (list[Tensor]): list[(n_p, )(n_p_gt), ()(), ...bs]
                    fg_mask_inboxes: gives pred indices where it has one matched gt
                    matched_gt_inds: gives the indices of matched gt for each pred box
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (bs, n_p, #cls)
        target_classes_list = [
            gt.labels[J] for gt, (_, J) in zip(batch_gt_instances, indices)
        ]  # [(n_p_gt, ),...]
        target_classes = torch.full(
            src_logits.shape[:2],  # (bs, n_p)
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device)
        for idx in range(len(batch_gt_instances)):
            target_classes[idx, indices[idx][0]] = target_classes_list[idx]
            # (bs, n_p)

        src_logits = src_logits.flatten(0, 1)  # (bs*n_p, #cls)
        target_classes = target_classes.flatten(0, 1)  # (bs*n_p, )

        # comp focal loss.
        num_instances = torch.cat(target_classes_list).shape[0]
        if self.sync_cls_avg_factor:
            num_instances = reduce_mean(src_logits.new_tensor([num_instances]))
        num_instances = max(num_instances, 1)
        # bs * n_p_gt
        loss_cls = self.loss_cls(
            src_logits,
            target_classes
        ) / num_instances

        loss_cls = torch.nan_to_num(loss_cls)
        return loss_cls

    def loss_boxes(self, outputs, batch_gt_instances, indices):
        """ BBox Loss
        Args:
            outputs (dict): {'pred_logits':(bs, n_p, #cls),
                             'pred_boxes':(bs, n_p, 10),
                             'aux_outputs': [{'pred_logits':(bs, n_p, #cls),
                                              'pred_boxes':(bs, n_p, 10)},
                                               { }, { }, ...]}
                    pred boxes [cx, cy, cz, w, l, h, sin, cos, vx, vy]
            batch_gt_instances (list[gt_instance]): bboxes (n_gt, 9),
                                                    labels (n_gt, ),
            indices (list[Tensor]): list[(n_p, )(n_p_gt), ()(), ...bs]
                    fg_mask_inboxes: gives pred indices where it has one matched gt
                    matched_gt_inds: gives the indices of matched gt for each pred box
        """
        assert 'pred_boxes' in outputs
        pred_boxes = outputs['pred_boxes']  # (bs, n_p, 10)

        target_bboxes_list = [
            gt.bboxes[J] for gt, (_, J) in zip(batch_gt_instances, indices)
        ]  # [(n_p_gt, 9),...]

        pred_bboxes_list = []
        for idx in range(len(batch_gt_instances)):
            pred_bboxes_list.append(pred_boxes[idx, indices[idx][0]])
            # (n_p_gt, 10)

        pred_boxes_cat = torch.cat(pred_bboxes_list)  # (bs*n_p_gt, 10)
        target_bboxes_cat = torch.cat(target_bboxes_list)  # (bs*n_p_gt, 9)

        if len(pred_boxes_cat) > 0:
            num_instances = pred_boxes_cat.shape[0]
            num_instances = pred_boxes_cat.new_tensor([num_instances])
            num_instances = torch.clamp(reduce_mean(num_instances),
                                        min=1).item()

            # for box weights
            bbox_weights = torch.ones_like(pred_boxes_cat)  # (bs*n_p_gt, 10)
            bbox_weights = bbox_weights * self.code_weights  # (bs*n_p_gt, 10)
            # calculate loss between normalized target boxes and pred boxes
            # so pred boes will give normalized result from forward
            normalized_target_bboxes_cat = normalize_bbox(target_bboxes_cat,
                                                          self.pc_range)
            # (bs*n_p_gt, 10)
            isnotnan = torch.isfinite(normalized_target_bboxes_cat).all(dim=-1)
            loss_bbox = self.loss_bbox(
                pred_boxes_cat[isnotnan, :len(self.code_weights)],
                normalized_target_bboxes_cat[
                isnotnan, :len(self.code_weights)], bbox_weights[
                                                    isnotnan,
                                                    :len(
                                                        self.code_weights)]) / num_instances
        else:
            loss_bbox = pred_boxes.sum() * 0

        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_bbox

    def simple_test_bboxes(self, img_feats, point_feats, img_metas):
        """ Test det bboxes without test-time augmentation.
        Args:
            img_feats (list[Tensor] | None): Image feats list of stride 4, 8,
                16, 32 of shape (bs, n_cam, C, H, W)
            point_feats (list[Tensor]): Point feat list [(B, 128, H, W)...]
                strides 8, 16, 32, 64 of 1472
            img_metas (list[dict]): A list of image info where each dict
                has: 'img_Shape', 'flip' and other details see
                :class `mmdet3d.datasets.pipelines.Collect`.

        Returns:
            list[tuple[LiDARBbox, Tensor, Tensor]]: Each item in result_list is
                3-tuple. The first item is an (n, 9) tensor, where the
                9 columns are bounding box positions
                (cx, cy, cz, l, w, h, theta, vx, vy). The second item is a (n,
                ) tensor where each item is predicted score between 0 and 1.
                The third item is a (n,) tensor where each item is the
                predicted class label of the corresponding box.
        """
        # hard-setting seed to keep results same (if necessary)
        # seed = 0
        # random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)

        if img_feats is not None:
            device = img_feats[-1].device
        elif point_feats is not None:
            device = point_feats[-1].device
        else:
            raise TypeError('{} must be a list, but got {}'.format(
                img_feats, type(img_feats)))

        # prepare the testing targets to get noise boxes
        (time_pairs, batch_noise_bboxes, batch_noise_bboxes_raw) = \
            self.prepare_testing_targets(
                img_metas, device)
        # [(999, -1)] (samp_ts+1, ), (bs, n_p, 10), (bs, n_p, 10)

        results_list = self.predict_by_feat(
            img_feats,
            point_feats,
            time_pairs=time_pairs,
            batch_noise_bboxes=batch_noise_bboxes,
            batch_noise_bboxes_raw=batch_noise_bboxes_raw,
            device=device,
            img_metas=img_metas)
        # [tuple[LiDARBbox, Tensor, Tensor],.... bs]
        # (n_p, 9) (n_p, ), (n_p, )

        return results_list

    def prepare_testing_targets(self, img_metas, device):
        """ Prepare the testing targets
        Args:
            img_metas (list[dict]): A list of image info where each dict
                has: 'img_Shape', 'flip' and other details see
                :class `mmdet3d.datasets.pipelines.Collect`.
            device (str): the name of the device to put the tensors
        """
        # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == timesteps
        times = torch.linspace(
            -1, self.timesteps - 1, steps=self.sampling_timesteps + 1)
        # [-1, 999]  (samp_ts+1, )
        times = list(reversed(times.int().tolist()))  # [999, -1]
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))
        # [(999, -1)](samp_ts+1, )

        noise_bboxes_list = []
        noise_bboxes_raw_list = []
        for _ in range(len(img_metas)):
            noise_bboxes_raw = torch.randn(
                (self.num_proposals, len(self.code_weights)),
                device=device)  # (n_p, 10)
            noise_bboxes = torch.clamp(
                noise_bboxes_raw, min=-1 * self.snr_scale, max=self.snr_scale)
            noise_bboxes = ((noise_bboxes / self.snr_scale) + 1) / 2

            # change any size=0 to min value so that log wont get -inf
            size = noise_bboxes[:, 3:6].clone()  # (n_p, 3)
            size[size == 0.0] = 1e-5
            noise_bboxes[:, 3:6] = size

            noise_bboxes = denormalize_0to1_bbox(noise_bboxes,
                                                 self.pc_range, self.size_norm)
            # (n_p, 10)

            noise_bboxes_raw_list.append(noise_bboxes_raw)
            noise_bboxes_list.append(noise_bboxes)

        batch_noise_bboxes = torch.stack(noise_bboxes_list)  # (bs, n_p, 10)
        batch_noise_bboxes_raw = torch.stack(noise_bboxes_raw_list)
        # (bs, n_p, 10)
        return time_pairs, batch_noise_bboxes, batch_noise_bboxes_raw
        # [(999, -1)] (samp_ts+1, ), (bs, n_p, 10), (bs, n_p, 10)

    def predict_by_feat(self,
                        img_feats,
                        point_feats,
                        time_pairs,
                        batch_noise_bboxes,
                        batch_noise_bboxes_raw,
                        device,
                        img_metas=None,
                        cfg=None):
        """ predcit the bboxes and labels
        Args:
            img_feats (list[Tensor] | None): Image feats list of stride 4, 8,
                16, 32 of shape (bs, n_cam, C, H, W)
            point_feats (list[Tensor]): Point feat list [(B, 128, H, W)...]
                strides 8, 16, 32, 64 of 1472
            time_pairs (list[Tensor]): of shape (samp_ts+1, ) [999, -1]
            batch_noise_bboxes (Tensor): initial noise boxes of shape
                (bs, n_p, 10) in world coords
            batch_noise_bboxes_raw (Tensor): initial noise boxes of shape
                (bs, n_p, 10) in raw randn format
            device (str): name of device
            img_metas (list[dict]): A list of image info where each dict
                has: 'img_Shape', 'flip' and other details see
                :class `mmdet3d.datasets.pipelines.Collect`.
            cfg (dict): test_cfg dict for nms related parameters
        Returns:
            list[tuple[Tensor, Tensor, Tensor]]: Each item in result_list is
            3-tuple. The first item is an (n, 9) tensor, where the
            7 columns are bounding box positions
            (cx, cy, cz, l, w, h, theta, vx, vy). The second item is a (n,
            ) tensor where each item is predicted score between 0 and 1.
            The third item is a (n,) tensor where each item is the
            predicted class label of the corresponding box.
        """
        batch_size = len(img_metas)

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)

        ensemble_score, ensemble_coord = [], []
        for time, time_next in time_pairs:  # 999, -1
            batch_time = torch.full((batch_size,),
                                    time,
                                    device=device,
                                    dtype=torch.long)  # (bs, )
            # self_condition = x_start if self.self_condition else None
            pred_logits, pred_bboxes = self(img_feats, point_feats,
                                            batch_noise_bboxes, batch_time,
                                            img_metas)
            # (#lay, bs, n_p, #cls), (#lay, bs, n_p, 10)
            # pred boxes  center:abs and size:log
            # [cx, cy, cz, w, l, h, sin, cos, vx, vy]

            x_start = pred_bboxes[-1]  # (bs, n_p, 10)
            # we can exp() the size of above boxes and send them t for next
            # sampling step, but instead we use DDIM sampling step and box
            # renewal to improve the model for progressive refinement

            # this completes a sampling step, so
            # if we use onlt one sampling step then we will continue and do
            # inference at last
            # if we use more than one sampling step, then there are two
            # options:
            # one is w/o box renewal and DDIM sampling
            # second is w/ box renewal and sampling
            # For, DDIM sampling, we predict the noise from inital noise
            # boxes and pred boxes; then use DDIM expression to get
            # noise_boxes for next step

            # now to get the bboxes norm between [0, 1], firstly,
            # denormalize the boxes to get exp() of size and then normalize
            # between [0, 1]
            x_start = normalize_0to1_bbox(
                denormalize_bbox(x_start, self.pc_range),
                self.pc_range, self.size_norm)
            # (bs, n_p, 10)  [0, 1]

            x_start = (x_start * 2 - 1.) * self.snr_scale
            x_start = torch.clamp(
                x_start, min=-1 * self.snr_scale, max=self.snr_scale)
            # now get the noise from start using expression
            # eps = (1/sqrt(alpha_cum) * x_t - x_0)/sqrt(1/alpha_cum - 1)
            pred_noise = self.predict_noise_from_start(batch_noise_bboxes_raw,
                                                       batch_time, x_start)
            # (bs, n_p, 10)

            pred_noise_list = []  # This contains pred noise for boxes >th
            x_start_list = []  # contains x_start in [-2, 2] for boxes >th
            noise_bboxes_list = []  # contains org noise boxes in absolute > th
            num_remain_list = []  # contains num or boxes >th
            if self.box_renewal:  # filter
                score_thr = cfg.get('box_score_thr', 0)
                for batch_id in range(batch_size):
                    score_per_image = pred_logits[-1][batch_id]  # (n_p, #cls)

                    score_per_image = torch.sigmoid(score_per_image)
                    value, _ = torch.max(score_per_image, -1, keepdim=False)
                    # (n_p, )
                    keep_idx = value > score_thr  # (n_p, )

                    num_remain_list.append(torch.sum(keep_idx))
                    pred_noise_list.append(pred_noise[batch_id, keep_idx, :])
                    x_start_list.append(x_start[batch_id, keep_idx, :])
                    noise_bboxes_list.append(batch_noise_bboxes[batch_id,
                                             keep_idx, :])
                    # All lists contain (n_p_th, 10)
            else:
                for batch_id in range(batch_size):
                    pred_noise_list.append(pred_noise[batch_id])
                    x_start_list.append(x_start[batch_id])
                    noise_bboxes_list.append(batch_noise_bboxes[batch_id])
                    num_remain_list.append(
                        x_start.new_tensor([pred_noise.shape[1]]))

            # # drop box visualization
            # drop_bboxes_vis = ((x_start_list[0] / self.snr_scale) + 1) / 2
            #
            # # change any size=0 to min value so that log wont get -inf
            # size = drop_bboxes_vis[:, 3:6].clone()  # (n_p, 3)
            # size[size == 0.0] = 1e-5
            # drop_bboxes_vis[:, 3:6] = size
            #
            # drop_bboxes_vis = denormalize_0to1_bbox(drop_bboxes_vis,
            #                                         self.pc_range,
            #                                         self.size_norm)
            # # (bs, n_p, 10)
            # rot_sine = drop_bboxes_vis[..., 6:7]
            #
            # rot_cosine = drop_bboxes_vis[..., 7:8]
            # rot = torch.atan2(rot_sine, rot_cosine)
            #
            # drop_bboxes_vis = torch.cat([drop_bboxes_vis[..., :6], rot], dim=1)
            #
            # drop_bboxes_vis = LiDARInstance3DBoxes(drop_bboxes_vis)
            #
            # drop_bboxes_vis = dict(
            #     boxes_3d=drop_bboxes_vis.to('cpu'),
            #     scores_3d=torch.ones((900,)),
            #     labels_3d=torch.randint(0, 10, (900,))
            # )
            # mmcv.dump([drop_bboxes_vis],
            #           '/mnt/hdd4/achieve-itn/PhD/Code/workdirs/Results'
            #           '/DDet3D/det3d_voxel_hung_nusc_L/pkls'
            #           '/ep29_samp2_drop_125'
            #           '.pkl')

            # the below is executed (in-flow) during sampling_ts is 1 or
            # else when time_next < 0 when sampling_ts is > 1
            if time_next < 0:
                # Not same as original DiffusionDet
                if self.use_ensemble and self.sampling_timesteps > 1:
                    box_pred_list, scores_list = \
                        self.inference(
                            box_cls=pred_logits[-1],
                            box_pred=pred_bboxes[-1],
                            img_metas=img_metas,
                            cfg=cfg,
                            device=device)
                    ensemble_coord.append(box_pred_list)
                    ensemble_score.append(scores_list)
                    # the above contains a list[list[tensor]] where outer list
                    # is for step and inner list is for batch_idx and 'Tensor'
                    # either contains (n_p, 9) or (n_p, #cls)
                continue

            # DDIM Sampling
            alpha = self.alphas_cumprod[time]  # (ts, ) which is 1000
            alpha_next = self.alphas_cumprod[time_next]  # (ts, )

            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) *
                                              (1 - alpha_next) /
                                              (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()  # (ts, )

            batch_noise_bboxes_list = []
            batch_noise_bboxes_raw_list = []
            for idx in range(batch_size):
                pred_noise = pred_noise_list[idx]  # (n_p_th, 10)
                x_start = x_start_list[idx]  # (n_p_th, 10)
                noise_bboxes = noise_bboxes_list[idx]  # (n_p_th, 10)
                num_remain = num_remain_list[idx]  # num
                noise = torch.randn_like(noise_bboxes)  # (n_p_th, 10)

                noise_bboxes = x_start * alpha_next.sqrt() + \
                               c * pred_noise + sigma * noise  # (n_p_th, 10)

                # # to get ddim sampling boxes for visualization
                # ddim_bboxes_vis = torch.clamp(
                #     noise_bboxes,
                #     min=-1 * self.snr_scale,
                #     max=self.snr_scale)
                # ddim_bboxes_vis = ((ddim_bboxes_vis / self.snr_scale) + 1) / 2
                #
                # # change any size=0 to min value so that log wont get -inf
                # size = ddim_bboxes_vis[:, 3:6].clone()  # (n_p, 3)
                # size[size == 0.0] = 1e-3
                # ddim_bboxes_vis[:, 3:6] = size
                #
                # ddim_bboxes_vis = denormalize_0to1_bbox(ddim_bboxes_vis,
                #                                         self.pc_range,
                #                                         self.size_norm)
                # # (bs, n_p, 10)
                # rot_sine = ddim_bboxes_vis[..., 6:7]
                #
                # rot_cosine = ddim_bboxes_vis[..., 7:8]
                # rot = torch.atan2(rot_sine, rot_cosine)
                #
                # ddim_bboxes_vis = torch.cat([ddim_bboxes_vis[..., :6], rot],
                #                             dim=1)
                #
                # ddim_bboxes_vis = LiDARInstance3DBoxes(ddim_bboxes_vis)
                #
                # ddim_bboxes_vis = dict(
                #     boxes_3d=ddim_bboxes_vis.to('cpu'),
                #     scores_3d=torch.rand((900,)),
                #     labels_3d=torch.randint(0, 10, (900,))
                # )
                # mmcv.dump([ddim_bboxes_vis],
                #           '/mnt/hdd4/achieve-itn/PhD/Code/workdirs/Results'
                #           '/DDet3D/det3d_voxel_hung_nusc_L/pkls/ep29_samp2_ddim_125.pkl')

                # if we use box_renewal then concatenate additional boxes
                # to noise_boxes as some boxes are removed which are <th
                if self.box_renewal:  # filter
                    # replenish with randn boxes
                    if num_remain < self.num_proposals:
                        noise_bboxes = torch.cat(
                            (noise_bboxes,
                             torch.randn(
                                 self.num_proposals - num_remain,
                                 len(self.code_weights),
                                 device=device)),
                            dim=0)  # (n_p, 10)
                    else:
                        select_mask = [True] * self.num_proposals + \
                                      [False] * (num_remain -
                                                 self.num_proposals)
                        random.shuffle(select_mask)
                        noise_bboxes = noise_bboxes[select_mask]  # (n_p, 10)

                # raw noise boxes
                batch_noise_bboxes_raw_list.append(noise_bboxes)

                noise_bboxes = torch.clamp(
                    noise_bboxes,
                    min=-1 * self.snr_scale,
                    max=self.snr_scale)
                noise_bboxes = ((noise_bboxes / self.snr_scale) + 1) / 2

                # change any size=0 to min value so that log wont get -inf
                size = noise_bboxes[:, 3:6].clone()  # (n_p, 3)
                size[size == 0.0] = 1e-5
                noise_bboxes[:, 3:6] = size

                noise_bboxes = denormalize_0to1_bbox(noise_bboxes,
                                                     self.pc_range,
                                                     self.size_norm)
                batch_noise_bboxes_list.append(noise_bboxes)

                # # renew box visulaization
                # rot_sine = noise_bboxes[..., 6:7]
                #
                # rot_cosine = noise_bboxes[..., 7:8]
                # rot = torch.atan2(rot_sine, rot_cosine)
                #
                # renew_bboxes_vis = torch.cat([noise_bboxes[..., :6], rot],
                #                              dim=1)
                #
                # renew_bboxes_vis = LiDARInstance3DBoxes(renew_bboxes_vis)
                #
                # renew_bboxes_vis = dict(
                #     boxes_3d=renew_bboxes_vis.to('cpu'),
                #     scores_3d=torch.rand((900,)),
                #     labels_3d=torch.randint(0, 10, (900,))
                # )
                # mmcv.dump([renew_bboxes_vis],
                #           '/mnt/hdd4/achieve-itn/PhD/Code/workdirs/Results'
                #           '/DDet3D/det3d_voxel_hung_nusc_L/pkls/ep29_samp2_renew_125.pkl')
            batch_noise_bboxes = torch.stack(batch_noise_bboxes_list)
            batch_noise_bboxes_raw = torch.stack(batch_noise_bboxes_raw_list)
            # (bs, n_p, 10)

            if self.use_ensemble and self.sampling_timesteps > 1:
                box_pred_list, scores_list = \
                    self.inference(
                        box_cls=pred_logits[-1],
                        box_pred=pred_bboxes[-1],
                        img_metas=img_metas,
                        cfg=cfg,
                        device=device)
                ensemble_coord.append(box_pred_list)
                ensemble_score.append(scores_list)
                # the above contains a list[list[tensor]] where outer list
                # is for step and inner list is for batch_idx and 'Tensor'
                # either contains (n_p, 9) or (n_p, #cls)

        # if ensemble with samp_timesteps then result from all ensemble data
        # with nms
        if self.use_ensemble and self.sampling_timesteps > 1:
            steps = len(ensemble_score)
            results_list = []
            for idx in range(batch_size):
                ensemble_score_per_sample = [
                    ensemble_score[i][idx] for i in range(steps)
                ]  # [(n_p, #cls), (),... steps]
                ensemble_coord_per_sample = [
                    ensemble_coord[i][idx] for i in range(steps)
                ]  # [(n_p, 9), (),... steps]

                if self.use_nms:
                    scores_per_sample = torch.cat(ensemble_score_per_sample,
                                                  dim=0)
                    box_pred_per_sample = torch.cat(ensemble_coord_per_sample,
                                                    dim=0)
                    # (n_p * steps, #cls), (n_p * steps, 9)

                    # get pred boxes for NMS
                    box_pred_per_sample_for_nms = xywhr2xyxyr(img_metas[idx][
                                                                  'box_type_3d'](
                        box_pred_per_sample,
                        box_dim=box_pred_per_sample.shape[
                            -1]).bev)  # (n_p * steps, 5)
                    padding = scores_per_sample.new_zeros(
                        scores_per_sample.shape[0], 1)  # (n_p * steps, 1)
                    # remind that we set FG labels to [0, num_class-1]
                    # BG cat_id: num_class
                    scores_per_sample = torch.cat([scores_per_sample,
                                                   padding], dim=1)
                    # (n_p * steps, #cls + 1)
                    results = box3d_multiclass_nms(box_pred_per_sample,
                                                   box_pred_per_sample_for_nms,
                                                   scores_per_sample,
                                                   cfg.score_thr,
                                                   cfg.max_per_img,
                                                   cfg)
                    box_pred_per_sample, scores_per_sample, \
                        labels_per_sample = results
                    # box_pred (max_num, 9)
                    # scores (max_num, )
                    # labels (max_num, )

                    # check boxes outside of post_center_range
                    post_center_range = torch.tensor(
                        cfg.post_center_range, device=scores_per_sample.device)
                    mask = (box_pred_per_sample[..., :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (box_pred_per_sample[..., :3] <=
                             post_center_range[3:]).all(1)
                    box_pred_per_sample = box_pred_per_sample[mask]
                    scores_per_sample = scores_per_sample[mask]
                    labels_per_sample = labels_per_sample[mask]
                else:
                    labels = torch.arange(self.num_classes,
                                          device=device).unsqueeze(0).repeat(
                        self.num_proposals,
                        1).flatten(0, 1)
                    # (n_p*#cls)
                    scores_per_sample = []
                    box_pred_per_sample = []
                    labels_per_sample = []
                    for i, (scores_per_sample_step,
                            box_pred_per_sample_step) in enumerate(zip(
                        ensemble_score_per_sample, ensemble_coord_per_sample)):
                        # scores_per_sample_step: (n_p, #cls)
                        # box_pred_per_sample_step: (n_p, 9)
                        scores_per_sample_step, topk_indices = scores_per_sample_step.flatten(
                            0, 1).topk(cfg.max_per_img)  # (n_max, )
                        labels_per_sample_step = topk_indices % self.num_classes  # (n_max, )
                        bbox_index_step = topk_indices // self.num_classes
                        box_pred_per_sample_step = box_pred_per_sample_step[
                            bbox_index_step]
                        # (n_max, 9)

                        scores_per_sample.append(scores_per_sample_step)
                        box_pred_per_sample.append(box_pred_per_sample_step)
                        labels_per_sample.append(labels_per_sample_step)
                        # contains (list[Tensor]) for steps ; (n_max, 9) (n_max, )

                    # concatenate all steps
                    scores_per_sample = torch.cat(scores_per_sample, dim=0)
                    box_pred_per_sample = torch.cat(box_pred_per_sample, dim=0)
                    labels_per_sample = torch.cat(labels_per_sample, dim=0)
                    # (n_max * steps, 9) (n_max * steps, ) (n_max * steps, )
                # results = InstanceData()
                # results.bboxes = box_pred_per_sample
                # results.scores = scores_per_sample
                # results.labels = labels_per_sample
                box_pred_per_sample = img_metas[idx]['box_type_3d'](
                    box_pred_per_sample, 9)
                results_list.append([box_pred_per_sample, scores_per_sample,
                                     labels_per_sample])
        else:
            box_cls = pred_logits[-1]
            box_pred = pred_bboxes[-1]
            results_list = self.inference(box_cls, box_pred, img_metas, cfg,
                                          device)
        return results_list
        # list[tuple[Tensor, Tensor, Tensor]]: Each item in result_list is
        # 3-tuple. The first item is an (n, 7) tensor, where the
        # 7 columns are bounding box positions
        # (cx, cy, cz, l, w, h, theta). The second item is a (n,
        # ) tensor where each item is predicted score between 0 and 1.
        # The third item is a (n,) tensor where each item is the
        # predicted class label of the corresponding box.

    def predict_noise_from_start(self, x_t, t, x0):
        """
        x_t: (bs, n_p, 10)  in [0, 1]
        t: (bs, )
        x0: (bs, n_p, 10)  in  [-snr_scale, snr_scale]
        """
        # eps = (1/sqrt(alpha_cum) * x_t - x_0)/sqrt(1/alpha_cum - 1)
        results = (extract(
            self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                  extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return results  # (bs, n_p, 10)

    @force_fp32(apply_to=('box_cls', 'box_pred'))
    def inference(self, box_cls, box_pred, img_metas, cfg, device):
        """
        Args:
            box_cls (Tensor): tensor of shape (bs, n_p, #cls).
                The tensor predicts the classification probability for
                each proposal.
            box_pred (Tensor): tensors of shape (bs, n_p, 10).
                            center:abs and size:log
                            [cx, cy, cz, w, l, h, sin, cos, vx, vy]
            img_metas (list[dict])

        Returns:
            list[tuple[Tensor, Tensor, Tensor]]: Each item in result_list is
                3-tuple. The first item is an (n, 9) tensor, where the
                9 columns are bounding box positions
                (cx, cy, cz, l, w, h, theta, vx, vy). The second item is a (n,
                ) tensor where each item is predicted score between 0 and 1.
                The third item is a (n,) tensor where each item is the
                predicted class label of the corresponding box.
        """
        results = []

        if self.use_focal_loss or self.use_fed_loss:
            scores = torch.sigmoid(box_cls)  # (bs, n_p, #cls)
            labels = torch.arange(self.num_classes,
                                  device=device).unsqueeze(0).repeat(
                self.num_proposals,
                1).flatten(0, 1)
            # (n_p*#cls)

            box_pred_list = []
            scores_list = []
            # labels_list = []
            # for each sample
            for i, (scores_per_sample,
                    box_pred_per_sample) in enumerate(zip(scores, box_pred)):
                # scores_per_sample: (n_p, #cls)
                # box_pred_per_sample: (n_p, 10) ; center:abs and size:log

                # decode the pred boxes: to convert size to exp() and sincos
                # to ry
                box_pred_per_sample = denormalize_bbox(box_pred_per_sample,
                                                       self.pc_range)
                # (n_p, 9) all absolute now
                # convert gravity center to bottom center
                box_pred_per_sample[:, 2] = box_pred_per_sample[:,
                                            2] - box_pred_per_sample[:,
                                                 5] * 0.5
                # (n_p, 9)

                if self.use_ensemble and self.sampling_timesteps > 1:
                    box_pred_list.append(box_pred_per_sample)  # (n_p, 9)
                    scores_list.append(scores_per_sample)  # (n_p, #cls)
                    continue

                if self.use_nms:
                    # get pred boxes for NMS
                    box_pred_per_sample_for_nms = xywhr2xyxyr(img_metas[i][
                                                                  'box_type_3d'](
                        box_pred_per_sample,
                        box_dim=box_pred_per_sample.shape[
                            -1]).bev)  # (n_p, 5)
                    padding = scores_per_sample.new_zeros(
                        scores_per_sample.shape[0], 1)  # (n_p, 1)
                    # remind that we set FG labels to [0, num_class-1]
                    # BG cat_id: num_class
                    scores_per_sample = torch.cat([scores_per_sample,
                                                   padding], dim=1)
                    # (n_p, #cls + 1)
                    results_nms = box3d_multiclass_nms(box_pred_per_sample,
                                                       box_pred_per_sample_for_nms,
                                                       scores_per_sample,
                                                       cfg.score_thr,
                                                       cfg.max_per_img,
                                                       cfg)
                    box_pred_per_sample, scores_per_sample, \
                        labels_per_sample = results_nms
                    # box_pred (max_num, 9)
                    # scores (max_num, )
                    # labels (max_num, )

                    # check boxes outside of post_center_range
                    post_center_range = torch.tensor(
                        cfg.post_center_range, device=scores_per_sample.device)
                    mask = (box_pred_per_sample[..., :3] >=
                            post_center_range[:3]).all(1)
                    mask &= (box_pred_per_sample[..., :3] <=
                             post_center_range[3:]).all(1)
                    box_pred_per_sample = box_pred_per_sample[mask]
                    scores_per_sample = scores_per_sample[mask]
                    labels_per_sample = labels_per_sample[mask]
                else:
                    scores_per_sample, topk_indices = scores_per_sample.flatten(
                        0, 1).topk(cfg.max_per_img)
                    labels_per_sample = topk_indices % self.num_classes
                    bbox_index = topk_indices // self.num_classes
                    box_pred_per_sample = box_pred_per_sample[bbox_index]
                    # (n_p, 9)
                # result = InstanceData()
                # result.bboxes = box_pred_per_sample
                # result.scores = scores_per_sample
                # result.labels = labels_per_sample
                # convert boxes to LiDARInstanceBbox
                box_pred_per_sample = img_metas[i]['box_type_3d'](
                    box_pred_per_sample, box_pred_per_sample.shape[-1])
                results.append([box_pred_per_sample, scores_per_sample,
                                labels_per_sample])

        else:
            raise NotImplementedError

        if self.use_ensemble and self.sampling_timesteps > 1:
            return box_pred_list, scores_list
        else:
            return results


@HEADS.register_module()
class SingleDDet3DHead(BaseModule):
    """
    This is the Single Diffusion Det 3D Head
    """

    def __init__(self,
                 num_classes=80,
                 feat_channels=256,
                 pooler_resolution=7,
                 use_focal_loss=True,
                 use_fed_loss=False,
                 dim_feedforward=2048,
                 num_cls_convs=1,
                 num_reg_convs=3,
                 num_heads=8,
                 dropout=0.0,
                 scale_clamp=_DEFAULT_SCALE_CLAMP,
                 bbox_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2,
                               0.2],
                 act_cfg=dict(type='ReLU', inplace=True),
                 dynamic_conv=dict(dynamic_dim=64, dynamic_num=2),
                 pc_range=None,
                 use_fusion=False,
                 voxel_size=None,
                 init_cfg=None):
        super(SingleDDet3DHead, self).__init__(init_cfg)

        self.feat_channels = feat_channels
        self.pc_range = pc_range
        self.use_fusion = use_fusion
        self.voxel_size = voxel_size

        # Dynamic
        self.self_attn = nn.MultiheadAttention(
            feat_channels, num_heads, dropout=dropout)
        self.inst_interact = DynamicConv(
            feat_channels=feat_channels,
            pooler_resolution=pooler_resolution,
            dynamic_dim=dynamic_conv['dynamic_dim'],
            dynamic_num=dynamic_conv['dynamic_num'])

        self.linear1 = nn.Linear(feat_channels, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, feat_channels)

        self.norm1 = nn.LayerNorm(feat_channels)
        self.norm2 = nn.LayerNorm(feat_channels)
        self.norm3 = nn.LayerNorm(feat_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = build_activation_layer(act_cfg)

        # block time mlp
        self.block_time_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(feat_channels * 4, feat_channels * 2))

        # cls.
        cls_module = list()
        for _ in range(num_cls_convs):
            cls_module.append(nn.Linear(feat_channels, feat_channels, False))
            cls_module.append(nn.LayerNorm(feat_channels))
            cls_module.append(nn.ReLU(inplace=True))
        self.cls_module = ModuleList(cls_module)

        # reg.
        reg_module = list()
        for _ in range(num_reg_convs):
            reg_module.append(nn.Linear(feat_channels, feat_channels, False))
            reg_module.append(nn.LayerNorm(feat_channels))
            reg_module.append(nn.ReLU(inplace=True))
        self.reg_module = ModuleList(reg_module)

        # pred.
        self.use_focal_loss = use_focal_loss
        self.use_fed_loss = use_fed_loss
        if self.use_focal_loss or self.use_fed_loss:
            self.class_logits = nn.Linear(feat_channels, num_classes)
        else:
            self.class_logits = nn.Linear(feat_channels, num_classes + 1)
        self.bboxes_delta = nn.Linear(feat_channels, len(bbox_weights))
        self.scale_clamp = scale_clamp
        self.bbox_weights = bbox_weights

        if self.use_fusion:
            self.output_fused_proj = nn.Linear(2 * feat_channels,
                                               feat_channels)
        # for srcn3d
        # if init_cfg is None:
        #     self.init_cfg = [
        #         dict(
        #             type='Normal', std=0.01, override=dict(name='class_logits')),
        #         dict(
        #             type='Normal', std=0.001, override=dict(name='bboxes_delta'))
        #     ]

    # def init_weights(self) -> None:  # yes for srcn3d
    #     super(SingleDDet3DHead, self).init_weights()
    #     for p in self.parameters():
    #         if p.dim() > 1:
    #             nn.init.xavier_uniform_(p)
    #         else:
    #             # adopt the default initialization for
    #             # the weight and bias of the layer norm
    #             pass
    #
    #     bias_value = -math.log((1 - 0.01) / 0.01)
    #     nn.init.constant_(self.class_logits.bias, bias_value)
    #     nn.init.constant_(self.bboxes_delta.bias.data[2:], 0.0)

    def forward(self, img_feats, point_feats, bboxes, prop_feats, pooler,
                time_emb, img_metas, pooler_img=None):
        """
        img_feats (list[Tensor]): shape (bs, n_cam, C, H, W)
        point_feats (list[Tensor]): shape (bs, C, H, W)
        bboxes (Tensor): (bs, n_p, 10)
        prop_feats (Tensor|None)
        pooler (nn.Module)
        time_emb (Tensor): (bs, 256*4)
        img_metas (list[dict])
        """

        bs, n_p = bboxes.shape[:2]

        if img_feats is not None:
            img_roi_feats = self.img_feats_sampling_bboxes_roi(img_feats,
                                                               bboxes,
                                                               pooler_img,
                                                               img_metas)
            # (bs*n_p, C, 7, 7)
        else:
            img_roi_feats = None

        if point_feats is not None:
            points_roi_feats = self.points_feats_sampling_bboxes_roi(
                point_feats,
                bboxes,
                pooler,
                img_metas)
            # (bs*n_p, C, 7, 7)
        else:
            points_roi_feats = None

        # for feature fusion
        if img_roi_feats is not None and points_roi_feats is not None and \
                self.use_fusion:
            feats_fused = torch.cat((img_roi_feats, points_roi_feats), dim=1)
            feats_fused = feats_fused.permute(0, 2, 3, 1)
            # (bs*n_p, C, 7, 7) --> (bs*n_p, 7, 7, 2C)
            feats_fused = self.output_fused_proj(feats_fused)
            # (bs*n_p, 7, 7, C)
            feats_fused = feats_fused.permute(0, 3, 1, 2)
            # (bs*n_p, 7, 7, C) --> (bs*n_p, C, 7, 7)
            roi_feats = feats_fused
        elif not self.use_fusion and img_roi_feats is not None and \
                points_roi_feats is None:
            roi_feats = img_roi_feats  # (bs*n_p, C, 7, 7)
        elif not self.use_fusion and points_roi_feats is not None and \
                img_roi_feats is None:
            roi_feats = points_roi_feats  # (bs*n_p, C, 7, 7)

        if prop_feats is None:
            prop_feats = roi_feats.view(bs, n_p, self.feat_channels,
                                        -1).mean(-1)
            # (bs, n_p, 256, 7*7) --> (bs, n_p, 256)

        roi_feats = roi_feats.view(bs * n_p, self.feat_channels,
                                   -1).permute(2, 0, 1)
        # (bs*n_p, 256, 7*7) --> (7*7, bs*n_p, 256)

        # self_attention
        prop_feats = prop_feats.view(bs, n_p, self.feat_channels).permute(1, 0,
                                                                          2)
        # (bs, n_p, 256) --> (n_p, bs, 256)
        prop_feats2 = self.self_attn(prop_feats, prop_feats,
                                     value=prop_feats)[0]
        prop_feats = prop_feats + self.dropout1(prop_feats2)
        prop_feats = self.norm1(prop_feats)  # (n_p, bs, 256)

        # inst_interact.
        prop_feats = prop_feats.view(n_p, bs, self.feat_channels).permute(
            1, 0, 2).reshape(1, bs * n_p, self.feat_channels)
        # (n_p, bs, 256) --> (bs, n_p, 256)  -> (1, bs * n_p, 256)
        # roi feats of shape (7*7, bs*n_p, 256)
        prop_feats2 = self.inst_interact(prop_feats, roi_feats)
        # (bs*n_p, 256)
        prop_feats = prop_feats + self.dropout2(prop_feats2)
        obj_feats = self.norm2(prop_feats)  # (bs*n_p, 256)

        # FFN
        obj_feats2 = self.linear2(
            self.dropout(self.activation(self.linear1(obj_feats))))
        obj_feats = obj_feats + self.dropout3(obj_feats2)
        obj_feats = self.norm3(obj_feats)  # (bs*n_p, 256)

        # fc_feature = obj_feats
        fc_feature = obj_feats.transpose(0, 1).reshape(bs * n_p, -1)
        # (bs*n_p, 256)

        scale_shift = self.block_time_mlp(time_emb)  # (bs, 512)
        scale_shift = torch.repeat_interleave(scale_shift, n_p, dim=0)
        #  (bs*n_p, 512)
        scale, shift = scale_shift.chunk(2, dim=1)
        # (bs*n_p, 256), (bs*n_p, 256)
        fc_feature = fc_feature * (scale + 1) + shift  # (bs*n_p, 256)

        cls_feature = fc_feature.clone()
        reg_feature = fc_feature.clone()
        for cls_layer in self.cls_module:
            cls_feature = cls_layer(cls_feature)  # (bs*n_p, 256)
        for reg_layer in self.reg_module:
            reg_feature = reg_layer(reg_feature)  # (bs*n_p, 256)
        class_logits = self.class_logits(cls_feature)  # (bs*n_p, #cls)
        bboxes_deltas = self.bboxes_delta(reg_feature)  # (bs*n_p, 10)
        pred_bboxes = self.apply_deltas(bboxes_deltas, bboxes.view(-1,
                                                                   len(
                                                                       self.bbox_weights)))
        # (bs*n_p, 10)
        # pred boxes  center:abs and size:log
        # [cx, cy, cz, w, l, h, sin, cos, vx, vy]

        return (class_logits.view(bs, n_p, -1), pred_bboxes.view(bs, n_p,
                                                                 -1),
                obj_feats)
        # (bs, n_p, #cls), (bs, n_p, 10), (bs*n_p, 256)
        # pred boxes  center:abs and size:log
        # [cx, cy, cz, w, l, h, sin, cos, vx, vy]

    def apply_deltas(self, deltas, boxes):
        """Apply transformation `deltas` (dx, dy, dz, dw, dl, dh) to `boxes`.

        Args:
            deltas (Tensor): transformation deltas of shape (N, k*10),
                where k >= 1. deltas[i] represents k potentially
                different class-specific box transformations for
                the single box boxes[i].
            boxes (Tensor): boxes to transform, of shape (N, 10)
                [cx, cy, cz, w, l, h, sin, cos, vx, vy]
        """
        boxes = boxes.to(deltas.dtype)
        deltas_split = torch.split(deltas, 1, dim=-1)
        # ((bs, n_p, 1), ... 10)
        boxes_split = torch.split(boxes, 1, dim=-1)
        # ((bs, n_p, 1), ... 10)
        if len(self.bbox_weights) == 10:
            wx, wy, wz, ww, wl, wh, _, _, _, _ = self.bbox_weights
        else:
            wx, wy, wz, ww, wl, wh, _, _ = self.bbox_weights

        dx = deltas_split[0] / wx
        dy = deltas_split[1] / wy
        dz = deltas_split[2] / wz
        dw = deltas_split[3] / ww
        dl = deltas_split[4] / wl
        dh = deltas_split[5] / wh

        ctr_x = boxes_split[0]
        ctr_y = boxes_split[1]
        ctr_z = boxes_split[2]
        # ctr_x = boxes_split[0] * (self.pc_range[3] - self.pc_range[0]) + \
        #     self.pc_range[0]
        # ctr_y = boxes_split[1] * (self.pc_range[4] - self.pc_range[1]) + \
        #         self.pc_range[1]
        # ctr_z = boxes_split[2] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

        widths = torch.exp(boxes_split[3])
        lengths = torch.exp(boxes_split[4])
        heights = torch.exp(boxes_split[5])
        # because log is applied at end so to reverse it
        # widths = boxes_split[3]
        # lengths = boxes_split[4]
        # heights = boxes_split[5]

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=self.scale_clamp)
        dl = torch.clamp(dl, max=self.scale_clamp)
        dh = torch.clamp(dh, max=self.scale_clamp)

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * lengths + ctr_y
        pred_ctr_z = dz * heights + ctr_z

        pred_w = torch.exp(dw) * widths
        pred_l = torch.exp(dl) * lengths
        pred_h = torch.exp(dh) * heights  # (bs*n_p, 1)

        # apply sigmoid on center and then back to world system
        # pred_ctr_x = pred_ctr_x.sigmoid()
        # pred_ctr_y = pred_ctr_y.sigmoid()
        # pred_ctr_z = pred_ctr_z.sigmoid()

        # pred_ctr_x = (pred_ctr_x - self.pc_range[0]) / (self.pc_range[3] -
        #                                                 self.pc_range[0])
        # pred_ctr_y = (pred_ctr_y - self.pc_range[1]) / (self.pc_range[4] -
        #                                                 self.pc_range[1])
        # pred_ctr_z = (pred_ctr_z - self.pc_range[2]) / (self.pc_range[5] -
        #                                                 self.pc_range[2])
        # pred_ctr_x = torch.clamp(pred_ctr_x, max=1.0, min=0.0)
        # pred_ctr_y = torch.clamp(pred_ctr_y, max=1.0, min=0.0)
        # pred_ctr_z = torch.clamp(pred_ctr_z, max=1.0, min=0.0)
        # pred_ctr_x = (pred_ctr_x * (self.pc_range[3] - self.pc_range[0])) + \
        #              self.pc_range[0]
        # pred_ctr_y = (pred_ctr_y * (self.pc_range[4] - self.pc_range[1])) + \
        #              self.pc_range[1]
        # pred_ctr_z = (pred_ctr_z * (self.pc_range[5] - self.pc_range[2])) + \
        #              self.pc_range[2]

        if len(self.bbox_weights) == 10:
            pred_boxes = torch.cat(
                [pred_ctr_x, pred_ctr_y, pred_ctr_z, pred_w.log(),
                 pred_l.log(),
                 pred_h.log(), deltas_split[6], deltas_split[7],
                 deltas_split[8],
                 deltas_split[9]], dim=-1)  # (bs*n_p, 10)
        else:
            pred_boxes = torch.cat(
                [pred_ctr_x, pred_ctr_y, pred_ctr_z, pred_w.log(),
                 pred_l.log(),
                 pred_h.log(), deltas_split[6], deltas_split[7]],
                dim=-1)  # (bs*n_p, 10)
        # pred_boxes = torch.cat(
        #     [pred_ctr_x, pred_ctr_y, pred_ctr_z, pred_w, pred_l,
        #      pred_h, deltas_split[6], deltas_split[7], deltas_split[8],
        #      deltas_split[9]], dim=-1)  # (bs*n_p, 10)

        return pred_boxes  # (bs*n_p, 10)
        # pred boxes  center:abs and size:log
        # [cx, cy, cz, w, l, h, sin, cos, vx, vy]

    def img_feats_sampling_bboxes_roi(self, img_feats, bboxes, pooler,
                                      img_metas):
        """
        This function samples the image features for the bboxes using pooler
        img_feats (list[Tensor]): shape (bs, n_cam, C, H, W)
        bboxes (Tensor): (bs, n_p, 10) [cx, cy, cz, w, l, h, sin, cos, vx, vy]
        pooler (nn.Module): ROI Extractor
        img_metas (list[dict])
        pc_range (list): [x_min, y_,in, z_min, x_max, y_max, z_max]
        """

        # desigmoid the center and denormalize the size
        # pc_range_ = bboxes.new_tensor([[self.pc_range[3] - self.pc_range[0],
        #                                 self.pc_range[4] - self.pc_range[1],
        #                                 self.pc_range[5] - self.pc_range[
        #                                     2]]])  # (1, 3)
        # pc_start_ = bboxes.new_tensor(
        #     [[self.pc_range[0], self.pc_range[1], self.pc_range[2]]])  # (1, 3)
        # bboxes[..., :3] = (bboxes[..., :3] * pc_range_) + pc_start_  # (n_p, 3)

        # bboxes[..., 3:6] = bboxes[..., 3:6].exp()

        # get the corners of the bboxes
        bbox_corners = boxes3d_to_corners3d(bboxes[..., :8],
                                            bottom_center=False, ry=False)
        # (bs, n_p, 8, 3) in world coord

        # project the corners in LiDAR coord system to six cameras
        lidar2img = []
        for img_meta in img_metas:
            lidar2img.append(img_meta['lidar2img'])
        lidar2img = np.asarray(lidar2img)
        lidar2img = bboxes.new_tensor(lidar2img)  # (bs, num_cam, 4, 4)
        bbox_corners = torch.cat((bbox_corners,
                                  torch.ones_like(
                                      bbox_corners[..., :1])), -1)
        # (bs, n_p, 8, 4)
        bs, num_prop = bbox_corners.size()[:2]
        num_cam = lidar2img.size(1)
        bbox_corners = bbox_corners.view(bs, 1, num_prop, 8, 4).repeat(1,
                                                                       num_cam,
                                                                       1,
                                                                       1,
                                                                       1).unsqueeze(
            -1)
        # (bs, n_p, 8, 4) --> (bs, 1, n_p, 8, 4) --> (bs, n_cam, n_p, 8, 4) -->
        # (bs, n_cam, n_p, 8, 4, 1)
        lidar2img = lidar2img.view(bs, num_cam, 1, 1, 4, 4).repeat(1, 1,
                                                                   num_prop,
                                                                   8, 1, 1)
        # (bs, num_cam, 4, 4) --> (bs, n_cam, 1, 1, 4, 4) --> (bs, n_cam, n_p,
        # 8, 4, 4)

        bbox_cam = torch.matmul(lidar2img, bbox_corners).squeeze(-1)
        # (bs, num_cam, num_proposals, 8, 4)
        # (bs, n_cam, n_p, 8, 4, 4) * (bs, n_cam, n_p, 8, 4, 1) =
        # (bs, n_cam, n_p, 8, 4, 1) --> (bs, n_cam, n_p, 8, 4)

        # normalize real-world points back to normalized [-1,-1,1,1]
        # image coordinate
        eps = 1e-5
        bbox_cam = bbox_cam[..., 0:2] / torch.maximum(bbox_cam[..., 2:3],
                                                      torch.ones_like(
                                                          bbox_cam[...,
                                                          2:3]) * eps)  # ?
        # (bs, n_cam, n_p, 8, 2)

        box_corners_in_image = bbox_cam

        # expect box_corners_in_image: [B,N, 8, 2] -- [B,num_cam,N,8,2]
        minxy = torch.min(box_corners_in_image, dim=3).values
        # (bs, n_cam, n_p, 2)
        maxxy = torch.max(box_corners_in_image, dim=3).values
        # (bs, n_cam, n_p, 2)

        bbox2d = torch.cat([minxy, maxxy], dim=3).permute(0, 2, 1, 3)
        # (bs, n_cam, n_p, 4) --> (bs, n_p, n_cam, 4)

        # convert bbox2d to ROI for all cameras
        sampled_rois = None
        for cam_id in range(num_cam):
            bs = img_feats[0].shape[0]
            C = img_feats[0].shape[2]

            bbox2d_percam = bbox2d[:, :, cam_id, :].reshape(bs, num_prop, 4)
            # (bs, n_p, 4)

            bbox2d_percam_list = torch.split(bbox2d_percam, 1)
            # ((1, n_p, 4), (1, n_p, 4),...bs)
            bbox2d_percam_list = [lvl[0, :, :] for lvl in bbox2d_percam_list]
            # [(n_p, 4), (n_p, 4), ... bs]

            if sampled_rois is None:
                temp_roi = bbox2roi(bbox2d_percam_list)  # (bs*n_p, 5) batch_id
                temp_roi[:, 0] = temp_roi[:, 0] + cam_id * bs
                # batch and cam ids
                sampled_rois = temp_roi
            else:
                temp_roi = bbox2roi(bbox2d_percam_list)
                temp_roi[:, 0] = temp_roi[:, 0] + cam_id * bs
                sampled_rois = torch.cat([sampled_rois, temp_roi], dim=0)
                # (bs*n_p*n_cam, 5)

                # here for bs=3, n_p=4 and n_cam=6, the temp_roi[:, 0] is...
                # cam1: 0000,1111,2222
                # cam2: 3333,4444,5555
                # cam3: 6666,7777,8888
                # cam4: 9999, 10101010, 11111111 and so on

                # img_feat_lvl is (bs, n_cam, C, H, W)
                # but extarctor expects it to be (bs, C, H, W)
                # so lets permute img_feat_lvl to (bs*n_cam, C, H, W)
                # here the 1st dimenion values would be
                # 0, 1, 2, 3, 4, 5, ... 18

        # mlvl_feats_cam = [feat[0, :, :, :, :] for feat in img_feats]
        mlvl_feats_cam = []
        for feat in img_feats:
            bs, n_cam, C, H, W = feat.shape
            mlvl_feats_cam.append(feat.reshape(bs * n_cam, C, H, W))
        sampled_feats = pooler(mlvl_feats_cam[:pooler.num_inputs],
                               sampled_rois)
        # (num_cam * num_prop, C, 7, 7)
        sampled_feats = sampled_feats.view(num_cam, bs, num_prop, C, 7, 7)
        # (n_cam, bs, n_p, C, 7, 7)
        sampled_feats = sampled_feats.permute(1, 0, 2, 3, 4, 5)
        # (bs, n_cam, n_p, C, 7, 7)
        sampled_feats = sampled_feats.permute(0, 2, 3, 1, 4, 5)
        # (bs, n_cam, n_p, C, 7, 7) --> (bs, n_p, C, n_cam, 7, 7)
        sampled_feats = sampled_feats.reshape(bs, num_prop, C, num_cam, 7, 7)
        sampled_feats = sampled_feats.permute(0, 1, 2, 4, 5, 3)
        # (bs, n_p, C, n_cam, 7, 7) --> (bs, n_p, C, 7, 7, n_cam)

        sampled_feats = sampled_feats.sum(-1)  # (bs, n_p, C, 7, 7)
        sampled_feats = sampled_feats.view(bs * num_prop, C, 7, 7)
        # (bs*n_p, C, 7, 7)

        return sampled_feats
        # (bs*n_p, C, 7, 7)

    def points_feats_sampling_bboxes_roi(self, points_feats, bboxes, pooler,
                                         img_metas):
        """
        This function samples the LiDAR features for the bboxes using pooler
        points_feats (list[Tensor]): shape (bs, 256, H, W) stride 8, 16, 32, 64
        bboxes (Tensor): (bs, n_p, 10) [cx, cy, cz, w, l, h, sin, cos, vx, vy]
        pooler (nn.Module): ROI Extractor
        img_metas (list[dict])
        pc_range (list): [x_min, y_,in, z_min, x_max, y_max, z_max]
        """

        # get the corners of the bboxes
        bbox_corners = boxes3d_to_corners3d(bboxes[..., :8],
                                            bottom_center=False, ry=False)
        # (bs, n_p, 8, 3) in world coord

        # convert corners to range [0, 110.4]
        pc_start_ = bboxes.new_tensor(
            [[self.pc_range[0], self.pc_range[1], self.pc_range[2]]])  # (1, 3)
        bbox_corners = bbox_corners - pc_start_  # (bs, n_p, 8, 3)

        # divide by voxel size to get index on BEV
        bbox_corners[..., 0:1] = bbox_corners[..., 0:1] / self.voxel_size[0]
        bbox_corners[..., 1:2] = bbox_corners[..., 1:2] / self.voxel_size[1]
        # range [0, 1472]

        bbox_corners_bev = bbox_corners[..., :2]  # (bs, n_p, 8, 2)

        # expect box_corners_in_bev: [B,N, 8, 2] -- [B,num_cam,N,8,2]
        minxy = torch.min(bbox_corners_bev, dim=2).values
        # (bs, n_p, 2)
        maxxy = torch.max(bbox_corners_bev, dim=2).values
        # (bs, n_p, 2)

        bbox2d = torch.cat([minxy, maxxy], dim=2)
        # (bs, n_p, 4)

        # convert bbox2d to ROI
        bbox2d_list = torch.split(bbox2d, 1)
        # ((1, n_p, 4), (1, n_p, 4),...bs)
        bbox2d_list = [lvl[0, :, :] for lvl in bbox2d_list]
        # [(n_p, 4), (n_p, 4), ... bs]

        rois = bbox2roi(bbox2d_list)  # (bs*n_p, 5) batch_id

        sampled_feats = pooler(points_feats[:pooler.num_inputs], rois)
        # (bs*n_p, C, 7, 7)

        return sampled_feats
        # (bs*n_p, C, 7, 7)


class DynamicConv(nn.Module):

    def __init__(self,
                 feat_channels: int,
                 dynamic_dim: int = 64,
                 dynamic_num: int = 2,
                 pooler_resolution: int = 7) -> None:
        super().__init__()

        self.feat_channels = feat_channels
        self.dynamic_dim = dynamic_dim
        self.dynamic_num = dynamic_num
        self.num_params = self.feat_channels * self.dynamic_dim
        self.dynamic_layer = nn.Linear(self.feat_channels,
                                       self.dynamic_num * self.num_params)

        self.norm1 = nn.LayerNorm(self.dynamic_dim)
        self.norm2 = nn.LayerNorm(self.feat_channels)

        self.activation = nn.ReLU(inplace=True)

        num_output = self.feat_channels * pooler_resolution ** 2
        self.out_layer = nn.Linear(num_output, self.feat_channels)
        self.norm3 = nn.LayerNorm(self.feat_channels)

    def forward(self, prop_feats, roi_feats):
        """Forward function.

        Args:
            prop_feats: (1,  bs * n_p, C)
            roi_feats: (7*7, bs * n_p, C)

        Returns:
        """
        features = roi_feats.permute(1, 0, 2)  # (bs*n_p, 7*7, C)
        parameters = self.dynamic_layer(prop_feats).permute(1, 0, 2)
        # (1, bs * n_p, 2 * C * C/4) --> (bs * n_p, 1, 2 * C * C/4)

        param1 = parameters[:, :, :self.num_params].view(
            -1, self.feat_channels, self.dynamic_dim)
        # (bs*n_p, C, C/4)
        param2 = parameters[:, :, self.num_params:].view(-1, self.dynamic_dim,
                                                         self.feat_channels)
        # (bs*n_p, C/4, C)

        # (bs*n_p, 7*7, C) * (bs*n_p, C, C/4) = (bs*n_p, 7*7, C/4)
        features = torch.bmm(features, param1)  # (bs*n_p, 7*7, C/4)
        features = self.norm1(features)
        features = self.activation(features)

        # (bs*n_p, 7*7, C/4) * (bs*n_p, C/4, C) = (bs*n_p, 7*7, C)
        features = torch.bmm(features, param2)  # (bs*n_p, 7*7, C)
        features = self.norm2(features)
        features = self.activation(features)

        features = features.flatten(1)  # (bs*n_p, 7*7*C)
        features = self.out_layer(features)  # (bs*n_p, C)
        features = self.norm3(features)
        features = self.activation(features)

        return features  # (bs*n_p, C)
