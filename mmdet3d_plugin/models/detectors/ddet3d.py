import torch
import torch.nn.functional as F
from mmcv.runner import auto_fp16, force_fp32
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d.models.builder import (build_voxel_encoder,
                                    build_middle_encoder, build_backbone,
                                    build_neck, build_head, DETECTORS)
from mmdet3d.ops import Voxelization
from mmdet3d_plugin.models.utils.grid_mask import GridMask


@DETECTORS.register_module()
class DDet3D(Base3DDetector):
    """
    DDet3D is Diffusion based 3D Object Detection
    """

    def __init__(self,
                 use_grid_mask=False,
                 img_backbone=None,
                 img_neck=None,
                 freeze_img=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_backbone=None,
                 pts_neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None
                 ):
        super(DDet3D, self).__init__(init_cfg)

        # IMAGE FEATURES : BACKBONE + NECK
        # build backbones (img)
        if img_backbone is not None:
            self.img_backbone = build_backbone(img_backbone)
        else:
            self.img_backbone = None
        self.grid_mask = GridMask(True, True, rotate=1, offset=False,
                                  ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.freeze_img = freeze_img

        # build neck (img)
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        else:
            self.img_neck = None

        # POINTS FEATURES : Points Voxel Layer, Points Voxel Encoder,
        # Points Voxel Scatter, Pts backbone (SECOND), Pts neck (FPN)
        if pts_voxel_layer:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
            self.pts_voxel_layer_cfg = pts_voxel_layer
        if pts_voxel_encoder:
            self.pts_voxel_encoder = build_voxel_encoder(
                pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = build_middle_encoder(
                pts_middle_encoder)
        if pts_backbone:
            self.pts_backbone = build_backbone(pts_backbone)
        if pts_neck is not None:
            self.pts_neck = build_neck(pts_neck)

        # build head
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def init_weights(self):
        super(DDet3D, self).init_weights()

        if self.freeze_img:
            if self.img_backbone is not None:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False
            if self.img_neck is not None:
                for param in self.img_neck.parameters():
                    param.requires_grad = False

    def train_step(self, data, optimizer, epoch=None):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a
                  weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                  logger.
                - ``num_samples`` indicates the batch size (when the model is
                  DDP, it means the batch size on each GPU), which is used for
                  averaging the logs.
        """
        losses = self(**data, epoch=epoch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    @force_fp32(apply_to=('img', 'points'))
    def forward(self, img=None, points=None, return_loss=True, epoch=None,
                **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.

        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            return self.forward_train(img, points, epoch=epoch, **kwargs)
        else:
            return self.forward_test(img, points, **kwargs)

    def forward_train(self,
                      img,
                      points,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      img_metas,
                      gt_bboxes_3d_ignore=None,
                      epoch=None):
        """
        Args:
            img (Tesnor): Input RGB image of shape (B, N, C, H, W)
            points (list[Tensor]): Points of each sample of shape (N, d)
            gt_bboxes_3d (list[Tensor]): A list of tensors of batch length,
                each containing the ground truth 3D boxes of shape (num_box, 7)
            gt_labels_3d (list[Tensor]): A list of tensors of batch length,
                each containing the ground truth 3D boxes labels
                of shape (num_box, )
            img_metas (list[dict]): A list of image info where each dict
                has: 'img_Shape', 'flip' and other detailssee
                :class `mmdet3d.datasets.pipelines.Collect`.
            gt_bboxes_3d_ignore (None | list[Tensor]): Specify which
                bounding boxes can be ignored when computing the loss.
        Returns:
            dict [str, Tensor]: A dictionary of loss components
        """
        # Extract Image and Point Features
        img_feats, point_feats = self.extract_feat(img, points, img_metas)
        # list[(bs, n_cam, C, H, W), ...], list[(B, 256, H, W), ...]

        losses = self.bbox_head.forward_train(img_feats, point_feats,
                                              gt_bboxes_3d, gt_labels_3d,
                                              gt_bboxes_3d_ignore, img_metas,
                                              points=points,
                                              epoch=epoch)
        return losses

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, points, img_metas=None):
        """Extract Image and Point Features
        Args:
            img (Tensor): Images of shape (B, N, C, H, W)
            points (List[Tensor]): Points for each sample
            img_metas (list[dict]): A list of image info where each dict
                has: 'img_Shape', 'flip' and other details see
                :class `mmdet3d.datasets.pipelines.Collect`.

        Returns:
            tuple(List[Tensor], List[Tensor]): First item of tuple is a list of
                image feats at different strides (x8, x16, x32, x64) of
                shape (B, N, C, H, W) and second item is a list of point feats
                as BEV in different strides (x4, x8, x16, x32) of 1024 size of
                shape (B, C, H, W).
        """

        # Image Features
        if img is not None:
            img_feats = self.extract_img_feat(img, img_metas)  # list[Tensor]
            # (bs, n_cam, C, H, W)
        else:
            img_feats = None

        # Point Features
        if points is not None:
            point_feats = self.extract_point_features(points)  # list[Tensor]
        else:
            point_feats = None

        return img_feats, point_feats
        # list[(bs, n_cam, C, H, W), ...], list[(bs, 256, H, W), ...]

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.img_neck is not None:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped  # list[Tensor] for each level

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """

        # Hard Voxelize
        if self.pts_voxel_layer_cfg['max_num_points'] != -1:
            voxels, coors, num_points = [], [], []
            for res in points:
                res_voxels, res_coors, res_num_points = self.pts_voxel_layer(
                    res)
                voxels.append(res_voxels)
                coors.append(res_coors)
                num_points.append(res_num_points)
            voxels = torch.cat(voxels,
                               dim=0)  # (B*num_voxels, num_points, dim)
            num_points = torch.cat(num_points, dim=0)  # (B*num_voxels, )
            coors_batch = []
            for i, coor in enumerate(coors):  # (num_voxels, 3)
                coor_pad = F.pad(coor, (1, 0), mode='constant',
                                 value=i)  # (n, 4)
                coors_batch.append(coor_pad)
            coors_batch = torch.cat(coors_batch, dim=0)  # (B*num_vox, 4)
            return voxels, num_points, coors_batch
        elif self.pts_voxel_layer_cfg['max_num_points'] == -1:
            # Dynamic Voxelization w/ center aware
            coors = []
            # dynamic voxelization only provide a coors mapping
            for res in points:
                res_coors = self.pts_voxel_layer(res)  # (N, 3)
                coors.append(res_coors)
            points = torch.cat(points, dim=0)  # (B*N, 4)
            coors_batch = []
            for i, coor in enumerate(coors):
                coor_pad = F.pad(coor, (1, 0), mode='constant',
                                 value=i)  # (N,1+3)
                coors_batch.append(coor_pad)
            coors_batch = torch.cat(coors_batch, dim=0)  # (B*N, 1+3)
            return points, coors_batch  # (B*N, 4), (B*N, 1+3)

    def extract_point_features(self, points):
        """ Extract features of Points using encoder, middle encoder,
        backbone and neck.
        Here points is list[Tensor] of batch """

        # Hard Voxelize
        if self.pts_voxel_layer_cfg['max_num_points'] != -1:
            voxels, num_points, coors = self.voxelize(points)
            voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
            batch_size = coors[-1, 0] + 1
            x = self.pts_middle_encoder(voxel_features, coors, batch_size)
            x = self.pts_backbone(x)
            if self.pts_neck is not None:
                x = self.pts_neck(x)
            return x
            # [(B, 256, H, W), ...] H, W is 8, 16, 32, 64 stride of 1472
        elif self.pts_voxel_layer_cfg['max_num_points'] == -1:
            # Dynamic Voxelization w/ center aware
            voxels, coors = self.voxelize(points)  # (B*N, 4), (B*N, 1+3)
            voxel_features, feature_coors = self.pts_voxel_encoder(voxels,
                                                                   coors)
            # (M, 128), (M, 1+3)
            batch_size = feature_coors[-1, 0].item() + 1
            x = self.pts_middle_encoder(voxel_features, feature_coors,
                                        batch_size)
            x = self.pts_backbone(x)
            if self.pts_neck is not None:
                x = self.pts_neck(x)
            return x

    def forward_test(self,
                     img,
                     points,
                     img_metas,
                     **kwargs):
        """
        Args:
            img (Tensor): Input RGB image of shape (B, N, C, H, W)
            points (list[Tensor]): Points of each sample of shape (N, d)
            img_metas (list[dict]): A list of image info where each dict
                has: 'img_Shape', 'flip' and other details see
                :class `mmdet3d.datasets.pipelines.Collect`.
        """
        # for var, name in [(img, 'img'), (img_metas, 'img_metas')]:
        #     if not isinstance(var, list):
        #         raise TypeError('{} must be a list, but got {}'.format(
        #             name, type(var)))
        #
        # num_augs = len(img)
        # if num_augs != len(img_metas):
        #     raise ValueError(
        #         'num of augmentations ({}) != num of image meta ({})'.format(
        #             len(img), len(img_metas)))

        if img is None:
            img = [img]
        elif points is None:
            points = [points]

        return self.simple_test(img[0], points[0], img_metas[0], **kwargs)

    def simple_test(self, img, points, img_metas, rescale=False):
        """ Test function without test-time augmentation.

        Args:
            img (Tensor): Input RGB image of shape (B, N, C, H, W)
            points (list[Tensor]): Points of each sample of shape (N, d)
            img_metas (list[dict]): A list of image info where each dict
                has: 'img_Shape', 'flip' and other details see
                :class `mmdet3d.datasets.pipelines.Collect`.

        Returns:
            list[dict]: Predicted 3d boxes. Each list consists of a dict
            with keys: boxes_3d, scores_3d, labels_3d.
        """
        img_feats, point_feats = self.extract_feat(img, points, img_metas)
        # list[(bs, n_cam, C, H, W), ...], list[(B, 256, H, W), ...]

        bbox_list = self.bbox_head.simple_test_bboxes(img_feats,
                                                      point_feats,
                                                      img_metas)
        # [tuple[LiDARBbox, Tensor, Tensor],.... bs]
        # (n_p, 9) (n_p, ), (n_p, )

        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        bbox_list = [dict() for i in range(len(img_metas))]
        for result_dict, pts_bbox in zip(bbox_list, bbox_results):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    def aug_test(self, img, proj_img, proj_idxs, img_idxs, img_metas,
                 rescale=False):
        pass
