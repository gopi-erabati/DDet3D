import torch
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.ops import DynamicScatter
from mmdet3d.models.builder import VOXEL_ENCODERS, build_fusion_layer
from .utils import DynamicVFELayer


@VOXEL_ENCODERS.register_module()
class DynamicVFECustom(nn.Module):
    """Dynamic Voxel feature encoder used in DV-SECOND.

    It encodes features of voxels and their points. It could also fuse
    image feature into voxel features in a point-wise manner.
    The number of points inside the voxel varies.

    Args:
        in_channels (int): Input channels of VFE. Defaults to 4.
        feat_channels (list(int)): Channels of features in VFE.
        with_distance (bool): Whether to use the L2 distance of points to the
            origin point. Default False.
        with_cluster_center (bool): Whether to use the distance to cluster
            center of points inside a voxel. Default to False.
        with_voxel_center (bool): Whether to use the distance to center of
            voxel for each points inside a voxel. Default to False.
        voxel_size (tuple[float]): Size of a single voxel. Default to
            (0.2, 0.2, 4).
        point_cloud_range (tuple[float]): The range of points or voxels.
            Default to (0, -40, -3, 70.4, 40, 1).
        norm_cfg (dict): Config dict of normalization layers.
        mode (str): The mode when pooling features of points inside a voxel.
            Available options include 'max' and 'avg'. Default to 'max'.
        fusion_layer (dict | None): The config dict of fusion layer used in
            multi-modal detectors. Default to None.
        return_point_feats (bool): Whether to return the features of each
            points. Default to False.
    """

    def __init__(self,
                 in_channels=4,
                 feat_channels=[],
                 with_distance=False,
                 with_cluster_center=False,
                 with_voxel_center=False,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 mode='max',
                 fusion_layer=None,
                 return_point_feats=False,
                 with_centroid_aware_vox=True,
                 centroid_to_point_pos_emb_dims=32,
                 ):
        super(DynamicVFECustom, self).__init__()
        assert mode in ['avg', 'max']
        assert len(feat_channels) > 0
        # if with_cluster_center:
        #     in_channels += 3
        if with_centroid_aware_vox:
            in_channels += centroid_to_point_pos_emb_dims
        if with_voxel_center:
            in_channels += 3
        if with_distance:
            in_channels += 3
        self.in_channels = in_channels
        self._with_distance = with_distance
        self._with_cluster_center = with_cluster_center
        self._with_voxel_center = with_voxel_center
        self._with_centroid_aware_vox = with_centroid_aware_vox
        self.return_point_feats = return_point_feats
        self.fp16_enabled = False

        # Need pillar (voxel) size and x/y offset in order to calculate offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.vz = voxel_size[2]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]
        self.z_offset = self.vz / 2 + point_cloud_range[2]
        self.point_cloud_range = point_cloud_range
        self.scatter = DynamicScatter(voxel_size, point_cloud_range, True)

        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2

            vfe_layers.append(
                DynamicVFELayer(
                    in_filters,
                    out_filters,
                    norm_cfg))
        self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)
        self.vfe_scatter = DynamicScatter(voxel_size, point_cloud_range,
                                          (mode != 'max'))
        self.cluster_scatter = DynamicScatter(
            voxel_size, point_cloud_range, average_points=True)
        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)

        if with_centroid_aware_vox:
            self.cen2point_pos_enc = nn.Sequential(
                nn.Linear(3, centroid_to_point_pos_emb_dims, bias=False),
                nn.BatchNorm1d(centroid_to_point_pos_emb_dims),
                nn.Tanh(),
                nn.Linear(centroid_to_point_pos_emb_dims,
                          centroid_to_point_pos_emb_dims, bias=False),
                nn.BatchNorm1d(centroid_to_point_pos_emb_dims),
                nn.Tanh(),
            )

    def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors):
        """Map voxel features to its corresponding points.

        Args:
            pts_coors (torch.Tensor): Voxel coordinate of each point.
            voxel_mean (torch.Tensor): Voxel features to be mapped.
            voxel_coors (torch.Tensor): Coordinates of valid voxels

        Returns:
            torch.Tensor: Features or centers of each point.
        """
        # Step 1: scatter voxel into canvas
        # Calculate necessary things for canvas creation
        canvas_z = round(
            (self.point_cloud_range[5] - self.point_cloud_range[2]) / self.vz)
        canvas_y = round(
            (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
        canvas_x = round(
            (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)
        # canvas_channel = voxel_mean.size(1)
        batch_size = pts_coors[-1, 0].int() + 1
        canvas_len = canvas_z * canvas_y * canvas_x * batch_size
        # Create the canvas for this sample
        canvas = voxel_mean.new_zeros(canvas_len, dtype=torch.long)
        # Only include non-empty pillars
        indices = (
                voxel_coors[:, 0] * canvas_z * canvas_y * canvas_x +
                voxel_coors[:, 1] * canvas_y * canvas_x +
                voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
        # Scatter the blob back to the canvas
        canvas[indices.long()] = torch.arange(
            start=0, end=voxel_mean.size(0), device=voxel_mean.device)

        # Step 2: get voxel mean for each point
        voxel_index = (
                pts_coors[:, 0] * canvas_z * canvas_y * canvas_x +
                pts_coors[:, 1] * canvas_y * canvas_x +
                pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
        voxel_inds = canvas[voxel_index.long()]
        center_per_point = voxel_mean[voxel_inds, ...]
        return center_per_point

    # if out_fp16=True, the large numbers of points
    # lead to overflow error in following layers
    @force_fp32(out_fp16=False)
    def forward(self,
                features,
                coors,
                points=None,
                img_feats=None,
                img_metas=None):
        """Forward functions.

        Args:
            features (torch.Tensor): Features of voxels, shape is NxC.
            coors (torch.Tensor): Coordinates of voxels, shape is  Nx(1+NDim).
            points (list[torch.Tensor], optional): Raw points used to guide the
                multi-modality fusion. Defaults to None.
            img_feats (list[torch.Tensor], optional): Image fetures used for
                multi-modality fusion. Defaults to None.
            img_metas (dict, optional): [description]. Defaults to None.

        Returns:
            tuple: If `return_point_feats` is False, returns voxel features and
                its coordinates. If `return_point_feats` is True, returns
                feature of each points inside voxels.
        """
        features_ls = [features]
        origin_point_coors = features[:, :3]
        # Find distance of x, y, and z from cluster center
        if self._with_cluster_center:
            voxel_mean, mean_coors = self.cluster_scatter(features, coors)
            points_mean = self.map_voxel_center_to_point(
                coors, voxel_mean, mean_coors)
            # TODO: maybe also do cluster for reflectivity
            f_cluster = features[:, :3] - points_mean[:, :3]
            # Centroid to Point Pos encoding
            if self._with_centroid_aware_vox:
                f_cluster = self.cen2point_pos_enc(f_cluster)  # (N, 32)
            features_ls.append(f_cluster)

        # Find distance of x, y, and z from pillar center
        if self._with_voxel_center:
            f_center = features.new_zeros(size=(features.size(0), 3))
            f_center[:, 0] = features[:, 0] - (
                    coors[:, 3].type_as(features) * self.vx + self.x_offset)
            f_center[:, 1] = features[:, 1] - (
                    coors[:, 2].type_as(features) * self.vy + self.y_offset)
            f_center[:, 2] = features[:, 2] - (
                    coors[:, 1].type_as(features) * self.vz + self.z_offset)
            features_ls.append(f_center)

        if self._with_distance:
            points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
            features_ls.append(points_dist)

        # Combine together feature decorations
        features = torch.cat(features_ls, dim=-1)

        # features.requires_grad = True # for cam vis
        # features.register_hook(append_grad(-1)) # for cam vis
        # point_feat_dict[-1] = features.detach() # for cam vis

        low_level_point_feature = features
        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)

            # point_feats.register_hook(append_grad(i)) # for cam vis
            # point_feat_dict[i] = point_feats.detach() # for cam vis

            if (i == len(self.vfe_layers) - 1 and self.fusion_layer is not None
                    and img_feats is not None):
                point_feats = self.fusion_layer(img_feats, points, point_feats,
                                                img_metas)
            voxel_feats, voxel_coors = self.vfe_scatter(point_feats, coors)
            if i != len(self.vfe_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = self.map_voxel_center_to_point(
                    coors, voxel_feats, voxel_coors)
                features = torch.cat([point_feats, feat_per_point], dim=1)
        if self.return_point_feats:
            return point_feats
        return voxel_feats, voxel_coors

#
# @VOXEL_ENCODERS.register_module()
# class DynamicVFEWithCenAware(nn.Module):
#     """Dynamic Voxel feature encoder used in DV-SECOND.
#
#     Includes Centroid Aware Voxelization as in Fast Point Transformer CVPR 2022
#     http://cvlab.postech.ac.kr/research/FPT/
#
#     It encodes features of voxels and their points. It could also fuse
#     image feature into voxel features in a point-wise manner.
#     The number of points inside the voxel varies.
#
#     Args:
#         in_channels (int): Input channels of VFE. Defaults to 4.
#         feat_channels (list(int)): Channels of features in VFE.
#         with_distance (bool): Whether to use the L2 distance of points to the
#             origin point. Default False.
#         with_cluster_center (bool): Whether to use the distance to cluster
#             center of points inside a voxel. Default to False.
#         with_voxel_center (bool): Whether to use the distance to center of
#             voxel for each points inside a voxel. Default to False.
#         voxel_size (tuple[float]): Size of a single voxel. Default to
#             (0.2, 0.2, 4).
#         point_cloud_range (tuple[float]): The range of points or voxels.
#             Default to (0, -40, -3, 70.4, 40, 1).
#         norm_cfg (dict): Config dict of normalization layers.
#         mode (str): The mode when pooling features of points inside a voxel.
#             Available options include 'max' and 'avg'. Default to 'max'.
#         fusion_layer (dict | None): The config dict of fusion layer used in
#             multi-modal detectors. Default to None.
#         return_point_feats (bool): Whether to return the features of each
#             points. Default to False.
#     """
#
#     def __init__(self,
#                  in_channels=4,
#                  feat_channels=[],
#                  with_distance=False,
#                  with_cluster_center=False,
#                  with_voxel_center=False,
#                  voxel_size=(0.2, 0.2, 4),
#                  point_cloud_range=(0, -40, -3, 70.4, 40, 1),
#                  norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
#                  mode='max',
#                  fusion_layer=None,
#                  return_point_feats=False,
#                  with_centroid_aware_vox=True,
#                  centroid_to_point_pos_emb_dims=32,
#                  ):
#         super(DynamicVFEWithCenAware, self).__init__()
#         assert mode in ['avg', 'max']
#         assert len(feat_channels) > 0
#         # if with_cluster_center:
#         #     in_channels += 3
#         if with_centroid_aware_vox:
#             in_channels += centroid_to_point_pos_emb_dims
#         if with_voxel_center:
#             in_channels += 3
#         if with_distance:
#             in_channels += 3
#         self.in_channels = in_channels
#         self._with_distance = with_distance
#         self._with_cluster_center = with_cluster_center
#         self._with_voxel_center = with_voxel_center
#         self._with_centroid_aware_vox = with_centroid_aware_vox
#         self.return_point_feats = return_point_feats
#         self.fp16_enabled = False
#
#         # Need pillar (voxel) size and x/y offset in order to calculate offset
#         self.vx = voxel_size[0]
#         self.vy = voxel_size[1]
#         self.vz = voxel_size[2]
#         self.x_offset = self.vx / 2 + point_cloud_range[0]
#         self.y_offset = self.vy / 2 + point_cloud_range[1]
#         self.z_offset = self.vz / 2 + point_cloud_range[2]
#         self.point_cloud_range = point_cloud_range
#         self.scatter = DynamicScatter(voxel_size, point_cloud_range, True)
#
#         feat_channels = [self.in_channels] + list(feat_channels)
#         vfe_layers = []
#         for i in range(len(feat_channels) - 1):
#             in_filters = feat_channels[i]
#             out_filters = feat_channels[i + 1]
#             if i > 0:
#                 in_filters *= 2
#
#             vfe_layers.append(
#                 DynamicVFELayer(
#                     in_filters,
#                     out_filters,
#                     norm_cfg))
#         self.vfe_layers = nn.ModuleList(vfe_layers)
#         self.num_vfe = len(vfe_layers)
#         self.vfe_scatter = DynamicScatter(voxel_size, point_cloud_range,
#                                           (mode != 'max'))
#         self.cluster_scatter = DynamicScatter(
#             voxel_size, point_cloud_range, average_points=True)
#         self.fusion_layer = None
#         if fusion_layer is not None:
#             self.fusion_layer = build_fusion_layer(fusion_layer)
#
#         if with_centroid_aware_vox:
#             self.cen2point_pos_enc = nn.Sequential(
#                 nn.Linear(3, centroid_to_point_pos_emb_dims, bias=False),
#                 nn.BatchNorm1d(centroid_to_point_pos_emb_dims),
#                 nn.Tanh(),
#                 nn.Linear(centroid_to_point_pos_emb_dims,
#                           centroid_to_point_pos_emb_dims, bias=False),
#                 nn.BatchNorm1d(centroid_to_point_pos_emb_dims),
#                 nn.Tanh(),
#             )
#
#     def map_voxel_center_to_point(self, pts_coors, voxel_mean, voxel_coors):
#         """Map voxel features to its corresponding points.
#
#         Args:
#             pts_coors (torch.Tensor): Voxel coordinate of each point.
#             voxel_mean (torch.Tensor): Voxel features to be mapped.
#             voxel_coors (torch.Tensor): Coordinates of valid voxels
#
#         Returns:
#             torch.Tensor: Features or centers of each point.
#         """
#         # Step 1: scatter voxel into canvas
#         # Calculate necessary things for canvas creation
#         canvas_z = round(
#             (self.point_cloud_range[5] - self.point_cloud_range[2]) / self.vz)
#         canvas_y = round(
#             (self.point_cloud_range[4] - self.point_cloud_range[1]) / self.vy)
#         canvas_x = round(
#             (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.vx)
#         # canvas_channel = voxel_mean.size(1)
#         batch_size = pts_coors[-1, 0].int() + 1
#         canvas_len = canvas_z * canvas_y * canvas_x * batch_size
#         # Create the canvas for this sample
#         canvas = voxel_mean.new_zeros(canvas_len, dtype=torch.long)
#         # Only include non-empty pillars
#         indices = (
#                 voxel_coors[:, 0] * canvas_z * canvas_y * canvas_x +
#                 voxel_coors[:, 1] * canvas_y * canvas_x +
#                 voxel_coors[:, 2] * canvas_x + voxel_coors[:, 3])
#         # Scatter the blob back to the canvas
#         canvas[indices.long()] = torch.arange(
#             start=0, end=voxel_mean.size(0), device=voxel_mean.device)
#
#         # Step 2: get voxel mean for each point
#         voxel_index = (
#                 pts_coors[:, 0] * canvas_z * canvas_y * canvas_x +
#                 pts_coors[:, 1] * canvas_y * canvas_x +
#                 pts_coors[:, 2] * canvas_x + pts_coors[:, 3])
#         voxel_inds = canvas[voxel_index.long()]
#         center_per_point = voxel_mean[voxel_inds, ...]
#         return center_per_point
#
#     # if out_fp16=True, the large numbers of points
#     # lead to overflow error in following layers
#     @force_fp32(out_fp16=False)
#     def forward(self,
#                 features,
#                 coors,
#                 points=None,
#                 img_feats=None,
#                 img_metas=None):
#         """Forward functions.
#
#         Args:
#             features (torch.Tensor): Features of voxels, shape is NxC.
#             coors (torch.Tensor): Coordinates of voxels, shape is  Nx(1+NDim).
#             points (list[torch.Tensor], optional): Raw points used to guide the
#                 multi-modality fusion. Defaults to None.
#             img_feats (list[torch.Tensor], optional): Image fetures used for
#                 multi-modality fusion. Defaults to None.
#             img_metas (dict, optional): [description]. Defaults to None.
#
#         Returns:
#             tuple: If `return_point_feats` is False, returns voxel features and
#                 its coordinates. If `return_point_feats` is True, returns
#                 feature of each points inside voxels.
#         """
#         features_ls = [features]
#         origin_point_coors = features[:, :3]
#         # Find distance of x, y, and z from cluster center
#         if self._with_cluster_center:
#             voxel_mean, mean_coors = self.cluster_scatter(features, coors)
#             points_mean = self.map_voxel_center_to_point(
#                 coors, voxel_mean, mean_coors)
#             # TODO: maybe also do cluster for reflectivity
#             f_cluster = features[:, :3] - points_mean[:, :3]
#             # Centroid to Point Pos encoding
#             if self._with_centroid_aware_vox:
#                 f_cluster = self.cen2point_pos_enc(f_cluster)  # (N, 32)
#             features_ls.append(f_cluster)
#
#         # Find distance of x, y, and z from pillar center
#         if self._with_voxel_center:
#             f_center = features.new_zeros(size=(features.size(0), 3))
#             f_center[:, 0] = features[:, 0] - (
#                     coors[:, 3].type_as(features) * self.vx + self.x_offset)
#             f_center[:, 1] = features[:, 1] - (
#                     coors[:, 2].type_as(features) * self.vy + self.y_offset)
#             f_center[:, 2] = features[:, 2] - (
#                     coors[:, 1].type_as(features) * self.vz + self.z_offset)
#             features_ls.append(f_center)
#
#         if self._with_distance:
#             points_dist = torch.norm(features[:, :3], 2, 1, keepdim=True)
#             features_ls.append(points_dist)
#
#         # Combine together feature decorations
#         features = torch.cat(features_ls, dim=-1)
#
#         # features.requires_grad = True # for cam vis
#         # features.register_hook(append_grad(-1)) # for cam vis
#         # point_feat_dict[-1] = features.detach() # for cam vis
#
#         low_level_point_feature = features
#         for i, vfe in enumerate(self.vfe_layers):
#             point_feats = vfe(features)
#
#             # point_feats.register_hook(append_grad(i)) # for cam vis
#             # point_feat_dict[i] = point_feats.detach() # for cam vis
#
#             if (i == len(self.vfe_layers) - 1 and self.fusion_layer is not None
#                     and img_feats is not None):
#                 point_feats = self.fusion_layer(img_feats, points, point_feats,
#                                                 img_metas)
#             voxel_feats, voxel_coors = self.vfe_scatter(point_feats, coors)
#             if i != len(self.vfe_layers) - 1:
#                 # need to concat voxel feats if it is not the last vfe
#                 feat_per_point = self.map_voxel_center_to_point(
#                     coors, voxel_feats, voxel_coors)
#                 features = torch.cat([point_feats, feat_per_point], dim=1)
#         if self.return_point_feats:
#             return point_feats
#         print(voxel_feats.shape)
#         return voxel_feats, voxel_coors  # (M, 128), (M, 1+3)
