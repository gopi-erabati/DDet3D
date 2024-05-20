# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified from https://github.com/ShoufaChen/DiffusionDet/blob/main/diffusiondet/loss.py   # noqa

# This work is licensed under the CC-BY-NC 4.0 License.
# Users should be careful about adopting these features in any commercial matters.  # noqa
# For more details, please refer to https://github.com/ShoufaChen/DiffusionDet/blob/main/LICENSE    # noqa


import torch
import torch.nn as nn
from mmengine.structures import InstanceData
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.core.bbox.iou_calculators import build_iou_calculator
from ..util import normalize_bbox, boxes3d_to_corners3d, denormalize_bbox


@BBOX_ASSIGNERS.register_module()
class DDet3DAssigner(nn.Module):
    """ This assigner computes an assignmnet between the targets and preds.
    targets don't include no-object

    Because of this, in general, there are more predictions than targets. In
    this case, we do a 1-to-k (dynamic) matching of the best predictions, while
    the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self,
                 cls_cost,
                 reg_cost,
                 iou_cost,
                 center_radius=1.5,
                 candidate_topk=5,
                 pc_range=None,
                 iou_calculator=None):
        super().__init__()

        if iou_calculator is None:
            iou_calculator = dict(type='BboxOverlaps3D',
                                  coordinate='lidar')
        self.center_radius = center_radius
        self.candidate_topk = candidate_topk
        self.pc_range = pc_range

        self.use_focal_loss = False
        if cls_cost.get('type') == 'FocalLossCost':
            self.use_focal_loss = True

        # build match costs
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.iou_calculator = build_iou_calculator(iou_calculator)

    def forward(self, outputs, batch_gt_instances):
        """ Forward fucntion of DDet3DAssigner
        Args:
            outputs (dict): {'pred_logits':(bs, n_p, #cls),
                             'pred_boxes':(bs, n_p, 10),
                             'aux_outputs': [{'pred_logits':(bs, n_p, #cls),
                                              'pred_boxes':(bs, n_p, 10)},
                                               { }, { }, ...]}
                    # pred boxes  center:abs and size:log
                    # [cx, cy, cz, w, l, h, sin, cos, vx, vy]
            batch_gt_instances (list[gt_instance]): bboxes (n_gt, 9),
                                                    labels (n_gt, ),
        """
        assert 'pred_logits' in outputs and 'pred_boxes' in outputs

        pred_logits = outputs['pred_logits']  # (bs, n_p, #cls)
        pred_bboxes = outputs['pred_boxes']  # (bs, n_p, 10)
        batch_size = len(batch_gt_instances)

        assert batch_size == pred_logits.shape[0] == pred_bboxes.shape[0]
        batch_indices = []
        for i in range(batch_size):
            pred_instances = InstanceData()
            pred_instances.bboxes = pred_bboxes[i, ...]  # (n_p, 10)
            pred_instances.scores = pred_logits[i, ...]  # (n_p, #cls)
            gt_instances = batch_gt_instances[i]
            indices = self.single_assigner(pred_instances, gt_instances)
            # (n_p, ), (n_p_gt, )
            # fg_mask_inboxes: gives pred indices where it has one matched gt
            # matched_gt_inds: gives the indices of matched gt for each pred box
            batch_indices.append(indices)
        return batch_indices
        # list[(n_p, )(n_p_gt), ()(), ...bs]
        # fg_mask_inboxes: gives pred indices where it has one matched gt
        # matched_gt_inds: gives the indices of matched gt for each pred box

    def single_assigner(self, pred_instances, gt_instances):
        """
        Args:
            pred_instances (InstanceData): bboxes (n_p, 10), scores (n_p, #cls)
                 pred boxes  center:abs and size:log
                [cx, cy, cz, w, l, h, sin, cos, vx, vy]
            gt_instances (InstanceData): bboxes (n_gt, 9), labels (n_gt, )

        """
        with torch.no_grad():
            gt_bboxes = gt_instances.bboxes  # (n_gt, 9)
            pred_bboxes = pred_instances.bboxes  # (n_p, 10)
            num_gt = gt_bboxes.size(0)  # n_gt

            if num_gt == 0:  # empty object
                valid_mask = pred_bboxes.new_zeros((pred_bboxes.shape[0],),
                                                   dtype=torch.bool)
                matched_gt_inds = pred_bboxes.new_zeros((gt_bboxes.shape[0],),
                                                        dtype=torch.long)
                return valid_mask, matched_gt_inds
                # (n_p, ), (n_gt, )

            valid_mask, is_in_boxes_and_center = \
                self.get_in_gt_and_in_center_info(pred_bboxes, gt_bboxes)
            # (n_p, ) , (n_p, n_gt)
            # valid_mask tells whether center of pred box is within gt
            # box or within some radius of any of the gt center
            # is_in_boxes_and_center tells whether center of pred box is within
            # each of the gt box and within some radius of each of the gt center

            # calculate costs
            pred_cls = pred_instances.scores  # (n_p, #cls)
            gt_labels = gt_instances.labels  # (n_gt, )
            cls_cost = self.cls_cost(pred_cls, gt_labels)  # (n_p, n_gt)
            reg_cost = self.reg_cost(pred_bboxes[:, :8], normalize_bbox(
                gt_bboxes[:, :7], self.pc_range))
            # firstly, denormalize pred_bboxes to get in world
            # coordinates as the 'preds' are trained for normalized gts
            ious = self.iou_calculator(denormalize_bbox(pred_bboxes,
                                                        self.pc_range),
                                       gt_bboxes)
            iou_cost = self.iou_cost(ious)  # (n_p, n_gt)

            cost_list = [cls_cost, reg_cost, iou_cost,
                         (~is_in_boxes_and_center) * 100.0]

            cost_matrix = torch.stack(cost_list).sum(0)  # (n_p, n_gt)
            cost_matrix[~valid_mask] = cost_matrix[~valid_mask] + 10000.0

            fg_mask_inboxes, matched_gt_inds = \
                self.dynamic_k_matching(cost_matrix, ious, num_gt)
        return fg_mask_inboxes, matched_gt_inds
        # (n_p, ), (n_p_gt, )
        # fg_mask_inboxes: gives pred indices where it has one matched gt
        # matched_gt_inds: gives the indices of matched gt for each pred box

    def get_in_gt_and_in_center_info(self, pred_bboxes, gt_bboxes):
        """Get the information of which prior is in gt bboxes and gt center
        priors.
        Args:
            pred_bboxes (Tensor): (n_p, 10)
            gt_bboxes (Tensor): (n_gt, 9)

        """
        pred_bboxes_center_x = pred_bboxes[:, 0].unsqueeze(1)  # (n_p, 1)
        pred_bboxes_center_y = pred_bboxes[:, 1].unsqueeze(1)  # (n_p, 1)
        pred_bboxes_center_z = pred_bboxes[:, 2].unsqueeze(1)  # (n_p, 1)

        # to check whether pred box center is in a gt box, we need the
        # min&max of x, y, z of gt box
        # to get min&max of gt box we need to get corners
        gt_bboxes_corners = boxes3d_to_corners3d(gt_bboxes[...,
                                                 :7].unsqueeze(0),
                                                 bottom_center=False, ry=True)
        # (1, n_gt, 8, 3)
        gt_bboxes_corners = gt_bboxes_corners.squeeze(0)  # (n_gt, 8, 3)
        # get minmax values of corners
        minxyz = torch.min(gt_bboxes_corners, dim=1).values
        # (n_gt, 3)
        maxxyz = torch.max(gt_bboxes_corners, dim=1).values
        # (n_gt, 3)

        # whether the center of each anchor is inside a gt box
        b_ba = pred_bboxes_center_x > minxyz[:, 0].unsqueeze(0)
        b_f = pred_bboxes_center_x < maxxyz[:, 0].unsqueeze(0)
        b_l = pred_bboxes_center_y > minxyz[:, 1].unsqueeze(0)
        b_r = pred_bboxes_center_y < maxxyz[:, 1].unsqueeze(0)
        b_b = pred_bboxes_center_z > minxyz[:, 2].unsqueeze(0)
        b_t = pred_bboxes_center_z < maxxyz[:, 2].unsqueeze(0)
        # (n_p, n_gt)
        # (b_l.long()+b_r.long()+b_t.long()+b_b.long())==4 [300,num_gt] ,
        is_in_boxes = ((b_ba.long() + b_f.long() + b_l.long() + b_r.long() +
                        b_t.long() + b_b.long()) == 6)  # (n_p, n_gt)
        is_in_boxes_all = is_in_boxes.sum(1) > 0  # [num_query]  (n_p,)

        # in fixed center
        center_radius = self.center_radius
        # Modified to self-adapted sampling --- the center size depends
        # on the size of the gt boxes
        # https://github.com/dulucas/UVO_Challenge/blob/main/Track1/detection/mmdet/core/bbox/assigners/rpn_sim_ota_assigner.py#L212    # noqa
        b_ba = pred_bboxes_center_x > (gt_bboxes[:, 0] - (center_radius *
                                                          gt_bboxes[:,
                                                          3])).unsqueeze(0)
        b_f = pred_bboxes_center_x < (gt_bboxes[:, 0] + (center_radius *
                                                         gt_bboxes[:,
                                                         3])).unsqueeze(0)
        b_l = pred_bboxes_center_y > (gt_bboxes[:, 1] - (center_radius *
                                                         gt_bboxes[:,
                                                         4])).unsqueeze(0)
        b_r = pred_bboxes_center_y < (gt_bboxes[:, 1] + (center_radius *
                                                         gt_bboxes[:,
                                                         4])).unsqueeze(0)
        b_b = pred_bboxes_center_z > (gt_bboxes[:, 2] - (center_radius *
                                                         gt_bboxes[:,
                                                         5])).unsqueeze(0)
        b_t = pred_bboxes_center_z < (gt_bboxes[:, 2] + (center_radius *
                                                         gt_bboxes[:,
                                                         5])).unsqueeze(0)
        # (n_p, n_gt)

        is_in_centers = ((b_ba.long() + b_f.long() + b_l.long() + b_r.long() +
                          b_t.long() + b_b.long()) == 6)  # (n_p, n_gt)
        is_in_centers_all = is_in_centers.sum(1) > 0  # (n_p,)

        # is_in_boxes_all: gives whether center of each of the pred box is in
        # any of the gt box
        # is_in_centers_all: gives whether center of each of the pred box is
        # within the radius of any of the gt center
        # is_in_boxes: gives whether center of each of the pred box is in
        # each of the gt box
        # is_in_centers: gives whether center of each of the pred box is
        # within the radius of each of the gt center
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all  # (n_p, )
        is_in_boxes_and_center = (is_in_boxes & is_in_centers)  # (n_p, n_gt)

        return is_in_boxes_anchor, is_in_boxes_and_center
        # (n_p, ), (n_p, n_gt)
        # is_in_boxes_anchor tells whether center of pred box is within gt
        # box or within some radius of any of the gt center
        # is_in_boxes_and_center tells whether center of pred box is within
        # each of the gt box and within some radius of each of the gt center

    def dynamic_k_matching(self, cost, ious, num_gt):
        """Use IoU and matching cost to calculate the dynamic top-k positive
        targets.
        Args:
            cost (Tensor): (n_p, n_gt)
            ious (Tensor): (n_p, n_gt)
            num_gt (int):
        """
        matching_matrix = torch.zeros_like(cost)  # (n_p, n_gt)

        # select candidate topk ious for dynamic-k calculation
        candidate_topk = min(self.candidate_topk, ious.size(0))
        topk_ious, _ = torch.topk(ious, candidate_topk, dim=0)
        # (topk, n_gt)

        # calculate dynamic k for each gt
        # get the dynamic 'k's where it is sum of topk of ious
        dynamic_ks = torch.clamp(topk_ious.sum(0).int(), min=1)  # (n_gt, )
        for gt_idx in range(num_gt):
            # get the indices of #dynamic-ks cost for each gt, where cost is
            # small
            _, pos_idx = torch.topk(
                cost[:, gt_idx], k=dynamic_ks[gt_idx], largest=False)
            matching_matrix[:, gt_idx][pos_idx] = 1.0
            # matching matrix contains the positions of pred for each gt,
            # where costs of pred is small for #dynamic-k

        del topk_ious, dynamic_ks, pos_idx

        # but matching cost may contain more gts for each pred, so remove them
        # get the locations of pred where it has more than one matched gt
        prior_match_gt_mask = matching_matrix.sum(1) > 1  # (n_p, )
        if prior_match_gt_mask.sum() > 0:
            # get the index where the cost is min for that pred which has
            # more than one gt
            _, cost_argmin = torch.min(cost[prior_match_gt_mask, :], dim=1)
            # zero the all gts of preds where it has more than one gt
            matching_matrix[prior_match_gt_mask, :] *= 0
            # but make that particular gt whih has min cost to 1, so that
            # that pred would have only one matched gt
            matching_matrix[prior_match_gt_mask, cost_argmin] = 1.0
        # the above is to remove more than one gt for each pred

        # if the gt is not matched with atleast one pred then enter while loop
        while (matching_matrix.sum(0) == 0).any():
            # increase the cost of matched preds
            matched_query_id = matching_matrix.sum(1) > 0  # (n_p, )
            cost[matched_query_id] += 100000.0
            # get the umatch ids of gt
            unmatch_id = torch.nonzero(
                matching_matrix.sum(0) == 0, as_tuple=False).squeeze(1)
            # for each unatch_id of gt, get the index of min cost of pred
            # and set that particular index matching cost to 1
            for gt_idx in unmatch_id:
                pos_idx = torch.argmin(cost[:, gt_idx])
                matching_matrix[:, gt_idx][pos_idx] = 1.0
            # the above is to make atleast one pred for each gt

            # the below is to make sure again that not more than one gt for
            # each pred
            if (matching_matrix.sum(1) > 1).sum() > 0:
                # prior_match_gt_mask = matching_matrix.sum(1) > 1  # (n_p, )
                _, cost_argmin = torch.min(cost[prior_match_gt_mask], dim=1)
                matching_matrix[prior_match_gt_mask] = matching_matrix[
                    prior_match_gt_mask] * 0
                matching_matrix[prior_match_gt_mask, cost_argmin] = 1.0

        assert not (matching_matrix.sum(0) == 0).any()
        # get foreground mask inside box and center prior
        fg_mask_inboxes = matching_matrix.sum(1) > 0  # (n_p, )
        matched_gt_inds = matching_matrix[fg_mask_inboxes, :].argmax(1)
        # (n_p_gt, )

        return fg_mask_inboxes, matched_gt_inds
        # (n_p, ), (n_p_gt, )
        # fg_mask_inboxes: gives pred indices where it has one matched gt
        # matched_gt_inds: gives the indices of matched gt for each pred box
