import torch
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import AssignResult
from mmdet.core.bbox.assigners import BaseAssigner
from mmdet.core.bbox.match_costs import build_match_cost
from ..util import (normalize_bbox, denormalize_bbox,
                    normalize_bbox_bev, denormalize_bbox_bev)

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@BBOX_ASSIGNERS.register_module()
class HungarianAssignerDDet3D(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.
    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. The
    targets don't include the no_object, so generally there are more
    predictions than targets. After the one-to-one matching, the un-matched
    are treated as backgrounds. Thus each query prediction will be assigned
    with `0` or a positive integer indicating the ground truth index:
    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    Args:
        cls_cost (dict): Config for classification cost
        reg_cost (dict): Config for regression cost
        pc_range (list[float]): Point cloud range
    """

    def __init__(self,
                 cls_cost=None,
                 reg_cost=None,
                 bev_iou_cost=None,
                 pc_range=None,
                 is_box_bev=False):
        if reg_cost is None:
            reg_cost = dict(type='BBoxL1Cost', weight=1.0)
        if cls_cost is None:
            cls_cost = dict(type='ClassificationCost', weight=1.)
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        if bev_iou_cost is not None:
            self.bev_iou_cost = build_match_cost(bev_iou_cost)
        else:
            self.bev_iou_cost = None
        self.pc_range = pc_range
        self.is_box_bev = is_box_bev

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.
        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.
        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.
        Args:
            bbox_pred (Tensor): Predicted boxes of shape (n_q, 10)
            cls_pred (Tensor): Predicted classification logits,
                shape (n_q, #cls)
            gt_bboxes (Tensor): Ground truth boxes of shape (n_gt, 7).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.
        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes,),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        if not self.is_box_bev:
            normalized_gt_bboxes = normalize_bbox(gt_bboxes, self.pc_range, is_height_norm=False)
            reg_cost = self.reg_cost(bbox_pred[:, :8],
                                     normalized_gt_bboxes[:, :8])

            # BEV IoU Cost
            # we need to get denormalized pred
            if self.bev_iou_cost:
                bbox_pred_world = denormalize_bbox(bbox_pred, self.pc_range,
                                                   is_height_norm=False)
                bbox_pred_world_bev = bbox_pred_world[..., [0, 1, 3, 4, 6]]
                bev_iou_cost = self.bev_iou_cost(bbox_pred_world_bev,
                                                 gt_bboxes[
                                                     ..., [0, 1, 3, 4, 6]])
            else:
                bev_iou_cost = 0.0
        else:
            normalized_gt_bboxes = normalize_bbox_bev(gt_bboxes)  # (n_gt, 6) log
            reg_cost = self.reg_cost(bbox_pred[:, :6],
                                     normalized_gt_bboxes[:, :6])

            # BEV IoU Cost
            # we need to get denormalized pred
            if self.bev_iou_cost:
                bbox_pred_world = denormalize_bbox_bev(bbox_pred)  # (n_p, 5) exp
                bev_iou_cost = self.bev_iou_cost(bbox_pred_world, gt_bboxes)
            else:
                bev_iou_cost = 0.0

        # weighted sum of above two costs
        cost = cls_cost + reg_cost + bev_iou_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
