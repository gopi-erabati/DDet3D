from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from .ddet3d import DDet3D


@DETECTORS.register_module()
class DDet3DKITTI(DDet3D):
    """
    DDet3D for Waymo dataset
    """
    def __init__(self, **kwargs):
        super(DDet3DKITTI, self).__init__(**kwargs)

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

        return bbox_results