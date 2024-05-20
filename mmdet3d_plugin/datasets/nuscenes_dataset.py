import numpy as np
from os import path as osp

from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.core.bbox import Box3DMode, Coord3DMode

from ..core.visualizer import (show_result, show_multi_modality_result,
                               show_bev_result)


@DATASETS.register_module()
class CustomNuScenesDataset(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            timestamp=info['timestamp'] / 1e6,
        )

        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                                  'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    @staticmethod
    def are_points_in_image(points, lidar2img, img_metas):
        """

        points (np.ndarray) points of shape (N, 3)
        lidar2img (np.ndarray): LiDAR to Camera transformation matrix (4, 4)
        img_metas (dict):

        Return:
            mask (np.array) mask of points inside image (N, )
        """

        points = np.concatenate((points, np.ones_like(
            points[..., :1])), axis=-1).transpose()
        # (4, n_gt)
        points_cam = np.matmul(lidar2img, points).transpose()
        # (n_gt, 4)
        eps = 1e-5
        mask = (points_cam[..., 2:3] > eps)  # (n_gt, 1)
        points_cam = points_cam[..., 0:2] / np.maximum(
            points_cam[..., 2:3],
            np.ones_like(points_cam[..., 2:3]) * eps)
        # (n_gt, 2)
        points_cam[..., 0] /= img_metas['img_shape'][1]
        points_cam[..., 1] /= img_metas['img_shape'][0]

        points_cam = (points_cam - 0.5) * 2
        mask = (mask & (points_cam[..., 0:1] > -1.0)
                & (points_cam[..., 0:1] < 1.0)
                & (points_cam[..., 1:2] > -1.0)
                & (points_cam[..., 1:2] < 1.0))
        mask = mask.reshape(-1)
        # (n_gt, )
        return mask

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline_compose = self._get_pipeline(pipeline)
        show_threshold = 0.2
        save_imgs = True
        from tqdm import tqdm
        for i, result in tqdm(enumerate(results)):
            # if i not in [167, 457, 464, 493, 770, 857, 976, 1141, 1154,
            #              1157, 1191, 1307, 1391, 1393, 1423, 1629, 2022,
            #              2183, 2211, 2254, 2565, 2711, 2721, 2730, 2911,
            #              3609, 3618, 3669, 3824, 4135, 4173, 4613, 4618,
            #              4630, 4648, 4897, 4941, 5434, 5640, 5736, 5835, 5845]:
            #     continue
            if i not in [75, 121]:
                continue
            print(i)
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]

            # Get Points and Images
            points, img, img_metas = self._extract_data(i, pipeline_compose,
                                                        ['points', 'img',
                                                         'img_metas'])
            points_org = points.numpy()

            # Show boxes on point cloud
            # for now we convert points into depth mode
            points_depth = Coord3DMode.convert_point(points_org,
                                                     Coord3DMode.LIDAR,
                                                     Coord3DMode.DEPTH)

            # Get GT Boxes and Filter by range and name
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d']
            gt_labels = self.get_ann_info(i)['gt_labels_3d']
            mask = gt_bboxes.in_range_bev([-51.2, -51.2, 51.2, 51.2])
            gt_bboxes = gt_bboxes[mask]
            gt_bboxes.limit_yaw(offset=0.5, period=2 * np.pi)
            gt_labels = gt_labels[mask.numpy().astype(np.bool)]
            # name filtering
            labels = list(range(10))
            gt_bboxes_mask = np.array([n in labels for n in gt_labels],
                                      dtype=np.bool_)
            gt_bboxes = gt_bboxes[gt_bboxes_mask]
            gt_labels = gt_labels[gt_bboxes_mask]

            # Convert GT boxes to Depth Mode to show with Visualizer
            gt_bboxes_numpy = gt_bboxes.tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes_numpy, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)

            # Get Prediction Boxes
            inds = result['scores_3d'] > show_threshold
            pred_bboxes = result['boxes_3d'][inds]
            pred_bboxes_numpy = pred_bboxes.tensor.numpy()
            pred_labels = result['labels_3d'][inds]
            show_pred_bboxes = Box3DMode.convert(pred_bboxes_numpy, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points_depth, show_gt_bboxes, show_pred_bboxes,
                        out_dir,
                        file_name, show, pred_labels=pred_labels,
                        gt_labels=gt_labels)

            # BEV Show and Save
            # show_bev_result(points_org, coord_type=pipeline[0]['coord_type'],
            #                 gt_bboxes=gt_bboxes, pred_bboxes=pred_bboxes,
            #                 out_dir=out_dir, filename=str(i), show=show,
            #                 pred_labels=pred_labels, gt_labels=gt_labels,
            #                 save=save_imgs, voxel_size=0.1, bev_img_size=1024,
            #                 img_num=i)

            # Show boxes on Image
            if self.modality['use_camera'] and 'lidar2img' in img_metas.keys():
                img = img.numpy()
                # img is (n_views, H, W, C)
                # need to transpose channel to last dim
                img = img.transpose(0, 2, 3, 1)

                gt_bboxes_center = gt_bboxes.bottom_center.numpy()  # (n_gt, 3)

                pred_bboxes_center = pred_bboxes.bottom_center.numpy()

                lidar2img = img_metas['lidar2img']  # (n_views, 4, 4)

                # for each view
                for idx, (img_view, lidar2img_view) in enumerate(zip(img,
                                                                     lidar2img)):
                    # check GT center in image view and apply mask to bboxes
                    gt_center_mask = self.are_points_in_image(
                        gt_bboxes_center, lidar2img_view, img_metas)
                    # (n_gt, )
                    gt_bboxes_view = gt_bboxes[gt_center_mask]
                    gt_labels_view = gt_labels[gt_center_mask]

                    # check Pred center in image view and apply mask to bboxes
                    pred_center_mask = self.are_points_in_image(
                        pred_bboxes_center, lidar2img_view, img_metas)
                    # (n_pred, )
                    pred_bboxes_view = pred_bboxes[pred_center_mask]
                    pred_labels_view = pred_labels[pred_center_mask]

                    filename_view = 'v_' + str(idx + 1)

                    show_multi_modality_result(
                        img_view,
                        gt_bboxes_view,
                        pred_bboxes_view,
                        lidar2img_view,
                        out_dir,
                        str(i),
                        box_mode='lidar',
                        show=False,
                        pred_labels=pred_labels_view,
                        gt_labels=gt_labels_view,
                        view=filename_view,
                        save=save_imgs
                    )
