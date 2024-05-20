# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import trimesh
from os import path as osp

from open3d import geometry
import cv2

from mmdet3d.core.points import get_points_type
from mmdet3d.core.visualizer.image_vis import (draw_camera_bbox3d_on_img,
                                               draw_depth_bbox3d_on_img,
                                               draw_lidar_bbox3d_on_img)


def _write_obj(points, out_filename):
    """Write points into ``obj`` format for meshlab visualization.

    Args:
        points (np.ndarray): Points in shape (N, dim).
        out_filename (str): Filename to be saved.
    """
    N = points.shape[0]
    fout = open(out_filename, 'w')
    for i in range(N):
        if points.shape[1] == 6:
            c = points[i, 3:].astype(int)
            fout.write(
                'v %f %f %f %d %d %d\n' %
                (points[i, 0], points[i, 1], points[i, 2], c[0], c[1], c[2]))

        else:
            fout.write('v %f %f %f\n' %
                       (points[i, 0], points[i, 1], points[i, 2]))
    fout.close()


def _write_oriented_bbox(scene_bbox, out_filename):
    """Export oriented (around Z axis) scene bbox to meshes.

    Args:
        scene_bbox(list[ndarray] or ndarray): xyz pos of center and
            3 lengths (dx,dy,dz) and heading angle around Z axis.
            Y forward, X right, Z upward. heading angle of positive X is 0,
            heading angle of positive Y is 90 degrees.
        out_filename(str): Filename.
    """

    def heading2rotmat(heading_angle):
        rotmat = np.zeros((3, 3))
        rotmat[2, 2] = 1
        cosval = np.cos(heading_angle)
        sinval = np.sin(heading_angle)
        rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
        return rotmat

    def convert_oriented_box_to_trimesh_fmt(box):
        ctr = box[:3]
        lengths = box[3:6]
        trns = np.eye(4)
        trns[0:3, 3] = ctr
        trns[3, 3] = 1.0
        trns[0:3, 0:3] = heading2rotmat(box[6])
        box_trimesh_fmt = trimesh.creation.box(lengths, trns)
        return box_trimesh_fmt

    if len(scene_bbox) == 0:
        scene_bbox = np.zeros((1, 7))
    scene = trimesh.scene.Scene()
    for box in scene_bbox:
        scene.add_geometry(convert_oriented_box_to_trimesh_fmt(box))

    mesh_list = trimesh.util.concatenate(scene.dump())
    # save to obj file
    trimesh.io.export.export_mesh(mesh_list, out_filename, file_type='obj')

    return


def show_result(points,
                gt_bboxes,
                pred_bboxes,
                out_dir,
                filename,
                show=False,
                snapshot=False,
                pred_labels=None,
                gt_labels=None):
    """Convert results into format that is directly readable for meshlab.

    Args:
        points (np.ndarray): Points.
        gt_bboxes (np.ndarray): Ground truth boxes.
        pred_bboxes (np.ndarray): Predicted boxes.
        out_dir (str): Path of output directory
        filename (str): Filename of the current frame.
        show (bool, optional): Visualize the results online. Defaults to False.
        snapshot (bool, optional): Whether to save the online results.
            Defaults to False.
        pred_labels (np.ndarray, optional): Predicted labels of boxes.
            Defaults to None.
        gt_labels (np.ndarray, optional): Ground truth labels of boxes.
            Defaults to None.
    """
    result_path = osp.join(out_dir, filename)
    # mmcv.mkdir_or_exist(result_path)

    if show:
        from mmdet3d.core.visualizer.open3d_vis import Visualizer

        vis = Visualizer(points)
        if pred_bboxes is not None:
            if pred_labels is None:
                vis.add_bboxes(bbox3d=pred_bboxes)
            else:
                # palette = np.random.randint(
                #     0, 255, size=(pred_labels.max() + 1, 3)) / 256
                palette = Colors()
                labelDict = {}
                for j in range(len(pred_labels)):
                    i = int(pred_labels[j].numpy())
                    if labelDict.get(i) is None:
                        labelDict[i] = []
                    labelDict[i].append(pred_bboxes[j])
                for i in labelDict:
                    vis.add_bboxes(
                        bbox3d=np.array(labelDict[i]),
                        bbox_color=(0, 0, 1))  #palette(i)

        if gt_bboxes is not None:
            if gt_labels is None:
                vis.add_bboxes(bbox3d=gt_bboxes, bbox_color=(0, 0, 1))
            else:
                palette = Colors()
                labelDict = {}
                for j in range(len(gt_labels)):
                    i = int(gt_labels[j])
                    if labelDict.get(i) is None:
                        labelDict[i] = []
                    labelDict[i].append(gt_bboxes[j])
                for i in labelDict:
                    vis.add_bboxes(
                        bbox3d=np.array(labelDict[i]),
                        bbox_color=(0, 1, 0))  #palette(i)
        show_path = osp.join(result_path,
                             f'{filename}_online.png') if snapshot else None
        vis.show(show_path)

    # if points is not None:
    #     _write_obj(points, osp.join(result_path, f'{filename}_points.obj'))
    #
    # if gt_bboxes is not None:
    #     # bottom center to gravity center
    #     gt_bboxes[..., 2] += gt_bboxes[..., 5] / 2
    #     # the positive direction for yaw in meshlab is clockwise
    #     gt_bboxes[:, 6] *= -1
    #     _write_oriented_bbox(gt_bboxes,
    #                          osp.join(result_path, f'{filename}_gt.obj'))
    #
    # if pred_bboxes is not None:
    #     # bottom center to gravity center
    #     pred_bboxes[..., 2] += pred_bboxes[..., 5] / 2
    #     # the positive direction for yaw in meshlab is clockwise
    #     pred_bboxes[:, 6] *= -1
    #     _write_oriented_bbox(pred_bboxes,
    #                          osp.join(result_path, f'{filename}_pred.obj'))


def show_seg_result(points,
                    gt_seg,
                    pred_seg,
                    out_dir,
                    filename,
                    palette,
                    ignore_index=None,
                    show=True,
                    snapshot=False):
    """Convert results into format that is directly readable for meshlab.

    Args:
        points (np.ndarray): Points.
        gt_seg (np.ndarray): Ground truth segmentation mask.
        pred_seg (np.ndarray): Predicted segmentation mask.
        out_dir (str): Path of output directory
        filename (str): Filename of the current frame.
        palette (np.ndarray): Mapping between class labels and colors.
        ignore_index (int, optional): The label index to be ignored, e.g. \
            unannotated points. Defaults to None.
        show (bool, optional): Visualize the results online. Defaults to False.
        snapshot (bool, optional): Whether to save the online results. \
            Defaults to False.
    """
    # we need 3D coordinates to visualize segmentation mask
    if gt_seg is not None or pred_seg is not None:
        assert points is not None, \
            '3D coordinates are required for segmentation visualization'

    # filter out ignored points
    if gt_seg is not None and ignore_index is not None:
        if points is not None:
            points = points[gt_seg != ignore_index]
        if pred_seg is not None:
            pred_seg = pred_seg[gt_seg != ignore_index]
        gt_seg = gt_seg[gt_seg != ignore_index]

    if gt_seg is not None:
        gt_seg_color = palette[gt_seg]
        gt_seg_color = np.concatenate([points[:, :3], gt_seg_color], axis=1)
    if pred_seg is not None:
        pred_seg_color = palette[pred_seg]
        pred_seg_color = np.concatenate([points[:, :3], pred_seg_color],
                                        axis=1)

    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    # online visualization of segmentation mask
    # we show three masks in a row, scene_points, gt_mask, pred_mask
    if show:
        from mmdet3d.core.visualizer.open3d_vis import Visualizer
        mode = 'xyzrgb' if points.shape[1] == 6 else 'xyz'
        vis = Visualizer(points, mode=mode)
        if gt_seg is not None:
            vis.add_seg_mask(gt_seg_color)
        if pred_seg is not None:
            vis.add_seg_mask(pred_seg_color)
        show_path = osp.join(result_path,
                             f'{filename}_online.png') if snapshot else None
        vis.show(show_path)

    if points is not None:
        _write_obj(points, osp.join(result_path, f'{filename}_points.obj'))

    if gt_seg is not None:
        _write_obj(gt_seg_color, osp.join(result_path, f'{filename}_gt.obj'))

    if pred_seg is not None:
        _write_obj(pred_seg_color, osp.join(result_path,
                                            f'{filename}_pred.obj'))


def show_multi_modality_result(img,
                               gt_bboxes,
                               pred_bboxes,
                               proj_mat,
                               out_dir,
                               filename,
                               box_mode='lidar',
                               img_metas=None,
                               show=True,
                               gt_bbox_color=(61, 102, 255),
                               pred_bbox_color=(241, 101, 72),
                               pred_labels=None,
                               gt_labels=None,
                               view=None,
                               save=True):
    """Convert multi-modality detection results into 2D results.

    Project the predicted 3D bbox to 2D image plane and visualize them.

    Args:
        img (np.ndarray): The numpy array of image in cv2 fashion.
        gt_bboxes (:obj:`BaseInstance3DBoxes`): Ground truth boxes.
        pred_bboxes (:obj:`BaseInstance3DBoxes`): Predicted boxes.
        proj_mat (numpy.array, shape=[4, 4]): The projection matrix
            according to the camera intrinsic parameters.
        out_dir (str): Path of output directory.
        filename (str): Filename of the current frame.
        box_mode (str): Coordinate system the boxes are in. Should be one of
           'depth', 'lidar' and 'camera'. Defaults to 'lidar'.
        img_metas (dict): Used in projecting depth bbox.
        show (bool): Visualize the results online. Defaults to False.
        gt_bbox_color (str or tuple(int)): Color of bbox lines.
           The tuple of color should be in BGR order. Default: (255, 102, 61)
        pred_bbox_color (str or tuple(int)): Color of bbox lines.
           The tuple of color should be in BGR order. Default: (72, 101, 241)
    """
    if box_mode == 'depth':
        draw_bbox = draw_depth_bbox3d_on_img
    elif box_mode == 'lidar':
        draw_bbox = draw_lidar_bbox3d_on_img
    elif box_mode == 'camera':
        draw_bbox = draw_camera_bbox3d_on_img
    else:
        raise NotImplementedError(f'unsupported box mode {box_mode}')

    result_path = osp.join(out_dir, filename)
    mmcv.mkdir_or_exist(result_path)

    if show:
        show_img = img.copy()
        if gt_bboxes is not None:
            # if there are no gt boxes, then don't draw anything
            if gt_bboxes.tensor.size(0) == 0:
                show_img = img.copy().astype(np.uint8)
            else:
                if gt_labels is None:
                    show_img = draw_bbox(
                        gt_bboxes, show_img, proj_mat, img_metas,
                        color=gt_bbox_color)
                else:
                    palette = Colors()
                    labelDict = {}
                    for j in range(len(gt_labels)):
                        i = int(gt_labels[j])
                        if labelDict.get(i) is None:
                            labelDict[i] = []
                        labelDict[i].append(j)
                    for i in labelDict:
                        show_img = draw_bbox(gt_bboxes[labelDict[i]], show_img,
                                             proj_mat, img_metas,
                                             color=palette(i, bgr=True))

        if pred_bboxes is not None:
            # if there are no pred boxes, then don't draw anything
            if pred_bboxes.tensor.size(0) == 0:
                pass
            else:
                if pred_labels is None:
                    show_img = draw_bbox(
                        pred_bboxes,
                        show_img,
                        proj_mat,
                        img_metas,
                        color=pred_bbox_color)
                else:
                    palette = Colors()
                    labelDict = {}
                    for j in range(len(pred_labels)):
                        i = int(pred_labels[j].numpy())
                        if labelDict.get(i) is None:
                            labelDict[i] = []
                        labelDict[i].append(j)
                    for i in labelDict:
                        show_img = draw_bbox(pred_bboxes[labelDict[i]],
                                             show_img,
                                             proj_mat, img_metas,
                                             color=palette(i, bgr=True))
        # resize to show small
        rescale_factor = 0.35
        show_img = mmcv.imrescale(show_img, rescale_factor)
        # show at different locations
        img_width, img_height = 1600, 900
        x_start, y_start = 512, 30
        img_titlebar_h = 30
        view_num = int(view.split('_')[1])
        if view_num == 1:
            location_xy = (x_start, y_start)
            window_name = 'CAM_FRONT'
        elif view_num == 2:
            location_xy = (x_start + int(img_width * rescale_factor), y_start)
            window_name = 'CAM_FRONT_RIGHT'
        elif view_num == 3:
            location_xy = (x_start + int(img_width * rescale_factor * 2),
                           y_start)
            window_name = 'CAM_FRONT_LEFT'
        elif view_num == 4:
            location_xy = (x_start, y_start + int(img_height * rescale_factor)
                           + img_titlebar_h)
            window_name = 'CAM_BACK'
        elif view_num == 5:
            location_xy = (x_start + int(img_width * rescale_factor),
                           y_start + int(img_height * rescale_factor)
                           + img_titlebar_h)
            window_name = 'CAM_BACK_LEFT'
        elif view_num == 6:
            location_xy = (x_start + int(img_width * rescale_factor * 2),
                           y_start + int(img_height * rescale_factor)
                           + img_titlebar_h)
            window_name = 'CAM_BACK_RIGHT'
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(window_name, *location_xy)
        cv2.imshow(window_name, show_img)
        cv2.waitKey()
        # mmcv.imshow(show_img, win_name=f'{view}', wait_time=0)

    if save:
        if img is not None:
            pass
            # rescale_factor = 0.35
            # gt_img = mmcv.imrescale(gt_img, rescale_factor)
            # mmcv.imwrite(img, osp.join(result_path, f'{view}_img.png'))

        if gt_bboxes is not None:
            # if there are no gt boxes, then don't draw anything
            if gt_bboxes.tensor.size(0) == 0:
                gt_img = img.copy()
            else:
                if gt_labels is None:
                    gt_img = draw_bbox(
                        gt_bboxes, img, proj_mat, img_metas,
                        color=gt_bbox_color)
                else:
                    palette = Colors()
                    labelDict = {}
                    for j in range(len(gt_labels)):
                        i = int(gt_labels[j])
                        if labelDict.get(i) is None:
                            labelDict[i] = []
                        labelDict[i].append(j)
                    gt_img = img.copy()
                    for i in labelDict:
                        gt_img = draw_bbox(gt_bboxes[labelDict[i]], gt_img,
                                           proj_mat, img_metas,
                                           color=palette(i, bgr=True),
                                           thickness=2)
            # rescale_factor = 0.25
            # gt_img = mmcv.imrescale(gt_img, rescale_factor)
            mmcv.imwrite(gt_img, osp.join(result_path, f'{view}_gt.png'))

        if pred_bboxes is not None:
            # if there are no pred boxes, then don't draw anything
            if pred_bboxes.tensor.size(0) == 0:
                pred_img = img.copy()
            else:
                if pred_labels is None:
                    pred_img = draw_bbox(
                        pred_bboxes,
                        img,
                        proj_mat,
                        img_metas,
                        color=pred_bbox_color)
                else:
                    palette = Colors()
                    labelDict = {}
                    for j in range(len(pred_labels)):
                        i = int(pred_labels[j].numpy())
                        if labelDict.get(i) is None:
                            labelDict[i] = []
                        labelDict[i].append(j)
                    pred_img = img.copy()
                    for i in labelDict:
                        pred_img = draw_bbox(pred_bboxes[labelDict[i]],
                                             pred_img,
                                             proj_mat, img_metas,
                                             color=palette(i, bgr=True),
                                             thickness=2)
            # rescale_factor = 0.25
            # pred_img = mmcv.imrescale(pred_img, rescale_factor)
            mmcv.imwrite(pred_img, osp.join(result_path, f'{view}_pred.png'))


def show_bev_result(points, coord_type,
                    gt_bboxes, pred_bboxes,
                    out_dir, filename, show,
                    pred_labels=None, gt_labels=None,
                    gt_bbox_color=(61, 102, 255),
                    pred_bbox_color=(241, 101, 72),
                    save=True,
                    voxel_size=0.2,
                    bev_img_size=512,
                    img_num=1,
                    ):
    # result_path = osp.join(out_dir, filename)
    # mmcv.mkdir_or_exist(result_path)

    #  #BEV Projection of points and boxes; show and save
    # convert points to LiDARPoints
    points_class = get_points_type(coord_type)
    points = points_class(points, points_dim=points.shape[-1])
    points_mask = points.in_range_3d(
        [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])  # filter
    points = points[points_mask]
    points = points.tensor.numpy()
    points = points[:, 0:3] + [51.2, 51.2, 0.0]  # make them 0 tp
    # 102.4
    # voxel_size = 0.2
    points = points / voxel_size  # divide by voxel size
    points = points.astype(np.int)  # convert to int

    bev_img = np.ones((bev_img_size, bev_img_size, 3), dtype=np.uint8) * 255
    bev_img[points[:, 0], points[:, 1]] = 128

    # show
    if show:
        if gt_bboxes is not None:
            # if there are no gt boxes, then don't draw anything
            if gt_bboxes.tensor.size(0) == 0:
                bev_img_show_gt = bev_img.copy()
            else:
                bev_img_show_gt = bev_img.copy()
                # gt_corners = gt_bboxes.corners.numpy()  # (N, 8, 3)
                gt_corners = get_bbox_corners(gt_bboxes.tensor.numpy())
                gt_corners = gt_corners + [51.2, 51.2, 0.0]
                gt_corners = gt_corners / voxel_size
                gt_corners = gt_corners.astype(np.int)
                gt_corners_show = gt_corners[:, [0, 1, 4, 2], 0:2]  # (N, 4, 2)
                if gt_labels is None:
                    bev_img_show_gt = draw_bev_boxes(gt_corners_show,
                                                     bev_img_show_gt,
                                                     color=gt_bbox_color)
                else:
                    palette = Colors()
                    labelDict = {}
                    for j in range(len(gt_labels)):
                        i = int(gt_labels[j])
                        if labelDict.get(i) is None:
                            labelDict[i] = []
                        labelDict[i].append(j)
                    for i in labelDict:
                        bev_img_show_gt = draw_bev_boxes(gt_corners_show[
                                                             labelDict[i]],
                                                         bev_img_show_gt,
                                                         # palette(i, bgr=True),
                                                         (0, 255, 0),
                                                         thickness=1)
        # # bev_img_show_gt = mmcv.imrescale(bev_img_show_gt, 0.7)
        # bev_img_show_gt = cv2.putText(bev_img_show_gt, str(img_num), (400, 50), cv2.FONT_HERSHEY_SIMPLEX,
        #                   1, (0, 255, 0), 1, cv2.LINE_4, False)
        # window_name = 'LIDAR_BEV_GT'
        # location_xy = (400, 100)
        # cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        # cv2.moveWindow(window_name, *location_xy)
        # cv2.imshow(window_name, bev_img_show_gt)
        # cv2.waitKey(300)
        # # mmcv.imshow(bev_img_show_gt, win_name='bev_gt', wait_time=0)

        # pred to bev
        if pred_bboxes is not None:
            # if there are no pred boxes, then don't draw anything
            if pred_bboxes.tensor.size(0) == 0:
                bev_img_show_pred = bev_img.copy()
            else:
                bev_img_show_pred = bev_img.copy()
                # pred_corners = pred_bboxes.corners.numpy()  # (N, 8, 3)
                pred_corners = get_bbox_corners(pred_bboxes.tensor.numpy())
                pred_corners += [51.2, 51.2, 0.0]
                pred_corners = pred_corners / voxel_size
                pred_corners = pred_corners.astype(np.int)
                pred_corners_show = pred_corners[:, [0, 1, 4, 2], 0:2]  # (N,
                # 4, 2)
                if pred_labels is None:
                    bev_img_show_pred = draw_bev_boxes(pred_corners_show,
                                                       bev_img_show_pred,
                                                       color=pred_bbox_color)
                else:
                    palette = Colors()
                    labelDict = {}
                    for j in range(len(pred_labels)):
                        i = int(pred_labels[j].numpy())
                        if labelDict.get(i) is None:
                            labelDict[i] = []
                        labelDict[i].append(j)
                    for i in labelDict:
                        bev_img_show_gt = draw_bev_boxes(pred_corners_show[
                                                               labelDict[i]],
                                                           bev_img_show_gt,
                                                           # palette(i,
                                                           #         bgr=True),
                                                         (255, 0, 0),
                                                         thickness=1)
        # bev_img_show_pred = mmcv.imrescale(bev_img_show_pred, 0.7)
        bev_img_show_gt = cv2.putText(bev_img_show_gt, str(img_num), (400, 50), cv2.FONT_HERSHEY_SIMPLEX,
                          1, (0, 255, 0), 1, cv2.LINE_4, False)
        window_name = 'LIDAR_BEV_PRED'
        location_xy = (400 + 600, 100)
        location_xy = (600, 100)
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.moveWindow(window_name, *location_xy)
        cv2.imshow(window_name, bev_img_show_gt)
        cv2.waitKey(500)
        # mmcv.imshow(bev_img_show_pred, win_name='bev_pred', wait_time=0)

    if save:
        # mmcv.imwrite(bev_img, osp.join(result_path, f'bev_img.png'))

        # convert boxes to BEV
        # gt
        if gt_bboxes is not None:
            # if there are no gt boxes, then don't draw anything
            if gt_bboxes.tensor.size(0) == 0:
                gt_img = bev_img.copy()
            else:
                # gt_corners = gt_bboxes.corners.numpy()  # (N, 8, 3)
                gt_corners = get_bbox_corners(gt_bboxes.tensor.numpy())
                gt_corners += [51.2, 51.2, 0.0]
                gt_corners = gt_corners / voxel_size
                gt_corners = gt_corners.astype(np.int)
                gt_corners_show = gt_corners[:, [0, 1, 4, 2], 0:2]  # (N, 4, 2)
                if gt_labels is None:
                    gt_img = draw_bev_boxes(gt_corners_show, bev_img,
                                            color=gt_bbox_color)
                else:
                    palette = Colors()
                    labelDict = {}
                    for j in range(len(gt_labels)):
                        i = int(gt_labels[j])
                        if labelDict.get(i) is None:
                            labelDict[i] = []
                        labelDict[i].append(j)
                    gt_img = bev_img.copy()
                    for i in labelDict:
                        gt_img = draw_bev_boxes(gt_corners_show[
                                                    labelDict[i]],
                                                gt_img, (0, 255, 0),
                                                # palette(i, bgr=True)
                                                thickness=1)
            # gt_img = mmcv.imresize(gt_img, (400, 225))
            mmcv.imwrite(gt_img, osp.join(out_dir, f'bev_gt_{filename}.png'))

        # pred to bev
        if pred_bboxes is not None:
            # if there are no pred boxes, then don't draw anything
            if pred_bboxes.tensor.size(0) == 0:
                pred_img = bev_img.copy()
            else:
                # pred_corners = pred_bboxes.corners.numpy()  # (N, 8, 3)
                pred_corners = get_bbox_corners(pred_bboxes.tensor.numpy())
                pred_corners += [51.2, 51.2, 0.0]
                pred_corners = pred_corners / voxel_size
                pred_corners = pred_corners.astype(np.int)
                pred_corners_show = pred_corners[:, [0, 1, 4, 2],
                                    0:2]  # (N, 4, 2)
                if pred_labels is None:
                    pred_img = draw_bev_boxes(pred_corners_show, bev_img,
                                              color=pred_bbox_color)
                else:
                    palette = Colors()
                    labelDict = {}
                    for j in range(len(pred_labels)):
                        i = int(pred_labels[j].numpy())
                        if labelDict.get(i) is None:
                            labelDict[i] = []
                        labelDict[i].append(j)
                    pred_img = bev_img.copy()
                    for i in labelDict:
                        gt_img = draw_bev_boxes(pred_corners_show[
                                                      labelDict[i]],
                                                  gt_img, (255, 0, 0),
                                                  # palette(i, bgr=True)
                                                  thickness=1)
            # pred_img = mmcv.imresize(pred_img, (400, 225))
            mmcv.imwrite(gt_img, osp.join(out_dir, f'bev_gt_{filename}.png'))


def draw_bev_boxes(gt_bboxes, img, color, thickness=1):
    if gt_bboxes.ndim != 3:
        gt_bboxes = np.expand_dims(gt_bboxes, axis=0)  # (1, 4, 2)
    line_indices = ((0, 1), (1, 2), (2, 3), (3, 0))
    for i in range(gt_bboxes.shape[0]):
        corners = gt_bboxes[i].astype(np.int)
        for start, end in line_indices:
            cv2.line(img, (corners[start, 1], corners[start, 0]),
                     (corners[end, 1], corners[end, 0]), color,
                     thickness, cv2.LINE_AA)

    return img.astype(np.uint8)


def get_bbox_corners(bbox3d):
    """

    bbox3d: numpy array (N, 9)
    returns:
        corners of shape (N, 8, 3)
    """
    rot_axis = 2
    center_mode = 'lidar+bottom'
    bbox_corners = []
    for i in range(len(bbox3d)):
        center = bbox3d[i, 0:3]
        dim = bbox3d[i, 3:6]
        yaw = np.zeros(3)
        yaw[rot_axis] = -bbox3d[i, 6]
        rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)

        if center_mode == 'lidar_bottom':
            center[rot_axis] += dim[
                                    rot_axis] / 2  # bottom center to gravity center
        elif center_mode == 'camera_bottom':
            center[rot_axis] -= dim[
                                    rot_axis] / 2  # bottom center to gravity center
        box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)
        box_points = box3d.get_box_points()
        # gives open3d.utility.Vector3dVector
        box_points = np.asarray(box_points)
        bbox_corners.append(box_points)
    bbox_corners = np.stack(bbox_corners)

    return bbox_corners


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        # hex = (
        #     'FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A',
        #     '92CC17',
        #     '3DDB86', '1A9334', '00D4BB',
        #     '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF',
        #     '520085',
        #     'CB38FF', 'FF95C8', 'FF37C7')
        # (light Red, very light red, orange, vivid orange, shade of green,
        # vivid green,
        # strong green, cyan, dark green, strong cyan,)
        # (dark cyan, pure cyan, dark blue, light blue, pure blue,
        # light voilet, dark voilet, purple, light pink, pink)
        hex = ('FF3838', 'FFB21D', '48F90A', '00D4BB', '344593', '6473FF',
               'CB38FF', 'FF37C7', '800000', '808000')
        # (light red, orange, green, cyan, dark blue, light blue, purple,
        # pink, maroon, olive)
        # 'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
        #     'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

# Car - Red
# Truck - Orange
# Trailer - Green
# Bus - Cyan
# ConstVeh - Dark Blue
# Bicycle - Light blue
# Motorcycle - Purple
# Pedestrian - Pink
# Cone - maroon
# Barrier - Olive
