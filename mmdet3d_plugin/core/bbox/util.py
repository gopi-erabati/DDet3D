import torch
from mmdet3d.core.bbox.box_np_ops import rotation_3d_in_axis


def normalize_bbox(bboxes, pc_range, is_height_norm=True):
    """ Normalize the bbox
    Normalize the center, log the size, sincos the rot

    Args:
        bboxes (Tensor): (n_p, 9) with vx and vy (or) (n_p, 7)

    Returns:
         Tensor of shape (n_p, 10) or (n_p, 8)
    """
    center = bboxes[..., 0:3]  # (n_p, 3)

    # # Normalize center to [0, 1]
    # # center normalize
    # pc_range_ = bboxes.new_tensor([[pc_range[3] - pc_range[0],
    #                                 pc_range[4] - pc_range[1],
    #                                 pc_range[5] - pc_range[2]]])  # (1, 3)
    # pc_start_ = bboxes.new_tensor(
    #     [[pc_range[0], pc_range[1], pc_range[2]]])  # (1, 3)
    # bbox_center = (center - pc_start_) / pc_range_  # (n_p, 3)

    if is_height_norm:
        size = bboxes[..., 3:6].log()  # (n_p, 3)
    else:
        size = torch.cat([bboxes[..., 3:5].log(), bboxes[..., 5:6]], dim=-1)

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (center, size, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (center, size, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes  # (n_p, 10) or (n_p, 8)


def normalize_bbox_bev(bboxes):
    """ Normalize the bbox
    Normalize the center, log the size, sincos the rot

    Args:
        bboxes (Tensor): (n_p, 5) xywhr

    Returns:
         Tensor of shape (n_p, 6)
    """
    center = bboxes[..., 0:2]  # (n_p, 2)
    size = bboxes[..., 2:4].log()  # (n_p, 2)
    rot = bboxes[..., 4:5]

    normalized_bboxes = torch.cat(
            (center, size, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes  # (n_p, 6)


def normalize_0to1_bbox(bboxes, pc_range, size_norm):
    center = bboxes[..., 0:3]  # (n_p, 3)
    size = bboxes[..., 3:6]  # (n_p, 3)
    rot_sin = bboxes[..., 6:7].sin()  # (n_p, 1)
    rot_cos = bboxes[..., 6:7].cos()  # (n_p, 1)

    # center normalize
    pc_range_ = bboxes.new_tensor([[pc_range[3] - pc_range[0],
                                    pc_range[4] - pc_range[1],
                                    pc_range[5] - pc_range[2]]])  # (1, 3)
    pc_start_ = bboxes.new_tensor(
        [[pc_range[0], pc_range[1], pc_range[2]]])  # (1, 3)
    bbox_center = (center - pc_start_) / pc_range_  # (n_p, 3)

    # size normalize
    bbox_max_size = bboxes.new_tensor(size_norm)  # max 20.0 m
    # objects assumption
    bbox_size = size / bbox_max_size  # (n_p, 3)

    # rot normalize
    rot_sin = (rot_sin - (-1.0)) / 2.0  # (n_p, 1)
    rot_cos = (rot_cos - (-1.0)) / 2.0  # (n_p, 1)

    # concatenate all
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8]  # (n_p, 1)
        vy = bboxes[..., 8:9]  # (n_p, 1)
        norm0to1_bboxes = torch.cat(
            (bbox_center, bbox_size, rot_sin, rot_cos, vx, vy), dim=-1)
    else:
        norm0to1_bboxes = torch.cat(
            (bbox_center, bbox_size, rot_sin, rot_cos), dim=-1)
    return norm0to1_bboxes  # (n_p, 10)


def normalize_0to1_bbox_bev(bboxes, pc_range):
    """ bboxes: (N, 5) of form XYXYR
    size_norm: (2, ) of XY
    """
    pc_range_ = bboxes.new_tensor([[pc_range[3] - pc_range[0],
                                    pc_range[4] - pc_range[1],
                                    pc_range[3] - pc_range[0],
                                    pc_range[4] - pc_range[1],]])  # (1, 4)
    pc_start_ = bboxes.new_tensor(
        [[pc_range[0], pc_range[1], pc_range[0], pc_range[1]]])  # (1, 4)
    bboxes_norm = (bboxes[..., 0:4] - pc_start_) / pc_range_  # (n_p, 4)

    rot_sin = bboxes[..., 4:5].sin()  # (n_p, 1)
    rot_cos = bboxes[..., 4:5].cos()  # (n_p, 1)
    # rot normalize
    rot_sin = (rot_sin - (-1.0)) / 2.0  # (n_p, 1)
    rot_cos = (rot_cos - (-1.0)) / 2.0  # (n_p, 1)

    bboxes_norm = torch.cat((bboxes_norm, rot_sin, rot_cos), dim=-1)
    # (n_p, 6)

    return bboxes_norm


def denormalize_bbox(normalized_bboxes, pc_range, is_height_norm=True):
    """ Denormlaize the bbox
    Denormalize the center, exp the size, convert sincos to rot

    Args:
        normalized_bboxes (Tensor): (n_p, 10) with vx and vy (or) (n_p, 8)
        pc_range (list): [xmin, ymin, zmin, xmax, ymax, zmax]

    Returns:
         Tensor of shape (n_p, 9) or (n_p, 7)
    """

    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center
    center = normalized_bboxes[..., 0:3]  # (n_p, 3)
    # # center denormalize
    # pc_range_ = normalized_bboxes.new_tensor([[pc_range[3] - pc_range[0],
    #                                            pc_range[4] - pc_range[1],
    #                                            pc_range[5] - pc_range[
    #                                                2]]])  # (1, 3)
    # pc_start_ = normalized_bboxes.new_tensor(
    #     [[pc_range[0], pc_range[1], pc_range[2]]])  # (1, 3)
    # bbox_center = (center * pc_range_) + pc_start_  # (n_p, 3)

    # size
    if is_height_norm:
        size = normalized_bboxes[..., 3:6].exp()  # (n_p, 3)
    else:
        size = torch.cat([normalized_bboxes[..., 3:5].exp(),
                          normalized_bboxes[..., 5:6]], dim=-1)

    if normalized_bboxes.size(-1) > 8:
        # velocity
        vx = normalized_bboxes[..., 8:9]
        vy = normalized_bboxes[..., 9:10]
        denormalized_bboxes = torch.cat([center, size, rot, vx, vy],
                                        dim=-1)
    else:
        denormalized_bboxes = torch.cat([center, size, rot], dim=-1)
    return denormalized_bboxes  # (n_p, 9) or (n_p, 7)


def denormalize_bbox_bev(normalized_bboxes):
    """ Denormlaize the bbox
    Denormalize the center, exp the size, convert sincos to rot

    Args:
        normalized_bboxes (Tensor): (n_p, 10) with vx and vy (or) (n_p, 8)
        pc_range (list): [xmin, ymin, zmin, xmax, ymax, zmax]

    Returns:
         Tensor of shape (n_p, 9) or (n_p, 7)
    """

    # rotation
    rot_sine = normalized_bboxes[..., 4:5]

    rot_cosine = normalized_bboxes[..., 5:6]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center
    center = normalized_bboxes[..., 0:2]  # (n_p, 2)
    size = normalized_bboxes[..., 2:4].exp()  # (n_p, 2)

    denormalized_bboxes = torch.cat([center, size, rot], dim=-1)
    return denormalized_bboxes  # (n_p, 5)


def denormalize_0to1_bbox(bboxes, pc_range, size_norm):
    center = bboxes[..., 0:3]  # (n_p, 3)
    size = bboxes[..., 3:6]  # (n_p, 3)
    rot_sin = bboxes[..., 6:7]  # (n_p, 1)
    rot_cos = bboxes[..., 7:8]  # (n_p, 1)

    # center denormalize
    pc_range_ = bboxes.new_tensor([[pc_range[3] - pc_range[0],
                                    pc_range[4] - pc_range[1],
                                    pc_range[5] - pc_range[2]]])  # (1, 3)
    pc_start_ = bboxes.new_tensor(
        [[pc_range[0], pc_range[1], pc_range[2]]])  # (1, 3)
    bbox_center = (center * pc_range_) + pc_start_  # (n_p, 3)

    # size normalize
    bbox_max_size = bboxes.new_tensor(
        size_norm)  # max 20.0 m objects assumption
    bbox_size = size * bbox_max_size  # (n_p, 3)

    # rot normalize
    rot_sin = (rot_sin * 2.0) + (-1.0)  # (n_p, 1)
    rot_cos = (rot_cos * 2.0) + (-1.0)  # (n_p, 1)

    # concatenate all
    if bboxes.size(-1) > 8:
        vx = bboxes[..., 8:9]  # (n_p, 1)
        vy = bboxes[..., 9:10]  # (n_p, 1)
        norm0to1_bboxes = torch.cat(
            (bbox_center, bbox_size, rot_sin, rot_cos, vx, vy), dim=-1)
    else:
        norm0to1_bboxes = torch.cat(
            (bbox_center, bbox_size, rot_sin, rot_cos), dim=-1)
    return norm0to1_bboxes


def denormalize_0to1_bbox_bev(bboxes, pc_range):
    """ bboxes: (N, 6) of form XYXYRR
    """
    pc_range_ = bboxes.new_tensor([[pc_range[3] - pc_range[0],
                                    pc_range[4] - pc_range[1],
                                    pc_range[3] - pc_range[0],
                                    pc_range[4] - pc_range[1],]])  # (1, 4)
    pc_start_ = bboxes.new_tensor(
        [[pc_range[0], pc_range[1], pc_range[0], pc_range[1]]])  # (1, 4)
    bbox = (bboxes[:, :4] * pc_range_) + pc_start_  # (n_p, 4)

    rot_sin = bboxes[..., 4:5]  # (n_p, 1)
    rot_cos = bboxes[..., 5:6]  # (n_p, 1)
    # rot denormalize
    rot_sin = (rot_sin * 2.0) + (-1.0)  # (n_p, 1)
    rot_cos = (rot_cos * 2.0) + (-1.0)  # (n_p, 1)

    ry = torch.atan2(rot_sin.clone(), rot_cos.clone())  # (n_p, 1)

    bboxes_norm = torch.cat((bbox, ry), dim=-1)
    # (n_p, 5)

    return bboxes_norm  # (n_p, 5) in XYXYR


def boxes3d_to_corners3d(boxes3d, bottom_center=True, ry=False):
    """Convert kitti center boxes to corners.

        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1

    Args:
        boxes3d (torch.tensor): Boxes with shape of (bs, N, 8) if ry=True
            or (bs, N, 7)
            cx, cy, cz, w, l,  h, sin(rot), cos(rot) in TOP LiDAR coords,
            see the definition of ry in nuScenes dataset.
        bottom_center (bool, optional): Whether z is on the bottom center
            of object. Defaults to True.
        ry (bool, optional): whether angle in ry or sincos format

    Returns:
        torch.tensor: Box corners with the shape of [bs,N, 8, 3].
    """

    bs = boxes3d.shape[0]
    boxes_num = boxes3d.shape[1]

    if ry:
        cx, cy, cz, w, l, h, ry = tuple(
            [boxes3d[:, :, i] for i
             in range(boxes3d.shape[2])])
    else:
        cx, cy, cz, w, l, h, sin_rot, cos_rot = tuple(
            [boxes3d[:, :, i] for i
             in range(boxes3d.shape[2])])
        # (bs, n_p)
        ry = torch.atan2(sin_rot.clone(), cos_rot.clone())  # (bs, n_p)

    w = w.exp()
    l = l.exp()
    h = h.exp()

    # w, l, h: (B,N)
    x_corners = torch.stack(
        [w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.],
        dim=2)  # (B,N,8)
    y_corners = torch.stack(
        [-l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2.],
        dim=2)  # .T
    if bottom_center:
        z_corners = torch.zeros((bs, boxes_num, 8), dtype=torch.float32).cuda()
        z_corners[:, :, 4:8] = torch.unsqueeze(h, 2).expand(bs, boxes_num,
                                                            4)  # (bs, N, 8)
    else:
        z_corners = torch.stack([
            -h / 2., -h / 2., -h / 2., -h / 2., h / 2., h / 2., h / 2., h / 2.
        ], dim=2)  # .T

    # ry = rot # (bs, N)
    zeros, ones = torch.zeros(
        ry.size(), dtype=torch.float32).cuda(), torch.ones(
        ry.size(), dtype=torch.float32).cuda()  # (bs, n_p)
    rot_list1 = torch.stack([torch.cos(ry), -torch.sin(ry), zeros], dim=0)
    # (3, bs, np)
    rot_list2 = torch.stack([torch.sin(ry), torch.cos(ry), zeros], dim=0)
    rot_list3 = torch.stack([zeros, zeros, ones], dim=0)
    # (3, bs, n_p)
    rot_list = torch.stack([rot_list1, rot_list2, rot_list3],
                           dim=0)  # (3, 3, bs, N)
    # (3, 3, bs, n_p)

    R_list = rot_list.permute(2, 3, 0, 1)  # (bs, n_p, 3, 3)

    temp_corners = torch.stack([x_corners, y_corners, z_corners],
                               dim=3)  # (bs, n_p, 8, 3)
    rotated_corners = torch.matmul(temp_corners, R_list)  # (bs, n_p, 8, 3)
    x_corners = rotated_corners[:, :, :, 0]  # (bs, n_p, 8, 1)
    y_corners = rotated_corners[:, :, :, 1]
    z_corners = rotated_corners[:, :, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, :, 0], boxes3d[:, :, 1], boxes3d[:, :,
                                                              2]  # (bs, n_p)

    x = torch.unsqueeze(x_loc, 2) + x_corners.reshape(-1, boxes_num,
                                                      8)  # (bs,n_p,8)
    y = torch.unsqueeze(y_loc, 2) + y_corners.reshape(-1, boxes_num, 8)
    z = torch.unsqueeze(z_loc, 2) + z_corners.reshape(-1, boxes_num, 8)

    corners = torch.stack(
        [x, y, z],
        dim=3)  # (bs, n_p, 8, 3)

    return corners.type(torch.float32)  # (bs, n_p, 8, 3)


def corners3d_to_boxes3d(corners3d):
    """
    Convert Corners to boxes

        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1

    Args:
        corners3d (Tensor): of shape (bs, n_p, 8, 3)

    """

    center = (corners3d[:, :, 0, :] + corners3d[:, :, 6, :]) / 2.0
    # points 0 and 6 is a diagonal, so midpoint of diagnoal is center
    # (bs, n_p, 3)

    sizex = (corners3d[:, :, 0, :] - corners3d[:, :, 1, :]).pow(2).sum(
        2).sqrt()  # (bs, n_p )
    sizey = (corners3d[:, :, 0, :] - corners3d[:, :, 3, :]).pow(2).sum(
        2).sqrt()  # (bs, n_p )
    sizez = (corners3d[:, :, 0, :] - corners3d[:, :, 4, :]).pow(2).sum(
        2).sqrt()  # (bs, n_p )
    size = torch.stack([sizex, sizey, sizez], dim=2)  # # (bs, n_p, 3 )

    # to get angle get the bottom center and midpoint of '0-3' and then
    # calculate angle
    bottom_center = (corners3d[:, :, 0, :2] + corners3d[:, :, 2, :2]) / 2.0
    # (bs, n_p, 2)
    mid03 = (corners3d[:, :, 0, :2] + corners3d[:, :, 3, :2]) / 2.0
    # (bs, n_p, 2)
    ry = torch.atan2(mid03[:, :, 1] - bottom_center[:, :, 1], mid03[:, :,
                                                              0] -
                     bottom_center[:, :, 0])  # (bs, n_p )
    # ry = torch.atan2(corners3d[:, :, 0, 1] - corners3d[:, :, 1, 1],
    #                  corners3d[:, :, 0, 0] - corners3d[:, :, 1, 0])
    # (bs, n_p )
    ry = ry.unsqueeze(2)  # (bs, n_p, 1)
    ry = -1.0 * ry

    boxes3d = torch.cat([center, size, ry], dim=2)  # (bs, n_p, 7 )

    return boxes3d


def xyxyr2xywhr(bboxes):
    """ convert boxes of shape (N, 6) XYXYRR to
    XYWHRR"""

    boxes_xywhr = torch.zeros_like(bboxes)

    boxes_xywhr[..., 0] = (bboxes[..., 0] + bboxes[..., 2]) / 2
    boxes_xywhr[..., 1] = (bboxes[..., 1] + bboxes[..., 3]) / 2
    boxes_xywhr[..., 2] = bboxes[..., 2] - bboxes[..., 0]
    boxes_xywhr[..., 3] = bboxes[..., 3] - bboxes[..., 1]

    if bboxes.size(-1) == 6:
        boxes_xywhr[..., 4:6] = bboxes[..., 4:6]
    else:
        boxes_xywhr[..., 4:5] = bboxes[..., 4:5]

    return boxes_xywhr  # (N, 6)


def xywhrr2xyxyrr(boxes_xywhr):
    """Convert a rotated boxes in XYWHR format to XYXYR format.

    Args:
        boxes_xywhr (torch.Tensor | np.ndarray): Rotated boxes in XYWHRR format.

    Returns:
        (torch.Tensor | np.ndarray): Converted boxes in XYXYRR format.
    """
    boxes = torch.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[..., 2] / 2
    half_h = boxes_xywhr[..., 3] / 2

    boxes[..., 0] = boxes_xywhr[..., 0] - half_w
    boxes[..., 1] = boxes_xywhr[..., 1] - half_h
    boxes[..., 2] = boxes_xywhr[..., 0] + half_w
    boxes[..., 3] = boxes_xywhr[..., 1] + half_h
    boxes[..., 4:6] = boxes_xywhr[..., 4:6]
    return boxes


def unravel_index(indices, shape):
    r"""Converts a tensor of flat indices into a tensor of coordinate vectors.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of flat indices, (*,).
        shape: The target shape.

    Returns:
        The unraveled coordinates, (*, D).
    """

    shape = indices.new_tensor((*shape, 1))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return torch.div(indices[..., None], coefs, rounding_mode='trunc') % shape[:-1]


def corners_nd(dims, origin=0.5):
    """Generate relative box corners based on length per dim and origin point.

    Args:
        dims (np.ndarray, shape=[N, ndim]): Array of length per dim
        origin (list or array or float, optional): origin point relate to
            smallest point. Defaults to 0.5

    Returns:
        np.ndarray, shape=[N, 2 ** ndim, ndim]: Returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1.
    """
    ndim = int(dims.shape[1])
    corners_norm = unravel_index(torch.arange(2**ndim), [2] * ndim).type(dims.dtype).to(dims.device)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - origin
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2**ndim, ndim])
    return corners


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """Convert kitti locations, dimensions and angles to corners.
    format: center(xy), dims(xy), angles(counterclockwise when positive)

    Args:
        centers (np.ndarray): Locations in kitti label file with shape (N, 2).
        dims (np.ndarray): Dimensions in kitti label file with shape (N, 2).
        angles (np.ndarray, optional): Rotation_y in kitti label file with
            shape (N). Defaults to None.
        origin (list or array or float, optional): origin point relate to
            smallest point. Defaults to 0.5.

    Returns:
        np.ndarray: Corners with the shape of (N, 4, 2).
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles.squeeze())
    corners += centers.reshape([-1, 1, 2])
    return corners