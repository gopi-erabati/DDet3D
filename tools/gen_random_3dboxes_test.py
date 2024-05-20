import torch
from mmdet3d_plugin.core.bbox.util import denormalize_0to1_bbox
from mmdet3d.core.bbox import LiDARInstance3DBoxes
import importlib
import mmcv
from tqdm import tqdm

importlib.import_module('mmdet3d_plugin')

results = []

for _ in tqdm(range(6019)):
    noise_bboxes_raw = torch.randn((900, 10))  # (n_p, 10)
    noise_bboxes = torch.clamp(noise_bboxes_raw, min=-1 * 2.0, max=2.0)
    noise_bboxes = ((noise_bboxes / 2.0) + 1) / 2

    # change any size=0 to min value so that log wont get -inf
    size = noise_bboxes[:, 3:6].clone()  # (n_p, 3)
    size[size == 0.0] = 1e-5
    noise_bboxes[:, 3:6] = size

    noise_bboxes = denormalize_0to1_bbox(noise_bboxes,
                                         [-55.2, -55.2, -5.0, 55.2, 55.2, 3.0],
                                         [40.0, 40.0, 20.0])
    # (n_p, 10)

    rot_sine = noise_bboxes[..., 6:7]

    rot_cosine = noise_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    boxes = torch.cat([noise_bboxes[..., :6], rot], dim=1)

    boxes = LiDARInstance3DBoxes(boxes)

    result_dict = dict(
        boxes_3d=boxes,
        scores_3d=torch.rand((900,)),
        labels_3d=torch.randint(0, 10, (900,))
    )

    results.append(result_dict)

mmcv.dump(results, './random_init/random_init'
                   '.pkl')
