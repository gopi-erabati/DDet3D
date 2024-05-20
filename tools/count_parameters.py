import argparse
import os
import torch
from mmcv import Config
from mmdet3d.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Count parameters')
    parser.add_argument('config', help='config file path')
    parser.add_argument('--module', default='total', type=str, help='type of module')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                _module_path = cfg.plugin_dir
                # _module_dir = os.path.dirname(plugin_dir)
                # _module_dir = _module_dir.split('/')
                # _module_path = _module_dir[0]
                #
                # for m in _module_dir[1:]:
                #     _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    # count parameters
    if args.module == 'total':
        num_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    elif args.module == 'backbone':
        num_par = 0
        for name, p in model.named_parameters():
            if 'pts_voxel_encoder' in name or 'pts_middle_encoder' in name or 'pts_backbone' in name:
                if p.requires_grad:
                    num_par += p.numel()
    elif args.module == 'neck':
        num_par = 0
        for name, p in model.named_parameters():
            if 'pts_neck' in name:
                if p.requires_grad:
                    num_par += p.numel()
    elif args.module == 'head':
        num_par = 0
        for name, p in model.named_parameters():
            if 'bbox_head' in name:
                if p.requires_grad:
                    num_par += p.numel()

    print(f'The number of parameters are: {num_par}')


if __name__ == '__main__':
    main()