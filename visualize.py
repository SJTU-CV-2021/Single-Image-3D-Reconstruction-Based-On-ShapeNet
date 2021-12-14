import os
import scipy.io as sio
from utils.visualize import format_bbox, format_layout, format_mesh, Box
from glob import glob
import argparse
from configs.config_utils import CONFIG
import numpy as np
from PIL import Image

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Total 3D Understanding.')
    parser.add_argument('config', type=str, default='configs/total3d.yaml',
                        help='configure file for training or testing.')
    parser.add_argument('--mode', type=str, default='train', help='train, test or demo.')
    parser.add_argument('--demo_path', type=str, default='demo/inputs/1', help='Please specify the demo path.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    cfg = CONFIG(args.config)
    cfg.update_config(args.__dict__)

    save_path = cfg.config['demo_path'].replace('inputs', 'outputs')

    pre_layout_data = sio.loadmat(os.path.join(save_path, 'layout.mat'))['layout']
    pre_box_data = sio.loadmat(os.path.join(save_path, 'bdb_3d.mat'))

    pre_boxes = format_bbox(pre_box_data, 'prediction')
    pre_layout = format_layout(pre_layout_data)
    pre_cam_R = sio.loadmat(os.path.join(save_path, 'r_ex.mat'))['cam_R']

    vtk_objects, pre_boxes = format_mesh(glob(os.path.join(save_path, '*.obj')), pre_boxes)

    image = np.array(Image.open(os.path.join(cfg.config['demo_path'], 'img.jpg')).convert('RGB'))
    cam_K = np.loadtxt(os.path.join(cfg.config['demo_path'], 'cam_K.txt'))

    scene_box = Box(image, None, cam_K, None, pre_cam_R, None, pre_layout, None, pre_boxes, 'prediction', output_mesh = vtk_objects)
    scene_box.draw_projected_bdb3d('prediction', if_save=True, save_path = '%s/3dbbox.png' % (save_path))
    # scene_box.draw_object(best_match, if_save=True, save_path = '%s/visualize.png' % (save_path))
    scene_box.draw3D(if_save=True, save_path = '%s/recon.png' % (save_path))