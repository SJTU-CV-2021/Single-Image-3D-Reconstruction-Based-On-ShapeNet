import argparse
import os
from typing import Type
import numpy as np
import open3d as o3d
from glob import glob
from copy import deepcopy

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Reconstructor')
    parser.add_argument('--input_path', type=str, default='demo/inputs/1', help='Please specify the path.')
    parser.add_argument('--output_path', type=str, default='demo/outputs/1', help='Please specify the path.')
    return parser.parse_args()

def get_bdb_form_from_corners(corners):
    vec_0 = (corners[:, 2, :] - corners[:, 1, :]) / 2.
    vec_1 = (corners[:, 0, :] - corners[:, 4, :]) / 2.
    vec_2 = (corners[:, 1, :] - corners[:, 0, :]) / 2.

    coeffs_0 = np.linalg.norm(vec_0, axis=1)
    coeffs_1 = np.linalg.norm(vec_1, axis=1)
    coeffs_2 = np.linalg.norm(vec_2, axis=1)
    coeffs = np.stack([coeffs_0, coeffs_1, coeffs_2], axis=1)

    centroid = (corners[:, 0, :] + corners[:, 6, :]) / 2.

    basis_0 = np.dot(np.diag(1 / coeffs_0), vec_0)
    basis_1 = np.dot(np.diag(1 / coeffs_1), vec_1)
    basis_2 = np.dot(np.diag(1 / coeffs_2), vec_2)

    basis = np.stack([basis_0, basis_1, basis_2], axis=1)

    return {'basis': basis, 'coeffs': coeffs, 'centroid': centroid}

def format_bbox(box, type):

    if type == 'prediction':
        boxes = {}
        basis_list = []
        centroid_list = []
        coeff_list = []

        # convert bounding boxes
        box_data = box['bdb'][0]

        for index in range(len(box_data)):
            basis = box_data[index]['basis'][0][0]
            centroid = box_data[index]['centroid'][0][0][0]
            coeffs = box_data[index]['coeffs'][0][0][0]
            basis_list.append(basis)
            centroid_list.append(centroid)
            coeff_list.append(coeffs)

        boxes['basis'] = np.stack(basis_list, 0)
        boxes['centroid'] = np.stack(centroid_list, 0)
        boxes['coeffs'] = np.stack(coeff_list, 0)
        boxes['class_id'] = box['class_id'][0]

    else:

        boxes = get_bdb_form_from_corners(box['bdb3D'])
        boxes['class_id'] = box['size_cls'].tolist()

    return boxes

def format_layout(layout_data):

    layout_bdb = {}

    centroid = (layout_data.max(0) + layout_data.min(0)) / 2.

    vector_z = (layout_data[1] - layout_data[0]) / 2.
    coeff_z = np.linalg.norm(vector_z)
    basis_z = vector_z/coeff_z

    vector_x = (layout_data[2] - layout_data[1]) / 2.
    coeff_x = np.linalg.norm(vector_x)
    basis_x = vector_x/coeff_x

    vector_y = (layout_data[0] - layout_data[4]) / 2.
    coeff_y = np.linalg.norm(vector_y)
    basis_y = vector_y/coeff_y

    basis = np.array([basis_x, basis_y, basis_z])
    coeffs = np.array([coeff_x, coeff_y, coeff_z])

    layout_bdb['coeffs'] = coeffs
    layout_bdb['centroid'] = centroid
    layout_bdb['basis'] = basis

    return layout_bdb

def format_mesh(obj_files, bboxes):

    o3d_objects = {}

    for obj_file in obj_files:
        filename = '.'.join(os.path.basename(obj_file).split('.')[:-1])
        obj_idx = int(filename.split('_')[0])
        class_id = int(filename.split('_')[1].split(' ')[0])
        assert bboxes['class_id'][obj_idx] == class_id

        obj = o3d.io.read_triangle_mesh(obj_file)

        mesh_coef = (obj.get_max_bound() - obj.get_min_bound()) / 2.
        scale = np.diag(np.append(1./mesh_coef * bboxes['coeffs'][obj_idx],1))
        print(scale)
        obj = obj.transform(scale)
        obj = obj.rotate(bboxes['basis'][obj_idx])

        # move to center
        obj = obj.translate(bboxes['centroid'][obj_idx])

        o3d_objects[obj_idx] = obj

    return o3d_objects, bboxes

def normalize_point(v):
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def get_world_R(cam_R):
    '''
    set a world system from camera matrix
    :param cam_R:
    :return:
    '''
    toward_vec = deepcopy(cam_R[:,0])
    toward_vec[1] = 0.
    toward_vec = normalize_point(toward_vec)
    up_vec = np.array([0., 1., 0.])
    right_vec = np.cross(toward_vec, up_vec)

    world_R = np.vstack([toward_vec, up_vec, right_vec]).T
    # yaw, _, _ = yaw_pitch_roll_from_R(cam_R)
    # world_R = R_from_yaw_pitch_roll(yaw, 0., 0.)
    return world_R

if __name__ == '__main__':
    args = parse_args()
    input_path = args.__dict__['input_path']
    output_path = args.__dict__['output_path']

    import scipy.io as sio
    pre_layout_data = sio.loadmat(os.path.join(output_path, 'layout.mat'))['layout']
    pre_box_data = sio.loadmat(os.path.join(output_path, 'bdb_3d.mat'))
    
    pre_boxes = format_bbox(pre_box_data, 'prediction')
    pre_layout = format_layout(pre_layout_data)
    pre_cam_R = sio.loadmat(os.path.join(output_path, 'r_ex.mat'))['cam_R']

    o3d_objects, pre_boxes = format_mesh(glob(os.path.join(output_path, '*_s.obj')), pre_boxes)

    from PIL import Image, ImageDraw, ImageFont

    image = np.array(Image.open(os.path.join(input_path, 'img.jpg')).convert('RGB'))
    cam_K = np.loadtxt(os.path.join(input_path, 'cam_K.txt'))

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=image.shape[1], height=image.shape[0], left=0, top=0)
    render_option: o3d.visualization.RenderOption = vis.get_render_option()	#设置点云渲染参数
    render_option.background_color = np.array([0, 0, 0])	#
    view_control: o3d.visualization.ViewControl = vis.get_view_control()

    cam = o3d.camera.PinholeCameraIntrinsic()
    cam.set_intrinsics(
        width=image.shape[1], height=image.shape[0], 
        fx=cam_K[0,0], fy=cam_K[1,1], cx=cam_K[0,2], cy=cam_K[1,2])

    view_control.convert_from_pinhole_camera_parameters()
    for obj in o3d_objects.values():
        vis.add_geometry(obj)
        vis.update_geometry(obj)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(os.path.join(output_path, 'visualize.png'))
    base_img = Image.open(os.path.join(output_path, 'visualize.png'))
    image = Image.blend(image, base_img, 0.5)
    image.save(os.path.join(output_path, 'visualize.png'))
    # o3d.visualization.draw_geometries([obj for obj in o3d_objects.values()])
    # vis.destroy_window()