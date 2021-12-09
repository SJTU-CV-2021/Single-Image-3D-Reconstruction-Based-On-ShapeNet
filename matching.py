import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import copy

def write_obj(objfile, data):
    with open(objfile, 'w+') as file:
        for item in data['v']:
            file.write('v' + ' %f' * len(item) % tuple(item) + '\n')

        for item in data['f']:
            file.write('f' + ' %s' * len(item) % tuple(item) + '\n')

def read_obj(objfile):
    data = {'v' : [], 'f' : []}
    with open(objfile, "r") as file:
        list_of_lines = file.readlines()
        for line in list_of_lines:
            items = line.split()
            if items[0] == 'v':
                data['v'].append(np.array(items[1:], dtype=float))
            elif items[0] == 'f':
                data['f'].append(np.array(items[1:], dtype=int))
    return data 


# mesh_obj = {'v': current_coordinates, 'f': current_faces}
# def mesh_diff(mesh_obj_1, mesh_obj_2):
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    vis = o3d.visualization.Visualizer() 
    vis.create_window(visible = False) 
    vis.add_geometry(source_temp)
    vis.add_geometry(target_temp) 
    img = vis.capture_screen_float_buffer(True)
    plt.imshow(np.asarray(img))
    plt.savefig('test.png')
    # o3d.visualization.draw_geometries([source_temp, target_temp])

source = o3d.io.read_point_cloud("models/Total3DUnderstanding/demo/outputs/2/0_7.pcd")
target = o3d.io.read_point_cloud("models/Total3DUnderstanding/demo/outputs/2/1_5.pcd")
threshold = 0.1
print(source)
print(target)
trans_init = np.asarray([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0], 
                         [0, 0, 0, 1]])
print("Apply point-to-point ICP")
reg_p2p = o3d.pipelines.registration.registration_icp(
    source, target, threshold, trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint())
print(reg_p2p)
print("Transformation is:")
print(reg_p2p.transformation)

draw_registration_result(source, target, reg_p2p.transformation)