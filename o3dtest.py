import open3d as o3d
import numpy as np
import copy

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


mesh = o3d.io.read_triangle_mesh('data\ShapeNetCore\\03001627\\1a6f615e8b1b5ae4dbbc9440457e303e\model.obj')
# mesh.compute_vertex_normals()
mesh = mesh.scale(2.6, center=mesh.get_center())
# o3d.visualization.draw_geometries([mesh])
pcd = mesh.sample_points_uniformly(number_of_points=2000)
# o3d.visualization.draw_geometries([pcd])
o3d.io.write_point_cloud('o3dtest.pcd', pcd)

source = pcd
target = o3d.io.read_point_cloud('data\\nyuv2\\format\\outputs\\0\\3_5.pcd')

threshold = 0.2

trans_init = np.asarray([[0, 0, 1, 0],
                        [0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1]])
draw_registration_result(source, target, trans_init)

# evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)

reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

draw_registration_result(source, target, reg_p2p.transformation)

