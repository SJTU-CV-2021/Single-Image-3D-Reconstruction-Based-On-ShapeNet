import os 
import numpy as np
import open3d as o3d
import sys
import shutil

def obj2pcd(filedir, shapenet):
    # print('当前工作目录为：{}\n'.format(wdir))
    
    for file in os.listdir(filedir):
        if not '.' in file:
            continue
        prefix = file.split('.')[0]
        suffix = file.split('.')[1]
        if suffix != 'obj':
            continue
        #f = open('0_pred.obj','rb')
        new_name = (prefix + '.pcd') if not shapenet else (prefix + '_sample.pcd')
        # print(new_name)

        if shapenet:
            mesh = o3d.io.read_triangle_mesh(os.path.join(filedir, file))
            pcd = mesh.sample_points_poisson_disk(number_of_points=500, init_factor=4)
            # o3d.visualization.draw_geometries([pcd])

            # pcd = mesh.sample_points_uniformly(number_of_points=0000)
            # pcd = mesh.sample_points_poisson_disk(number_of_points=2000, pcl=pcd)
            o3d.io.write_point_cloud(os.path.join(filedir, new_name), pcd)
            continue
            # o3d.visualization.draw_geometries([pcd])
        else:
            mesh = o3d.io.read_triangle_mesh(os.path.join(filedir, file))
            mesh = mesh.scale(1 / 2.6, center=mesh.get_center())
            pcd = mesh.sample_points_poisson_disk(number_of_points=1500, init_factor=4)
            # o3d.visualization.draw_geometries([pcd])

            # pcd = mesh.sample_points_uniformly(number_of_points=0000)
            # pcd = mesh.sample_points_poisson_disk(number_of_points=2000, pcl=pcd)
            o3d.io.write_point_cloud(os.path.join(filedir, new_name), pcd)
            continue
        f = open(os.path.join(filedir, new_name),'w')

        #pcd的数据格式 https://blog.csdn.net/BaiYu_King/article/details/81782789  
        
        lines_v = []
        f1 = open(os.path.join(filedir, file),'rb')
        lines = f1.readlines()
        for line in lines:
            line = line.decode().split(' ')
            if line[0] != 'v':
                continue
            lines_v.append(line)

        num_lines = len(lines_v)
        # print(num_lines)  

        f.write('# .PCD v0.7 - Point Cloud Data file format \nVERSION 0.7 \nFIELDS x y z \nSIZE 4 4 4 \nTYPE F F F \nCOUNT 1 1 1 \n' )
        f.write('WIDTH {} \nHEIGHT 1 \nVIEWPOINT 0 0 0 1 0 0 0 \n'.format(num_lines))
        f.write('POINTS {} \nDATA ascii\n'.format(num_lines))
        
        a = []
        for line1 in lines_v:
            new_line = line1[1] + ' ' + line1[2] + ' ' + line1[3] 
            
            f.write(new_line)
        f.close()

if __name__ == '__main__':
    obj2pcd('models/Total3DUnderstanding/demo/outputs/2')
