import open3d as o3d
import os
import sys
import re
from open3d import geometry


class Converter:
    def __init__(self, target_name, result_name):

        mesh = o3d.io.read_triangle_mesh(target_name)
        point_cloud_mash = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, 18241)
        o3d.visualization.draw_geometries([point_cloud_mash])
        o3d.io.write_point_cloud(result_name, point_cloud_mash, True)

    @staticmethod
    def stl_to_point_cloud():
        path = ""
        path_directories = sys.path.pop(0).split("\\")
        for directory in path_directories:
            path += str(directory) + str('\\\\')
        for file in os.listdir(path):
            if file.endswith(".stl") and re.search("^seabed_sample#",file):
                mesh = o3d.io.read_triangle_mesh(file)
                point_cloud_mash = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, int(12100))
                #o3d.visualization.draw_geometries([point_cloud_mash])

                o3d.io.write_point_cloud(str("C:\\Users\\adria\\Desktop\\New folder (2)\\DRL_SeaClear_SIM\\Env\\xyz_env\\")+file.split(".")[0]+str(".xyz"), point_cloud_mash, True)
