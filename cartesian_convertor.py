import open3d as o3d
from math import cos, sin

# approx the radius of earth in km
earth_radius = 6371
file_path= "real_data/IN2020_V08/xyz/in2020_v08_Investigator_em122_2020-339_0001_20201204_121432_Investigator_em122_EM122.txt"
X=[]
Y=[]
Z=[]
with open(file_path) as f:
    for line in f:
        [x, y, z] = line.split()
        x_conv = earth_radius*cos(float(x))*cos(float(y))
        y_conv = earth_radius*cos(float(x))*sin(float(y))
        z_conv = float(z)/10
        X.append(x_conv)
        Y.append(y_conv)
        Z.append(z_conv)
X = X[0:12100*2]
Y = Y[0:12100*2]
Z = Z[0:12100*2]

print("jhj")
myfile = open("real_data/IN2020_V08/data_patches/real_sample.xyz","w")
for line in range(12100*2):
     myfile.write("{}\t {}\t {}\t\n".format(X[line],Y[line],Z[line]))
myfile.close()

# mesh = o3d.io.read_triangle_mesh("real_data/IN2020_V08/data_patches/real_sample.xyz")
# point_cloud_mash = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, int(12100))
# o3d.visualization.draw_geometries([point_cloud_mash])


