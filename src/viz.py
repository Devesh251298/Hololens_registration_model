# examples/Python/Basic/visualization.py

import numpy as np
import open3d as o3d

if __name__ == "__main__":

    print("Load a ply point cloud, print it, and render it")
    pcd = o3d.io.read_point_cloud("STL/final_demo.ply")
    print(np.asarray(pcd.points).shape)
    
    # Check if the loaded point cloud has colors
    # if pcd.has_colors():
    o3d.visualization.draw_geometries([pcd])
    # else:
    #     print("The loaded point cloud does not have color information.")