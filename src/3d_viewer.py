import sys
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from stl import mesh
import pyvista as pv

def plot_stl(file_path):
    mesh = pv.read(file_path)
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color='w')
    plotter.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 3d_viewer.py <stl_file_path>")
        sys.exit(1)

    stl_file_path = sys.argv[1]
    plot_stl(stl_file_path)
