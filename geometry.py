import numpy as np
from scipy.spatial import Delaunay

def delaunay_triangulation(points):
    tri = Delaunay(points)
    return tri.simplices

def affine_transform(src_tri, dst_tri):
    src = np.hstack((src_tri, np.ones((3, 1))))
    dst = dst_tri
    M = np.linalg.solve(src, dst)
    return M

