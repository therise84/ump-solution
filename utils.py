import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from scipy.spatial import KDTree

def snap_to_nearest_road(mask, point, max_dist=50):
    road_pixels = np.column_stack(np.where(mask > 0))
    tree = KDTree(road_pixels)
    dist, idx = tree.query([point[1], point[0]])
    if dist <= max_dist:
        y, x = road_pixels[idx]
        return (x, y)
    return None

def build_graph_from_mask(mask):
    h, w = mask.shape
    G = nx.Graph()
    for y in range(h):
        for x in range(w):
            if mask[y, x]:
                for dy in [-1,0,1]:
                    for dx in [-1,0,1]:
                        ny, nx_ = y+dy, x+dx
                        if 0<=ny<h and 0<=nx_<w:
                            if mask[ny, nx_]:
                                weight = np.sqrt(dx*dx+dy*dy)
                                G.add_edge((x,y),(nx_,ny),weight=weight)
    return G

def simplify_path(path, epsilon=2.0):
    from skimage.measure import approximate_polygon
    coords = np.array(path)
    simplified = approximate_polygon(coords, tolerance=epsilon)
    return [(int(p[0]), int(p[1])) for p in simplified]