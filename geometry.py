import numpy as np
from scipy.spatial import Delaunay

class Triangle:
    """Represents a triangle with three vertices"""

    def __init__(self, p1, p2, p3):
        self.vertices = [p1, p2, p3]
        self.circumcenter = None
        self.circumradius = None
        self._compute_circumcircle()

    def _compute_circumcircle(self):
        """Compute circumcenter and circumradius of the triangle"""
        ax, ay = self.vertices[0]
        bx, by = self.vertices[1]
        cx, cy = self.vertices[2]

        d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        if abs(d) < 1e-10:
            # Degenerate triangle
            self.circumcenter = None
            self.circumradius = float("inf")
            return

        ux = (
            (ax**2 + ay**2) * (by - cy)
            + (bx**2 + by**2) * (cy - ay)
            + (cx**2 + cy**2) * (ay - by)
        ) / d
        uy = (
            (ax**2 + ay**2) * (cx - bx)
            + (bx**2 + by**2) * (ax - cx)
            + (cx**2 + cy**2) * (bx - ax)
        ) / d

        self.circumcenter = (ux, uy)
        self.circumradius = np.sqrt((ax - ux) ** 2 + (ay - uy) ** 2)

    def point_in_circumcircle(self, p):
        """Check if point p is inside the circumcircle"""
        if self.circumcenter is None:
            return False

        dx = p[0] - self.circumcenter[0]
        dy = p[1] - self.circumcenter[1]
        dist = np.sqrt(dx**2 + dy**2)

        return dist < self.circumradius - 1e-10

    def has_vertex(self, p):
        """Check if triangle contains vertex p"""
        for v in self.vertices:
            if abs(v[0] - p[0]) < 1e-10 and abs(v[1] - p[1]) < 1e-10:
                return True
        return False


class Edge:
    """Represents an edge between two points"""

    def __init__(self, p1, p2):
        # Sort points to make comparison easier
        if (p1[0], p1[1]) < (p2[0], p2[1]):
            self.p1, self.p2 = p1, p2
        else:
            self.p1, self.p2 = p2, p1

    def __eq__(self, other):
        return (
            abs(self.p1[0] - other.p1[0]) < 1e-10
            and abs(self.p1[1] - other.p1[1]) < 1e-10
            and abs(self.p2[0] - other.p2[0]) < 1e-10
            and abs(self.p2[1] - other.p2[1]) < 1e-10
        )

    def __hash__(self):
        return hash(
            (
                round(self.p1[0], 8),
                round(self.p1[1], 8),
                round(self.p2[0], 8),
                round(self.p2[1], 8),
            )
        )


def delaunay_triangulation(points):
    """
    Bowyer-Watson algorithm for Delaunay triangulation

    Args:
        points: numpy array of shape (n, 2) containing 2D points

    Returns:
        numpy array of shape (m, 3) containing triangle indices
    """
    if len(points) < 3:
        raise ValueError("Need at least 3 points for triangulation")

    points = np.array(points, dtype=np.float64)

    # Create super-triangle that contains all points
    min_x, min_y = points.min(axis=0)
    max_x, max_y = points.max(axis=0)

    dx = max_x - min_x
    dy = max_y - min_y
    delta_max = max(dx, dy)
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2

    # Create super-triangle vertices (large enough to contain all points)
    p1 = (mid_x - 20 * delta_max, mid_y - delta_max)
    p2 = (mid_x, mid_y + 20 * delta_max)
    p3 = (mid_x + 20 * delta_max, mid_y - delta_max)

    # Initialize triangulation with super-triangle
    triangles = [Triangle(p1, p2, p3)]

    # Add points one at a time
    for point in points:
        point = tuple(point)
        bad_triangles = []

        # Find all triangles whose circumcircle contains the point
        for tri in triangles:
            if tri.point_in_circumcircle(point):
                bad_triangles.append(tri)

        # Find the boundary of the polygonal hole
        polygon = []
        for tri in bad_triangles:
            for i in range(3):
                edge = Edge(tri.vertices[i], tri.vertices[(i + 1) % 3])

                # Check if edge is shared by another bad triangle
                is_shared = False
                for other_tri in bad_triangles:
                    if other_tri is tri:
                        continue
                    for j in range(3):
                        other_edge = Edge(
                            other_tri.vertices[j], other_tri.vertices[(j + 1) % 3]
                        )
                        if edge == other_edge:
                            is_shared = True
                            break
                    if is_shared:
                        break

                if not is_shared:
                    polygon.append(edge)

        # Remove bad triangles
        for tri in bad_triangles:
            triangles.remove(tri)

        # Create new triangles from the point to each edge of the polygon
        for edge in polygon:
            new_tri = Triangle(point, edge.p1, edge.p2)
            triangles.append(new_tri)

    # Remove triangles that contain super-triangle vertices
    final_triangles = []
    for tri in triangles:
        if not (tri.has_vertex(p1) or tri.has_vertex(p2) or tri.has_vertex(p3)):
            final_triangles.append(tri)

    # Convert triangles to indices
    point_list = [tuple(p) for p in points]
    result = []

    for tri in final_triangles:
        indices = []
        for vertex in tri.vertices:
            # Find index of vertex in original points
            for i, p in enumerate(point_list):
                if abs(p[0] - vertex[0]) < 1e-10 and abs(p[1] - vertex[1]) < 1e-10:
                    indices.append(i)
                    break

        if len(indices) == 3:
            result.append(indices)

    return np.array(result, dtype=np.int32)


def affine_transform(src_tri, dst_tri):
    src = np.hstack((src_tri, np.ones((3, 1))))
    dst = dst_tri
    M = np.linalg.solve(src, dst)
    return M
