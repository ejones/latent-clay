import copy

import open3d as o3d
import numpy as np


def find_horizontal_rotation_to_min_box(mesh):
    mesh = copy.deepcopy(mesh)
    result = np.array([0, 0, 0], float)

    basis = np.array([0, 0.05, 0], float)
    rot = o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(basis)
    current_vol = mesh.get_axis_aligned_bounding_box().volume()
    cum_rot = np.array([0, 0, 0], float)

    mesh.rotate(rot)
    result += basis
    vol = mesh.get_axis_aligned_bounding_box().volume()
    if vol > current_vol:
        basis = -basis
        rot = o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(basis)
    current_vol = vol

    found = False
    for attempt in range(50):
        mesh.rotate(rot)
        vol = mesh.get_axis_aligned_bounding_box().volume()
        if vol > current_vol:
            return result
        result += basis
        current_vol = vol

    return np.array([0, 0, 0], float)


def get_triangle_area(p1, p2, p3):
    x = np.subtract(p1, p2)
    y = np.subtract(p1, p3)
    return 0.5 * np.linalg.norm(np.cross(x, y))


def get_axis_angle_to_align(src, dest):
    axis = np.cross(src, dest)
    angle = np.arccos(np.dot(src, dest))
    if np.linalg.norm(axis) != 0:
        axis = axis / np.linalg.norm(axis)
    return axis, angle


def get_stable(mesh, n=50):
    hull = mesh.compute_convex_hull()[0]
    hull = hull.simplify_quadric_decimation(n)
    hull.compute_triangle_normals()
    #print(hull)
    max_angle = np.pi / 7
    area_of_max = 0
    selected = None
    
    center = hull.get_center()

    for i, norm in enumerate(hull.triangle_normals):
        axis, angle = get_axis_angle_to_align(norm, [0, -1, 0])
        if angle < max_angle:
            verts = [hull.vertices[vi] for vi in hull.triangles[i]]

            # FIXME, use centroid instead?
            if any(p[1] < center[1] for p in verts):
                area = get_triangle_area(*verts)
                if area > area_of_max:
                    area_of_max = area
                    selected = i, axis, angle

    if selected is not None:
        i, axis, angle = selected
        print('rotating by', angle, 'rad about', axis)
        rot = o3d.geometry.TriangleMesh.get_rotation_matrix_from_axis_angle(axis * angle)
        mesh = copy.deepcopy(mesh)
        mesh.rotate(rot)
        hull.rotate(rot)
        colors = np.ones((len(hull.vertices), 3))
        for vi in hull.triangles[i]:
            colors[vi] = [1, 0, 0]
        hull.vertex_colors = o3d.utility.Vector3dVector(colors)
    else:
        print('not rotating')
    return hull, mesh


def process_tripo_mesh(mesh):
    rot = o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz((-np.pi / 2, 0, -np.pi / 2))
    new_mesh = copy.deepcopy(mesh)
    new_mesh.rotate(rot)
    new_mesh.remove_non_manifold_edges()
    new_mesh = new_mesh.simplify_quadric_decimation(10000)

    _, new_mesh = get_stable(new_mesh)

    rot = find_horizontal_rotation_to_min_box(new_mesh)
    rx, ry, rz = rot * 180 / np.pi
    if rx or ry or rz:
        print('rotating by x:', round(rx, 1), 'y:', round(ry, 1), 'z:', round(rz, 1), 'to level model')
        new_mesh.rotate(o3d.geometry.TriangleMesh.get_rotation_matrix_from_xyz(rot))
    else:
        print('could not find rotation to get model even or is already event')
        
    return new_mesh


def main(mesh_path, output_path):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh = process_tripo_mesh(mesh)
    o3d.io.write_triangle_mesh(output_path, mesh)


if __name__ == '__main__':
    import sys
    main(sys.argv[1], sys.argv[2])

