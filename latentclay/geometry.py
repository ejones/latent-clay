from dataclasses import dataclass
import os
from typing import Callable, List, Tuple

import tripy
import numpy as np 

def dist(p1, p2):
    return np.linalg.norm(np.subtract(p2, p1))


@dataclass(frozen=True)
class SimpleMesh:
    vertices: List[Tuple[float]]
    triangles: List[Tuple[int]]
    triangle_uvs: List[Tuple[float]]


def gen_line_segment_wall_uvs(xzs, y, tex_scale=12, tex_start=0):
    print('gen_line_segment_wall_uvs', xzs, y, tex_scale, tex_start)
    first = True
    pu = tex_start / tex_scale # REVIEW: tex_start is in world distance I guess?
    for x, z in xzs:
        if not first:
            u = pu + dist([px, pz], [x, z]) / tex_scale
            yield pu / y, 1, u / y, 0
            pu = u
        first = False
        px, pz = x, z


def make_line_segment_wall(xzs, y, tex_scale=12, tex_start=0, mask=None, mirror=False):
    vertices = []
    triangles = []
    triangle_uvs = []
    if mask is None:
        mask = [True] * max(0, len(xzs) - 1)

    # TODO: add wall thickness
    # for now, we double back the segments
    full_path = [*xzs, *list(reversed(xzs))[1:]]
    full_mask = [*mask, *reversed(mask)]
    if mirror:
        path_uvs = list(gen_line_segment_wall_uvs(xzs, y, tex_scale, tex_start))
        full_uvs = [*path_uvs, *((r, t, l, b) for l, t, r, b in reversed(path_uvs))]
    else:
        full_uvs = list(gen_line_segment_wall_uvs(full_path, y, tex_scale, tex_start))

    first = True
    for i, (x, z) in enumerate(full_path):
        vertices.append([x, 0, z])
        vertices.append([x, y, z])
        if not first:
            nv = len(vertices)
            if full_mask[i - 1]:
                u_left, v_top, u_right, v_bot = full_uvs[i - 1]
                triangles.append((nv - 4, nv - 2, nv - 1))
                triangles.append((nv - 4, nv - 1, nv - 3))
                triangle_uvs.extend([
                    (u_left, v_bot), (u_right, v_bot), (u_right, v_top),
                    (u_left, v_bot), (u_right, v_top), (u_left, v_top),
                ])
        first = False

    return SimpleMesh(vertices, triangles, triangle_uvs)


def make_polygon_floor(xzs, y, rev=False):
    tri_points = tripy.earclip(xzs)

    vertices = list(set((x, y, z) for t in tri_points for x, z in t))
    vert_idx = {p: i for i, p in enumerate(vertices)}
    triangles = [
        ([vert_idx[(x, y, z)] for x, z in t] if rev else list(reversed([vert_idx[(x, y, z)] for x, z in t])))
        for t in tri_points
    ]

    min_x = min(p[0] for p in vertices)
    max_x = max(p[0] for p in vertices)
    min_z = min(p[2] for p in vertices)
    max_z = max(p[2] for p in vertices)

    # square floor texture
    # TODO min scale
    dim = max(max_x - min_x, max_z - min_z)

    triangle_uvs = [
        ((x - min_x) / dim, (z - min_z) / dim)
        for t in triangles
        for p in t
        for x, _, z in (vertices[p],)
    ]

    return SimpleMesh(vertices, triangles, triangle_uvs)


def line_seg_intersection(seg1, seg2):
    a, b = np.array(seg1)
    c, d = np.array(seg2)
    e = b - a
    f = d - c
    # AB = A + E * g; CD = C + F * h

    pe = [-e[1], e[0]]
    pf = [-f[1], f[0]]

    g = np.dot(c - a, pf) / np.dot(e, pf)
    h = np.dot(a - c, pe) / np.dot(f, pe)

    return g if 0 <= g <= 1 and 0 <= h <= 1 else None


def get_door_clearance_edges(start, end, clearance=1):
    door_vec = np.subtract(end, start)
    door_norm = np.array([door_vec[1], -door_vec[0]])
    door_norm = door_norm / np.linalg.norm(door_norm)
    half_clr_vec = door_norm * (clearance / 2)
    start_edge = half_clr_vec + start, -half_clr_vec + start
    end_edge = half_clr_vec + end, -half_clr_vec + end
    return start_edge, end_edge


def cut_doorway(start, end, line, clearance=1, mask=None):
    start_edge, end_edge = get_door_clearance_edges(start, end, clearance)
    new_line = [line[0]]
    new_mask = []
    found = False
    d_door = 1
    for i in range(1, len(line)):
        line_edge = [line[i - 1], line[i]]
        edge_vec = np.subtract(line[i], line[i - 1])
        d1 = line_seg_intersection(line_edge, start_edge)
        d2 = line_seg_intersection(line_edge, end_edge)

        # TODO handle door spanning multiple segments
        if d1 is not None and d2 is not None:
            if d2 < d1:
                d1, d2 = d2, d1
            new_line.extend(
                (tuple(edge_vec * d1 + line[i - 1]), tuple(edge_vec * d2 + line[i - 1]))
            )
            new_mask.append(mask[i - 1] if mask is not None else True)
            new_mask.append(False)
            found = True
            d_door = line_seg_intersection(start_edge, line_edge)
        new_line.append(line[i])
        new_mask.append(mask[i - 1] if mask is not None else True)

    return (new_line, new_mask, abs(d_door - 0.5)) if found else (None, None, 1)


def uvs_to_mesh(uvs, u_start, u_end, v_start, v_end, tolerance=0.01):
    assert len(uvs) % 3 == 0, 'uneven number of UVs for triangles'
    uv_arr = np.array(uvs)
    uv_arr[uv_arr < tolerance] = 0
    uv_arr[uv_arr > (1 - tolerance)] = 1
    tri_verts = list(map(tuple, (
        uv_arr[:, 0, None] * np.subtract(u_end, u_start) + u_start +
        uv_arr[:, 1, None] * np.subtract(v_end, v_start) + v_start
    )))

    vertices = list(set(tri_verts))
    vert_idx = {p: i for i, p in enumerate(vertices)}
    triangles = [
        (vert_idx[tri_verts[i]], vert_idx[tri_verts[i + 1]], vert_idx[tri_verts[i + 2]])
        for i in range(0, len(tri_verts), 3)
    ]

    return SimpleMesh(vertices, triangles, uvs)


def write_obj_file(name, mesh, tex_path=None):
    with open(f'{name}.obj', 'w') as f:
        print(f'mtllib {os.path.basename(name)}.mtl', file=f)
        for x, y, z in mesh.vertices:
            print('v', f'{x:.6g} {y:.6g} {z:.6g}', file=f)
        for u, v in mesh.triangle_uvs:
            print('vt', f'{u:.6g} {v:.6g}', file=f)
        print(f'usemtl {os.path.basename(name)}_0', file=f)
        for i, (a, b, c) in enumerate(mesh.triangles):
            uvn = i * 3 + 1
            print('f', f'{a + 1}/{uvn} {b + 1}/{uvn + 1} {c + 1}/{uvn + 2}', file=f)

    with open(f'{name}.mtl', 'w') as f:
        print(f'newmtl {os.path.basename(name)}_0', file=f)
        print('Ka 1.000 1.000 1.000\nKd 1.000 1.000 1.000\nKs 0.000 0.000 0.000', file=f)
        if tex_path:
            print(f'\nmap_Ka {tex_path}\nmap_Kd {tex_path}', file=f)


def update_obj_file_texture(name, tex_path):
    with open(f'{name}.mtl', 'w') as f:
        print(f'newmtl {os.path.basename(name)}_0', file=f)
        print('Ka 1.000 1.000 1.000\nKd 1.000 1.000 1.000\nKs 0.000 0.000 0.000', file=f)
        if tex_path:
            print(f'\nmap_Ka {tex_path}\nmap_Kd {tex_path}', file=f)
