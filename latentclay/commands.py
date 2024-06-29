import argparse
import json
import math
import os
import re
import shlex
import shutil
import uuid

from PIL import Image

from latentclay.geometry import cut_doorway, dist, gen_line_segment_wall_uvs, make_line_segment_wall, make_polygon_floor, uvs_to_mesh, write_obj_file, update_obj_file_texture


def get_actual_path(arg_path):
    return [[v / 10 for v in pt] for pt in arg_path]


def get_arg_path(actual_path):
    return [[v * 10 for v in pt] for pt in actual_path]


def _patched_exit(self, msg):
    raise argparse.ArgumentError(None, msg)


def create_command_parser():
    parser = argparse.ArgumentParser(exit_on_error=False)
    parser.exit = _patched_exit
    sub = parser.add_subparsers(dest='cmd_name')

    p_create = sub.add_parser('/create', exit_on_error=False)
    p_create.exit = _patched_exit
    p_create.add_argument('desc', nargs='+')
    p_create.add_argument('--reuse-last', action='store_true')
    p_create.add_argument('--x', type=int)
    p_create.add_argument('--z', type=int)
    p_create.add_argument('--ry', type=int)
    p_create.add_argument('--scale', type=float, default=1)

    p_wall = sub.add_parser('/room', exit_on_error=False)
    p_wall.exit = _patched_exit
    p_wall.add_argument('desc', nargs='+')
    p_wall.add_argument('--xz', type=lambda s: [tuple(float(v) for v in p.split(',')) for p in s.split()])

    p_door = sub.add_parser('/door', exit_on_error=False)
    p_door.exit = _patched_exit
    p_door.add_argument('desc', nargs='+')
    p_door.add_argument('--xz', type=lambda s: [tuple(float(v) for v in p.split(',')) for p in s.split()])

    p_undo = sub.add_parser('/undo', exit_on_error=False)
    p_undo.exit = _patched_exit
    
    arg_start_re = re.compile(r'\s--')

    def parse(cmd):
        arg_match = arg_start_re.search(cmd)
        if arg_match is None:
            args = cmd.split(None, 1)
        else:
            start = arg_match.start()
            args = [*cmd[:start].split(None, 1), *shlex.split(cmd[start:])]
        return parser.parse_args(args)

    return parse


def handle_command_sync(id, args, _world_log):
    if args.cmd_name == '/create':
        if args.reuse_last:
            last_id = _world_log[-1][0]
            _world_log.append([id, 'set', 'alias', {'x': args.x, 'z': args.z, 'ry': args.ry, 'id': last_id}])
        else:
            _world_log.append([id, 'set', 'msg', {'x': args.x, 'z': args.z, 'msg': ' '.join(args.desc)}])
    elif args.cmd_name == '/room':
        path = get_actual_path(args.xz)
        mesh_floor = make_polygon_floor(path, 0)
        mesh_ceil = make_polygon_floor(path, 2, rev=True)
        mesh_wall = make_line_segment_wall(path, 2)
        mesh_wall_tex = make_line_segment_wall(path, 2, mirror=True)
        os.makedirs(os.path.join('public', 'output', id), exist_ok=True)
        for typ, mesh, mesh_tex in (('floor', mesh_floor, mesh_floor), ('wall', mesh_wall, mesh_wall_tex), ('ceil', mesh_ceil, mesh_ceil)):
            write_obj_file(os.path.join('public', 'output', id, f'mesh-{typ}'), mesh)
            write_obj_file(os.path.join('public', 'output', id, f'mesh-{typ}-tex'), mesh_tex)
            _world_log.append([f'{id}.{typ}', 'set', typ, {
                'x': 0, 'z': 0, 'path': args.xz, 'obj': f'output/{id}/mesh-{typ}.obj', 'msg': ' '.join(args.desc)
            }])
    elif args.cmd_name == '/undo':
        _world_log.append([_world_log[-1][0], 'del', '', {}])
    else:
        assert args.cmd_name == '/door'
        clearance = 1

        # REVIEW: ignore interior points of path
        path = get_actual_path(args.xz)
        start, end = path[0], path[-2]

        new_logs = []

        for log_id, (typ, data) in _world_log.objects.items():
            if typ != 'wall':
                continue
            wall_id = log_id.split('.')[0]
            path = get_actual_path(data['path'])
            # FIXME TODO: use previous mask
            new_path, mask, _ = cut_doorway(start, end, path, clearance)
            if new_path is None:
                continue

            mesh_wall = make_line_segment_wall(new_path, 2, mask=mask, mirror=True)

            # REVIEW - put edits under same folder
            edit_id = str(uuid.uuid4())
            base_path = os.path.join('public', 'output', wall_id)
            edit_path = os.path.join(base_path, 'edits', edit_id)
            os.makedirs(edit_path, exist_ok=True)
            shutil.copyfile(
                os.path.join(base_path, 'tex_wall.png'),
                os.path.join(edit_path, 'tex_wall.png'),
            )
            write_obj_file(os.path.join(edit_path, f'mesh-wall-tex'), mesh_wall, 'tex_wall.png')

            # TODO handle if tex version is not ready yet??
            new_logs.append([log_id, 'set', 'wall', {
                'x': 0, 'z': 0, 'path': get_arg_path(new_path), 'mask': mask, 
                'obj': f'output/{wall_id}/edits/{edit_id}/mesh-wall-tex.obj',
            }])

        _world_log.extend(new_logs)

        door_mesh = make_line_segment_wall([start, end], 2)
        os.makedirs(os.path.join('public', 'output', id), exist_ok=True)
        write_obj_file(os.path.join('public', 'output', id, f'mesh'), door_mesh)
        _world_log.append([id, 'set', 'door', {
            'x': 0, 'z': 0, 'path': args.xz, 'obj': f'output/{id}/mesh.obj', 'msg': ' '.join(args.desc)
        }])


def run_command(id, args, capabilities, _world_log):
    if args.cmd_name == '/create':
        os.makedirs(os.path.join('public', 'output', id), exist_ok=True)
        capabilities.txt2img.run(
            {'desc': ' '.join(args.desc), 'output': os.path.join('public', 'output', id, 'guide.png')},
            lambda msg: _world_log.append([id, 'set', 'msg', {'x': args.x, 'z': args.z, 'msg': msg}]),
        )
        _world_log.append([id, 'set', 'msg', {'x': args.x, 'z': args.z, 'msg': ' '.join(args.desc), 'img': f'output/{id}/guide.png'}])

        capabilities.img2mesh.run(
            {
                'img': os.path.join('public', 'output', id, 'guide.png'),
                'output': os.path.join('public', 'output', id),
            },
            lambda msg: _world_log.append([id, 'set', 'msg', {'x': args.x, 'z': args.z, 'msg': msg, 'img': f'output/{id}/guide.png'}]),
        )
        _world_log.append([id, 'set', 'obj', {'x': args.x, 'z': args.z, 'ry': args.ry, 'obj': f'output/{id}/mesh.obj', **({'scale': args.scale} if args.scale != 1 else {})}])
    elif args.cmd_name == '/room':
        capabilities.txt2wall_floor.run(
            {
                'desc': ' '.join(args.desc),
                'wall_ar': 12,
                'output_wall': os.path.join('public', 'output', id, 'tex_wall.png'),
                'output_floor': os.path.join('public', 'output', id, 'tex_floor.png'),
                'output_ceil': os.path.join('public', 'output', id, 'tex_ceil.png'),
            },
            lambda msg: _world_log.append([f'{id}.wall', 'set', 'wall', {'x': 0, 'z': 0, 'path': args.xz, 'obj': f'output/{id}/mesh-wall.obj', 'msg': msg}]),
        )
        for typ in ('floor', 'wall', 'ceil'):
            update_obj_file_texture(os.path.join('public', 'output', id, f'mesh-{typ}-tex'), f'tex_{typ}.png')
            _world_log.append([f'{id}.{typ}', 'set', typ, {'x': 0, 'z': 0, 'path': args.xz, 'obj': f'output/{id}/mesh-{typ}-tex.obj'}])
    elif args.cmd_name == '/door':
        clearance = 1

        # REVIEW: ignore interior points of path
        path = get_actual_path(args.xz)
        start, end = path[0], path[-2]

        def wall_candidates():
            for log_id, (typ, data) in _world_log.objects_unmasked.items():
                if typ != 'wall':
                    continue
                path = get_actual_path(data['path'])
                # FIXME TODO: use previous mask
                new_path, mask, dist = cut_doorway(start, end, path, clearance)
                yield log_id, new_path, mask, dist

        log_id, new_path, mask, _ = min(wall_candidates(), key=lambda item: item[3])
        wall_id = log_id.split('.')[0]

        if new_path is not None:
            masked_idx = next((i for i, mval in enumerate(mask) if not mval), None)
            assert masked_idx is not None, 'couldn\'t find masked out portion of wall for door'

            # snap door to chosen wall and align to wall's direction (i.e texture direction)
            if dist(start, new_path[masked_idx]) < dist(start, new_path[masked_idx + 1]):
                start, end = new_path[masked_idx + 1], new_path[masked_idx]
            else:
                start, end = new_path[masked_idx], new_path[masked_idx + 1]

            tex_im = Image.open(os.path.join('public', 'output', wall_id, 'tex_wall.png'))
            target_uvs = list(gen_line_segment_wall_uvs(new_path, 2))
            u_left, v_top, u_right, v_bot = target_uvs[masked_idx]
            # TODO - this assumes the UVs fit within one copy of the texture. Handle tiling, edge cases
            x_left = round(u_left * (tex_im.width - 1)) % tex_im.width
            x_right = round(u_right * (tex_im.width - 1)) % tex_im.width
            y_top = tex_im.height - (round(v_top * (tex_im.height - 1)) % tex_im.height)
            y_bot = tex_im.height - (round(v_bot * (tex_im.height - 1)) % tex_im.height)
            x_mid = (x_left + x_right) // 2
            width = x_right - x_left
            height = y_bot - y_top
            xs_adj = (
                (x_left - 32, x_left + round(width / 8) * 8 + 32)
                if width >= height
                else (x_mid - height // 2, x_mid + math.ceil(height / 2))
            )

            scale = 512 / height
            tex_crop = tex_im.crop((xs_adj[0], y_top, xs_adj[1], y_bot)).resize((
                max(512, round(scale * (xs_adj[1] - xs_adj[0]) / 8) * 8), 512))

            base_img_path = os.path.join('public', 'output', id, 'base.png')
            tex_crop.save(base_img_path)

            capabilities.img2door.run(
                {
                    'desc': ' '.join(args.desc),
                    'img': base_img_path,
                    'x': 256,
                    'width': round(width * scale),
                    'height': round(height * scale * 0.8), # TODO
                    'output_img': os.path.join('public', 'output', id, 'tex_door.png'),
                    'output_uvs': os.path.join('public', 'output', id, 'tex_uvs.json'),
                },
                lambda msg: _world_log.append([
                    id, 'set', 'wall', {'x': 0, 'z': 0, 'path': args.xz, 'obj': f'output/{id}/mesh.obj', 'msg': msg}
                ]),
            )
            with open(os.path.join('public', 'output', id, 'tex_uvs.json')) as f:
                uvs = json.load(f)

            mesh = uvs_to_mesh(uvs, [start[0], 0, start[1]], [end[0], 0, end[1]], [0, 0, 0], [0, 2, 0])
            write_obj_file(os.path.join('public', 'output', id, f'mesh-tex'), mesh, f'tex_door.png')
            _world_log.append([id, 'set', 'door', {'x': 0, 'z': 0, 'path': args.xz, 'obj': f'output/{id}/mesh-tex.obj'}])

        else:
            # TODO - no matching wall for door
            print('no matching wall for door!')

    else:
        assert args.cmd_name == '/undo'
