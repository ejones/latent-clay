import itertools
import argparse
import os.path
import readline
import shlex
import time

from PIL import Image
import numpy as np

from mflux.config.model_config import ModelConfig
from mflux.config.config import Config
from mflux.flux.flux import Flux1
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import tqdm

SCR_HEIGHT = 256
SCR_WIDTH = 336

parser = argparse.ArgumentParser(description='Generate an image based on a prompt.')
parser.add_argument('--model', default='dev')
parser.add_argument('--seed', type=int, default=None, help='Entropy Seed (Default is time-based random-seed)')
parser.add_argument('--steps', type=int, help='Inference Steps', default=None)
parser.add_argument('--guidance', type=float, default=3.5, help='Guidance Scale (Default is 3.5)')


fluxes = {
    'dev': Flux1(
        model_config=ModelConfig.from_alias('dev'),
    ),
    'schnell': Flux1(
        model_config=ModelConfig.from_alias('schnell'),
        local_path='/Users/evan/mflux-use/schnell.q4',
    ),
}

depth_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
depth_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf")

base_path = os.path.dirname(os.path.abspath(__file__))


def predict_depth(image):
    inputs = depth_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        predicted_depth = depth_model(**inputs).predicted_depth

    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    return Image.fromarray(prediction.squeeze().cpu().numpy().astype('u1'))


def get_point_cloud(image, depth_im):
    depth_scaling = 1
    depth_values = np.array(depth_im).flatten() * (image.size[1] / 255) * depth_scaling
    points = np.indices(image.size[::-1]).reshape(2, -1).T
    points = np.c_[points, depth_values]

    # Perspective correction
    depth_norm = points[:, 2] / np.max(points[:, 2])
    points[:, 0] -= (points[:, 0] - image.size[1] / 2) * depth_norm * 0.8
    points[:, 1] -= (points[:, 1] - image.size[0] / 2) * depth_norm * 0.8
    return points
# 
#     with open(path, "w") as ply_file:
#         ply_file.write("ply\n")
#         ply_file.write("format ascii 1.0\n")
#         ply_file.write(f"element vertex {len(points)}\n")
#         ply_file.write("property float x\n")
#         ply_file.write("property float y\n")
#         ply_file.write("property float z\n")
#         ply_file.write("property uchar red\n")
#         ply_file.write("property uchar green\n")
#         ply_file.write("property uchar blue\n")
#         ply_file.write("end_header\n")
#         for (y, x, z), (r, g, b) in zip(points, colors):
#             ply_file.write(f"{x} {SCR_HEIGHT - y} {z} {r} {g} {b}\n")


def create_mesh_from_points(image, points, path, tolerance=5.0):
    """
    Create a mesh by connecting points that are within a certain tolerance.
    """
    colors = np.array(image).reshape(-1, 3)

    edges = []
    width, height = image.size
    for y in range(height):
        for x in range(width):
            current_index = y * width + x
            if x < width - 1:  # Horizontal edge
                edges.append((current_index, current_index + 1))
            if y < height - 1:  # Vertical edge
                edges.append((current_index, current_index + width))

    faces = []
    for y in range(height - 1):
        for x in range(width - 1):
            current_index = y * width + x
            # Create two triangles for each quad
            faces.append((current_index, current_index + 1, current_index + width))
            faces.append((current_index + 1, current_index + width + 1, current_index + width))

    with open(path, "w") as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {len(points)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write(f"element edge {len(edges)}\n")
        ply_file.write("property int vertex1\n")
        ply_file.write("property int vertex2\n")
        ply_file.write(f"element face {len(faces)}\n")
        ply_file.write("property list uchar int vertex_index\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")

        for (y, x, z), (r, g, b) in zip(points, colors):
            ply_file.write(f"{x} {SCR_HEIGHT - y} {z} {r} {g} {b}\n")

        for i, j in edges:
            ply_file.write(f"{i} {j}\n")

        for i, j, k in faces:
            # Calculate the average color of the vertices
            r = (int(colors[i][0]) + int(colors[j][0]) + int(colors[k][0])) // 3
            g = (int(colors[i][1]) + int(colors[j][1]) + int(colors[k][1])) // 3
            b = (int(colors[i][2]) + int(colors[j][2]) + int(colors[k][2])) // 3
            ply_file.write(f"3 {i} {j} {k} {r} {g} {b}\n")


histfile = os.path.join(base_path, '.history')
try:
    readline.read_history_file(histfile)
except FileNotFoundError:
    pass

while (line := input('> ')):
    try:
        arg_idx = line.index(' --')
    except ValueError:
        arg_idx = None
    prompt = line[:arg_idx].strip()
    if not prompt:
        continue
    readline.write_history_file(histfile)
    args = parser.parse_args(shlex.split(line[arg_idx:] if arg_idx is not None else ''))

    steps = args.steps if args.steps is not None else 20 if args.model == 'dev' else 2
    flux_model = fluxes[args.model]
    flux_params = dict(
        seed=int(time.time()) if args.seed is None else args.seed,
        prompt=prompt,
        config=Config(
            num_inference_steps=steps,
            height=SCR_HEIGHT,
            width=SCR_WIDTH,
            guidance=args.guidance,
        )
    )
    try:
        image = fluxes[args.model].generate_image(**flux_params).image
    except KeyboardInterrupt:
        continue

    depth_im = predict_depth(image)

    #image = image.resize((800, 600))
    #depth_im = depth_im.resize((800, 600))
    points = get_point_cloud(image, depth_im)
    create_mesh_from_points(image, points, os.path.join(base_path, 'mesh.ply'), tolerance=5.0)


