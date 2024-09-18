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


def write_point_cloud(image, depth_im, path):
    depth_scaling = 2
    depth_values = np.array(depth_im).flatten() * (image.size[1] / 255) * depth_scaling
    colors = np.array(image).reshape(-1, 3)
    points = np.indices(image.size[::-1]).reshape(2, -1).T

    # Perspective correction
    focal_length = 1.0  # Adjust this value based on your camera model
    points[:, 0] = (points[:, 0] - image.size[0] / 2) * depth_values / focal_length
    points[:, 1] = (points[:, 1] - image.size[1] / 2) * depth_values / focal_length
    points = np.c_[points, depth_values]

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
        ply_file.write("end_header\n")
        for (y, x, z), (r, g, b) in zip(points, colors):
            ply_file.write(f"{x} {SCR_HEIGHT - y} {z} {r} {g} {b}\n")

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

    image = image.resize((800, 600))
    depth_im = depth_im.resize((800, 600))
    
    write_point_cloud(image, depth_im, os.path.join(base_path, 'point_cloud.ply'))


