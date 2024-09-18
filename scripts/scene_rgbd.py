import argparse
import time
import shlex
import readline

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

    return prediction.squeeze().cpu().numpy().astype('u1')


while (line := input('> ')):
    try:
        arg_idx = line.index(' --')
    except ValueError:
        arg_idx = None
    prompt = line[:arg_idx].strip()
    if not prompt:
        continue
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
      
    image.show()

    depth_im = predict_depth(image)

    Image.fromarray(depth_im.astype('u1')).show()

