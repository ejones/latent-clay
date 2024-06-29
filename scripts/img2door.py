import argparse
import json
import math
import sys
import random

import cv2
from diffusers import StableDiffusionControlNetInpaintPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, DDIMScheduler
from PIL import Image, ImageDraw
import numpy as np
import skimage
import torch
import tripy

_DEFAULT_DEVICE = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)


def get_door_inpainting(base_img, x, size, border_scale=.15):
    base_img = base_img.copy()
    w, h = size
    h_img = base_img.height
    thickness = min(w // 2 - 8, round(h_img * border_scale))
    margin = 5 # REVIEW
    w -= thickness + margin * 2
    x_start, x_end = x - w // 2, x + w // 2
    base_img.paste((0, 0, 0), (x_start, h_img - h, x_end, h_img))
    mask_img = Image.new('RGB', base_img.size)
    draw = ImageDraw.Draw(mask_img)
    draw.line(
        [(x_start, h_img), (x_start, h_img - h), (x_end, h_img - h), (x_end, h_img)],
        (255, 255, 255),
        thickness,
        'curve'
    )

    return base_img, mask_img


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


def polygonize(flood_mask):
    ctrs, _ = cv2.findContours(
        skimage.morphology.binary_closing(~flood_mask).astype('u1'),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    ctr = max(ctrs, key=len)
    epsilon = 0.0003 * cv2.arcLength(ctr, True)
    approx = cv2.approxPolyDP(ctr, epsilon, True)
    return list(map(tuple, approx.reshape(-1, 2).tolist()))


def main(
    desc,
    base_img_path,
    out_path_img,
    out_path_uvs,
    position_x,
    size,
    steps,
    device=_DEFAULT_DEVICE,
):
    seed = random.randint(0, 2 ** 16)
    print(f'{seed=}, {device=}')
    torch.manual_seed(seed)

    ref_img = Image.open(base_img_path)
    ref_img, mask_img = get_door_inpainting(ref_img, position_x, size)
    assert size[0] < ref_img.width

    # base_img, mask_img = (
    #     im.crop(
    #         (
    #             position_x - im.height // 2,
    #             0,
    #             position_x - im.height // 2 + im.height,
    #             im.height,
    #         )
    #     ).resize((512, 512))
    #     for im in (ref_img, mask_img)
    # )

    #base_img.show()
    #mask_img.show()

    controlnet_inpaint = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, variant='fp16',
    )

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "Lykon/dreamshaper-8",
        controlnet=controlnet_inpaint,
        torch_dtype=torch.float16,
        variant='fp16',
        safety_checker=None,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    control_img = make_inpaint_condition(ref_img, mask_img)

    inpainted = pipe(
        prompt=desc,
        num_inference_steps=steps,
        eta=1.0,
        num_images_per_prompt=1,
        image=ref_img,
        mask_image=mask_img,
        control_image=control_img,
        strength=1.0,
    ).images[0]

    # inpainted.show()

    # ref_img.paste(
    #     inpainted.resize((ref_img.height, ref_img.height)),
    #     (position_x - ref_img.height // 2, 0),
    # )
    ref_img = inpainted.crop(
        (
            (ref_img.width - size[0]) // 2, 0,
            (ref_img.width - size[0]) // 2 + size[0], ref_img.height
        ),
    )
    #ref_img.show()

    im_hsv = skimage.color.rgb2hsv(np.array(ref_img))
    entrance_mask = skimage.segmentation.flood(
        im_hsv[..., 2], (ref_img.height - 1, position_x), tolerance=.125,
    )
    imdata = np.array(ref_img.convert('RGBA'))
    imdata[entrance_mask] = [0, 0, 0, 0]

    poly = polygonize(entrance_mask)
    tri_points = tripy.earclip(poly)

    print(len(tri_points), 'triangles')

    # poly_prvw = Image.new('L', ref_img.size)
    # poly_draw = ImageDraw.Draw(poly_prvw)
    # for tri in tri_points:
    #     poly_draw.polygon(poly, outline=255)
    # poly_prvw.show()

    Image.fromarray(imdata).save(out_path_img)
    with open(out_path_uvs, 'w') as f:
        json.dump([
            [x / ref_img.width, (ref_img.height - y) / ref_img.height]
            for tri in tri_points
            for x, y in tri
        ], f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('desc')
    parser.add_argument('base_img_path')
    parser.add_argument('position_x', type=int)
    parser.add_argument('size_x', type=int)
    parser.add_argument('size_y', type=int)
    parser.add_argument('out_path_img')
    parser.add_argument('out_path_uvs')
    parser.add_argument('--steps', default=10, type=int)
    parser.add_argument('--device', default=_DEFAULT_DEVICE)

    args = parser.parse_args()
    main(
        desc=args.desc,
        base_img_path=args.base_img_path,
        position_x=args.position_x,
        size=(args.size_x, args.size_y),
        out_path_img=args.out_path_img,
        out_path_uvs=args.out_path_uvs,
        steps=args.steps,
        device=args.device,
    )
