import argparse
import math
import sys
import random

from diffusers import StableDiffusionControlNetInpaintPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler, DDIMScheduler
from PIL import Image, ImageDraw
import numpy as np
import skimage
import torch

_DEFAULT_DEVICE = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)


def mk_wall_floor_depth_img(e, width=512, wall_height=256, show_wall=True, add_roof=False, add_other_wall=False):
    im = Image.new('RGB', (width, 512))
    floor_height = 256 - wall_height // 2
    if show_wall:
        im.paste((e, e, e), (floor_height, floor_height, width - 1, 512 - floor_height))
    draw = ImageDraw.Draw(im)
    for i in range(floor_height):
        p = i / (floor_height - 1)
        c = round(255 * p + e * (1 - p))
        y = 512 - floor_height + i
        end_x = width - floor_height + i if add_other_wall else width
        draw.line([(floor_height - 1 - i, y), (end_x, y)], (c, c, c), 1)
        draw.line([(floor_height - 1 - i, y), (floor_height - 1 - i, floor_height - 1 - i)], (c, c, c), 1)
        if add_other_wall:
            draw.line([(end_x, y), (end_x, floor_height - 1 - i)], (c, c, c), 1)
        if add_roof:
            draw.line([(floor_height - 1 - i, floor_height - 1 - i), (end_x, floor_height - 1 - i)],
            (c, c, c), 1)
    return im


def mk_floor_grad(e, wall_height=0):
    im = Image.new('RGB', (512, 512))
    draw = ImageDraw.Draw(im)
    if wall_height:
        draw.line([(256, 128), (256, 128 - wall_height)], (e, e, e), 1)
    for i in range(128):
        p = i / 256
        c = round(255 * p + e * (1 - p))
        y = 128 + i
        draw.line([(254 - 2 * i, y), (257 + 2 * i, y)], (c, c, c), 1)
        if wall_height:
            draw.line([(254 - 2 * i, y), (254 - 2 * i, y - wall_height)], (c, c, c), 1)
            draw.line([(254 - 2 * i + 1, y), (254 - 2 * i + 1, y - wall_height)], (c, c, c), 1)
            draw.line([(257 + 2 * i, y), (257 + 2 * i, y - wall_height)], (c, c, c), 1)
            draw.line([(257 + 2 * i + 1, y), (257 + 2 * i + 1, y - wall_height)], (c, c, c), 1)
    for i in range(128):
        p = 0.5 - i / 256
        c = round(255 - (255 - e) * p)
        y = 256 + i
        draw.line([(2 * i, y), (511 - 2 * i, y)], (c, c, c), 1)
    return im


def mk_top_down_grad(end_color, margin=48, show_walls=True):
    im = Image.new('RGB', (512, 512))
    im.paste((end_color,) * 3, (margin, margin, 512 - margin, 512 - margin))
    draw = ImageDraw.Draw(im)
    for i in range(margin):
        p = i / margin
        c = round(end_color * p + 255 * (1 - p))
        if show_walls:
            draw.line([(i, i), (i, 512 - i)], (c, c, c), 1)
            draw.line([(i, i), (512 - i, i)], (c, c, c), 1)
            draw.line([(511 - i, i), (511 - i, 512 - i)], (c, c, c), 1)
            draw.line([(i, 511 - i), (511 - i, 511 - i)], (c, c, c), 1)
    return im


def get_top_down_wall_areas(margin=48):
    return [
        [(0, 512), (margin, 512 - margin), (margin, margin), (0, 0)],
        [(0, 0), (margin, margin), (512 - margin, margin), (512, 0)],
        [(512, 0), (511 - margin, margin), (511 - margin, 512 - margin), (512, 512)],
        [(511, 511), (511 - margin, 511 - margin), (margin, 511 - margin), (0, 511)],
    ]


def get_top_down_floor_area(margin=48):
    return (margin, margin, 512 - margin, 512 - margin)


def get_floor_painting_scene(wall_im, floor_im, margin=64):
    margin_scale = 2
    depth_im = mk_top_down_grad(127, margin=margin)

    #imdata = np.array(depth_im.copy())
    img = Image.new('RGB', (512, 512))

    w_floor = 512 - margin * 2
    h_floor = w_floor * floor_im.height // floor_im.width
    for y in range(margin, 512, h_floor):
        for x in range(margin, 512, w_floor):
            img.paste(floor_im.resize((w_floor, h_floor)), (x, y))
    imdata = np.array(img)

    wall_im_tiled = Image.new('RGB', (wall_im.width * 2, wall_im.height))
    wall_im_tiled.paste(wall_im, (0, 0))
    wall_im_tiled.paste(wall_im, (wall_im.width, 0))

    x_wall_offs = 0
    for area in get_top_down_wall_areas(margin=margin):
        src = np.array(area)
        used_wall_width = min(wall_im.width, (512 * wall_im.height) // (margin * margin_scale))
        dest = np.array([[0, 0], [0, wall_im.height], [used_wall_width, wall_im.height], [used_wall_width, 0]])
        tform = skimage.transform.ProjectiveTransform()
        tform.estimate(src, dest)
        cropped = wall_im_tiled.crop((x_wall_offs, 0, used_wall_width + x_wall_offs, wall_im.height))
        warped = skimage.util.img_as_ubyte(
            skimage.transform.warp(
                np.array(cropped),
                tform,
                output_shape=(512, 512),
            )
        )
        mask = Image.new('RGB', (512, 512))
        ImageDraw.Draw(mask).polygon(area, fill=(255, 255, 255))
        np.putmask(imdata, np.array(mask) == 255, warped)
        x_wall_offs = (x_wall_offs + used_wall_width) % wall_im.width

    def make_mask(band_size):
        mask_im = Image.new('RGB', (512, 512))
        for y in range(margin + h_floor, 512 - margin, h_floor):
            mask_im.paste(
                (255, 255, 255),
                (margin, y - round(h_floor * band_size / 2), 512 - margin, y + round(h_floor * band_size // 2)),
            )
        return mask_im

    imdata = skimage.restoration.inpaint_biharmonic(
        imdata,
        np.array(make_mask(0.1).convert('L')),
        channel_axis=-1,
    )

    img = Image.fromarray(skimage.util.img_as_ubyte(imdata))

    # floor_tile_height = (512 * floor_im.height) // floor_im.width
    # floor_tile = floor_im.resize((512, floor_tile_height))
    # floor_ref_img = Image.new('RGB', (512, 512))
    # floor_mask_src = Image.new('RGB', (512, 512))
    # seam_size = 32

    # #for y in range(0, 512, floor_tile_height):
    # floor_ref_img.paste(floor_tile, (0, 0))
    # #floor_mask_src.paste((255, 255, 255), (0, y - seam_size // 2, 512, y + seam_size // 2))
    # floor_mask_src.paste((255, 255, 255), (0, floor_tile_height, 512, 512))

    # tform = get_floor_painting_transform()
    # warped = skimage.util.img_as_ubyte(skimage.transform.warp(np.array(floor_ref_img), tform, output_shape=(512, 512)))
    # np.putmask(imdata, warped != [0, 0, 0], warped)

    #mask_imdata = skimage.util.img_as_ubyte(skimage.transform.warp(np.array(floor_mask_src), tform, output_shape=(512, 512)))
    #mask_im = mk_top_down_grad(255, margin=margin, show_walls=False)

    strength = 0.6

    return img, depth_im, make_mask(4 / 5), strength


def get_floor_painting_transform(margin=0):
    src = np.array([[255, 128 + margin], [margin * 2, 256], [255, 384 - margin], [512 - margin * 2, 256]])
    dest = np.array([[0, 0], [0, 512], [512, 512], [512, 0]])

    tform = skimage.transform.ProjectiveTransform()
    tform.estimate(src, dest)
    return tform


def get_floor_area(width, wall_height, margin=8, half=False, warp_factor=0.0):
    floor_height = 256 - wall_height // 2
    warp = width * warp_factor
    if half:
        return [
            (floor_height // 2 - 1 + margin + warp, 512 - (floor_height + margin) // 2),
            (margin * 2, 511 - margin),
            (width - 1 - margin * 2, 511 - margin),
            (width - floor_height  // 2 - margin - warp, 512 - (floor_height + margin) // 2),
        ]
    return [
        (floor_height - 1, 512 - floor_height + margin),
        (margin * 2, 511 - margin),
        (width - 1 - margin * 2, 511 - margin),
        (width - floor_height, 512 - floor_height + margin),
    ]


def get_ceil_area(width, wall_height, warp_factor=0):
    margin = 8
    floor_height = 256 - wall_height // 2
    warp = width * warp_factor
    return [
        (margin * 2, margin),
        (floor_height - 1 + warp, floor_height - 1 - margin),
        (width - floor_height - warp, floor_height - 1 - margin),
        (width - 1 - margin * 2, margin),
    ]


def get_transform(width, area):
    src = np.array([[0, 0], [0, 256], [width, 256], [width, 0]])
    dest = np.array(area)

    tform = skimage.transform.ProjectiveTransform()
    tform.estimate(src, dest)
    return tform


def get_wall(img, width, wall_height, has_other_wall=False):
    margin = 2
    floor_height = 256 - wall_height // 2
    right_space = floor_height if has_other_wall else 0
    img = img.crop(((floor_height + margin, floor_height + margin, width - 1 - right_space - margin, 512 - floor_height - margin)))
    return img.resize((width - floor_height - right_space, wall_height))


def get_stitch_mask_vertical(imwidth, size):
    return np.concatenate(
        (
            np.zeros((256 - size // 2, imwidth), 'bool'),
            np.ones((size, imwidth), 'bool'),
            np.zeros((256 - math.ceil(size // 2), imwidth), 'bool'),
        ),
    )


def partial_stitch_vertical(imdata, stitch_size=10):
    new_im = np.concatenate((imdata[-256:, :, :], imdata[:256, :, :]))
    mask = get_stitch_mask_vertical(imdata.shape[1], stitch_size)
    return skimage.restoration.inpaint_biharmonic(new_im, mask, channel_axis=-1)


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


# TODO: configurable wall height
def dolly_out(wall_floor_im, add_roof=False):
    w_orig, h_orig = wall_floor_im.size
    wall_height = 128
    floor_height = 256 - wall_height // 2
    width = min(128 + w_orig // 2 + floor_height, 1280)

    depth_im = mk_wall_floor_depth_img(127, width, wall_height=wall_height, add_other_wall=True, add_roof=add_roof)
    mask_im = mk_wall_floor_depth_img(255, width, wall_height=wall_height, show_wall=False, add_other_wall=True, add_roof=add_roof)

    im = Image.new('RGB', depth_im.size)
    inset_im = wall_floor_im.resize((w_orig // 2, h_orig // 2))
    im.paste(inset_im, (128, 128))

    #inset_w, inset_h = inset_im.size
    #depth_im = im.copy()
    #depth_im.paste((0, 0, 0), (128, 128, inset_w + 128, inset_h + 128))

    #mask_im = Image.new('L', im.size)
    #mask_im.paste(255, (0, 0, *im.size))
    #mask_im.paste(0, (132, 132, im.width, inset_h + 124))


    return im, depth_im, mask_im


def dolly_out_floor_ceil(wall_floor_im, wall_height):
    floor_area = get_floor_area(wall_floor_im.width, wall_height, margin=0, half=True)
    x_min = min(x for x, y in floor_area)
    x_max = max(x for x, y in floor_area)
    y_min = min(y for x, y in floor_area)
    y_max = max(y for x, y in floor_area)
    ar = (x_max - x_min) / (y_max - y_min)
    new_width = floor_area[-1][0] - floor_area[0][0]
    floor_cut = wall_floor_im.crop((x_min, y_min, x_max, y_max)).resize((new_width, round(new_width / ar)))

    new_im = wall_floor_im.copy()
    new_floor_pos = floor_area[0][0], floor_area[0][1] - floor_cut.height
    new_im.paste(floor_cut, new_floor_pos)

    mask_im = mk_wall_floor_depth_img(
        255,
        wall_floor_im.width,
        wall_height=wall_height, 
        show_wall=False,
        add_roof=False,
        add_other_wall=False,
    )
    mask_im.paste(
        (0, 0, 0),
        (
            new_floor_pos[0] + 5,
            new_floor_pos[1] + 5,
            new_floor_pos[0] + floor_cut.width - 5,
            new_floor_pos[1] + floor_cut.height - 5,
        )
    )
    mask_im.paste(
        (0, 0, 0),
        (
            floor_area[0][0],
            round(floor_area[0][1] * 0.3  + floor_area[1][1] * 0.7),
            floor_area[2][0],
            floor_area[2][1],
        )
    )
        
    return new_im, mask_im


# TODO wall height configurable
def main(
    desc,
    out_path_wall,
    out_path_floor,
    out_path_ceil,
    steps,
    end_depth_rgb=127,
    wall_ar=8,
    dolly_steps=10,
    negative_prompt='opening, hole, window, door, archway, portal, shadows, lighting',
    device=_DEFAULT_DEVICE,
):
    seed = random.randint(0, 2 ** 16)
    torch.manual_seed(seed)

    print(f'{seed=} {device=}')

    controlnet_depth = ControlNetModel.from_pretrained(
        'lllyasviel/control_v11f1p_sd15_depth', torch_dtype=torch.float16, variant='fp16',
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "Lykon/dreamshaper-8",
        controlnet=controlnet_depth,
        torch_dtype=torch.float16,
        variant='fp16',
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    width = wall_ar * 128 + 192
    print('total generation:', width, 'x', 512)
    depth_guide_img = mk_wall_floor_depth_img(end_depth_rgb, width, wall_height=128, add_roof=True)

    #depth_guide_img.show()

    wall_floor_img = pipe(
        prompt=desc,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        num_images_per_prompt=1,
        image=depth_guide_img,
    ).images[0]

    #wall_floor_img.show()

    wall_img = get_wall(wall_floor_img, width, 128)
    wall_img.save(out_path_wall)

    floor_area = get_floor_area(wall_floor_img.width, 128, half=True)
    floor_tform = get_transform(wall_floor_img.width, floor_area)

    floor_imdata = skimage.transform.warp(np.array(wall_floor_img), floor_tform, output_shape=(256, width))

    floor_img = Image.fromarray(skimage.util.img_as_ubyte(floor_imdata))
    #floor_img.show()

    ceil_area = get_ceil_area(wall_floor_img.width, 128)
    ceil_tform = get_transform(wall_floor_img.width, ceil_area)

    ceil_imdata = skimage.transform.warp(np.array(wall_floor_img), ceil_tform, output_shape=(256, width))

    ceil_img = Image.fromarray(skimage.util.img_as_ubyte(ceil_imdata))
    ceil_img.save(out_path_ceil)

    # dolly_img, dolly_mask = dolly_out_floor_ceil(wall_floor_img, 128)
    floor_painting_margin = 64
    dolly_img, dolly_depth, dolly_mask, dolly_strength = get_floor_painting_scene(
        wall_img, floor_img, margin=floor_painting_margin,
    )
    #dolly_img.show()
    #dolly_mask.show()

    controlnet_inpaint = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, variant='fp16',
    )

    pipe = StableDiffusionControlNetInpaintPipeline(**{
        **pipe.components,
        'controlnet': [
            controlnet_inpaint,
            controlnet_depth,
            #controlnet_normal,
        ],
    })
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    dolly_painted_img = pipe(
        f'{desc}, top-down, overhead, floor, empty',
        num_inference_steps=steps,
        eta=1.0,
        image=dolly_img,
        mask_image=dolly_mask,
        control_image=[
            make_inpaint_condition(dolly_img, dolly_mask),
            #depth_guide_img,
            dolly_depth,
        ],
        negative_prompt=negative_prompt,
        strength=dolly_strength,
    ).images[0]

    expanded_floor_img = dolly_painted_img.crop(get_top_down_floor_area(margin=floor_painting_margin))
    #expanded_floor_img.show()

    expanded_floor_img.save(out_path_floor)

# floor_area = get_floor_area(wall_floor_img.width, 128)
# floor_tform = get_transform(wall_floor_img.width, floor_area)
# 
# # dolly_draw = ImageDraw.Draw(dolly_painted_img)
# # dolly_draw.polygon(floor_area, outline=(255, 0, 0))
# # dolly_painted_img.show()
# 
# floor_imdata = skimage.transform.warp(np.array(dolly_painted_img), floor_tform, output_shape=(256, width))
# floor_img = Image.fromarray(skimage.util.img_as_ubyte(floor_imdata))
# 
# #floor_img.show()
# 
# floor_img.save(out_path_floor)
# 
# ceil_img = Image.fromarray(skimage.util.img_as_ubyte(ceil_imdata))
# ceil_img.save(out_path_ceil)

# floor_painting_im, floor_depth, floor_mask = get_floor_painting_scene(wall_img, floor_img)
# 
# #floor_painting_im.show()
# #floor_depth.show()
# #floor_mask.show()
# 
# 
# # controlnet_normal = ControlNetModel.from_pretrained(
# #     'lllyasviel/control_v11p_sd15_normalbae', torch_dtype=torch.float16, variant='fp16', 
# # )
# 
# #floor_painted_img.show()
# 
# floor_tform = get_floor_painting_transform(margin=5)
# floor_imdata = skimage.transform.warp(np.array(floor_painted_img), floor_tform.inverse)
# floor_img = Image.fromarray(skimage.util.img_as_ubyte(floor_imdata))
# floor_img.save(out_path_floor)

# TODO: wall height configurable
# floor_area = get_floor_area(dolly_painted_img.width, 128)
# ceil_area = get_ceil_area(dolly_painted_img.width, 128)

# floor_tform = get_transform(dolly_painted_img.width, floor_area)
# floor_imdata = skimage.transform.warp(np.array(dolly_painted_img), floor_tform)
# floor_img = Image.fromarray(skimage.util.img_as_ubyte(floor_imdata))
# floor_img.save(out_path_floor)

# ceil_tform = get_transform(dolly_painted_img.width, ceil_area)
# ceil_imdata = skimage.transform.warp(np.array(dolly_painted_img), ceil_tform)
# ceil_img = Image.fromarray(skimage.util.img_as_ubyte(ceil_imdata))
# ceil_img.save(out_path_ceil)

# tform = get_floor_transform(width)
# floor_imdata = skimage.transform.warp(np.array(wall_floor_img), tform)
# floor_imdata = partial_stitch_vertical(floor_imdata)

# stitch_imdata = np.array(wall_floor_img.copy())
# floor_rewarp = skimage.util.img_as_ubyte(skimage.transform.warp(floor_imdata, tform.inverse))
# np.putmask(stitch_imdata, floor_rewarp != [0, 0, 0], floor_rewarp)

# stitch_img = Image.fromarray(stitch_imdata)
# stitch_mask = Image.fromarray(
#     skimage.util.img_as_ubyte(
#         skimage.transform.warp(
#             get_stitch_mask_vertical(width, 92),
#             tform.inverse
#         )
#     )
# )

# stitch_img.show()
# #stitch_mask.show()
# pipe = StableDiffusionControlNetInpaintPipeline(**{
#     **pipe.components,
#     'controlnet': controlnet_inpaint,
# })
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# pipe.to(device)

# control_image = make_inpaint_condition(stitch_img, stitch_mask)

# wall_floor_stitched_img = pipe(
#     desc,
#     num_inference_steps=steps,
#     eta=1.0,
#     image=stitch_img,
#     mask_image=stitch_mask,
#     control_image=control_image,
#     negative_prompt=negative_prompt,
#     strength=0.9,
# ).images[0]

# #wall_floor_stitched_img.show()

# floor_stitched_img = Image.fromarray(
#     skimage.util.img_as_ubyte(skimage.transform.warp(np.array(wall_floor_stitched_img), tform))
# )

# floor_stitched_img.save(out_path_floor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('desc')
    parser.add_argument('out_path_wall')
    parser.add_argument('out_path_floor')
    parser.add_argument('out_path_ceil')
    parser.add_argument('wall_ar', type=int)
    parser.add_argument('--steps', default=16, type=int)
    parser.add_argument('--device', default=_DEFAULT_DEVICE)

    args = parser.parse_args()
    main(
        desc=args.desc,
        out_path_wall=args.out_path_wall,
        out_path_floor=args.out_path_floor,
        out_path_ceil=args.out_path_ceil,
        wall_ar=args.wall_ar,
        steps=args.steps,
        device=args.device,
    )

