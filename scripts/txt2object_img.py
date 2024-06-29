import argparse
import sys
import random

from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler
from PIL import Image, ImageDraw
import numpy as np
import torch

_DEFAULT_DEVICE = (
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert image.shape[0:1] == image_mask.shape[0:1], "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


def main(desc, out_path, steps=20, device=_DEFAULT_DEVICE):
    print(f'{device=}')

    init_image = Image.new('RGB', (512, 512))
    init_image.paste((128, 128, 128), (0, 0, 512, 512))

    draw = ImageDraw.Draw(init_image)
    draw.line([(0, 384), (256, 256), (512, 384), (256, 256), (256, 0)], (100, 100, 100), 3)

    #generator = torch.Generator(device).manual_seed(1)

    mask_image = Image.new('RGB', (512, 512))
    draw = ImageDraw.Draw(mask_image)
    draw.polygon([(128, 128), (128, 386), (256, 450), (386, 386), (386, 128), (256, 50)], (255, 255, 255))

    #mask_image2 = Image.new('RGB', (512, 512))
    #draw = ImageDraw.Draw(mask_image2)
    #draw.polygon([(40, 404), (296, 296), (296, 40), (40, 168)], (255, 255, 255))

    #mask_image2.show()

    torch.manual_seed(random.randint(0, 2 ** 16))

    control_image = make_inpaint_condition(init_image, mask_image)

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, variant='fp16',
    )
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "Lykon/dreamshaper-8", controlnet=controlnet, torch_dtype=torch.float16,
        variant='fp16', safety_checker=None
    )

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    #pipe.enable_model_cpu_offload()

    # generate image
    image = pipe(
        desc,
        num_inference_steps=steps,
        #generator=generator,
        eta=1.0,
        image=init_image,
        mask_image=mask_image,
        control_image=control_image,
    ).images[0]

    image.save(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('desc')
    parser.add_argument('out_path')
    parser.add_argument('--steps', default=20, type=int)
    parser.add_argument('--device', default=_DEFAULT_DEVICE)

    args = parser.parse_args()
    main(
        desc=args.desc,
        out_path=args.out_path,
        steps=args.steps,
        device=args.device,
    )
