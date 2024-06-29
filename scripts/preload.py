from diffusers import StableDiffusionPipeline, ControlNetModel
import torch

models = [
    (StableDiffusionPipeline, 'Lykon/dreamshaper-8'),
    (ControlNetModel, 'lllyasviel/control_v11f1p_sd15_depth'),
    (ControlNetModel, 'lllyasviel/control_v11p_sd15_inpaint'),
    (ControlNetModel, 'lllyasviel/control_v11p_sd15_normalbae'),
]

for cls, name in models:
    print('loading', name)
    cls.from_pretrained(name, torch_dtype=torch.float16, variant='fp16')
