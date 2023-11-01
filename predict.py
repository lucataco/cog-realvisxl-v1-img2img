from cog import BasePredictor, Input, Path
import os
import time
import math
import torch
from PIL import Image
from diffusers import (DDIMScheduler, 
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler, 
    EulerDiscreteScheduler,
    HeunDiscreteScheduler, 
    PNDMScheduler
    )

MODEL_NAME = "SG161222/RealVisXL_V1.0"
MODEL_CACHE = "model-cache"

SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
}

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        t1 = time.time()
        print("Loading sdxl txt2img pipeline...")
        self.img2img_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.img2img_pipe.to("cuda")
        t2 = time.time()
        print("Setup sdxl took: ", t2 - t1)

    def scale_down_image(self, image_path, max_size):
        image = Image.open(image_path)
        width, height = image.size
        scaling_factor = min(max_size/width, max_size/height)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        resized_image = image.resize((new_width, new_height))
        cropped_image = self.crop_center(resized_image)
        return cropped_image

    def crop_center(self, pil_img):
        img_width, img_height = pil_img.size
        crop_width = self.base(img_width)
        crop_height = self.base(img_height)
        return pil_img.crop(
                (
                    (img_width - crop_width) // 2,
                    (img_height - crop_height) // 2,
                    (img_width + crop_width) // 2,
                    (img_height + crop_height) // 2)
                )

    def base(self, x):
        return int(8 * math.floor(int(x)/8))

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(description="Input image"),
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a rainbow unicorn",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="(worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth",
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="DPMSolverMultistep",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=100, default=40
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=10, default=8
        ),
        strength: float = Input(
            description="strength/weight", ge=0, le=1, default=0.9
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        generator = torch.Generator("cuda").manual_seed(seed)

        pipe = self.img2img_pipe
        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)
        
        resized_image = self.scale_down_image(image, 1024)

        common_args = {
            "image": resized_image,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "strength": strength,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
        }
        output = pipe(**common_args)

        output_path = f"/tmp/output.png"
        output.images[0].save(output_path)

        return Path(output_path)