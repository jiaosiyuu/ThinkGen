import dotenv

dotenv.load_dotenv(override=True)

import argparse
import os
from typing import List, Tuple

from PIL import Image, ImageOps

import torch
from torchvision.transforms.functional import to_pil_image, to_tensor

from accelerate import Accelerator
from diffusers.hooks import apply_group_offloading
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from ThinkGen.pipelines.pipeline_thinkgen import ThinkGenCoTPipeline
from ThinkGen.models.transformers.transformer_thinkgen import ThinkGenTransformer2DModel



from datetime import datetime
now = datetime.now()
time_str = now.strftime("%Y_%m_%d_%H_%M_%S")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ThinkGen image generation script.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="JSYuuu/ThinkGen",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default=None,
        help="Path to transformer checkpoint.",
    )
    parser.add_argument(
        "--transformer_lora_path",
        type=str,
        default=None,
        help="Path to transformer LoRA checkpoint.",
    )
    parser.add_argument(
        "--mllm",
        type=str,
        default=None,
        # default="Qwen3-VL/Qwen3-VL-8B-Thinking",
        help="Path to transformer checkpoint.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="euler",
        choices=["euler", "dpmsolver++"],
        help="Scheduler to use.",
    )
    parser.add_argument(
        "--num_inference_step",
        type=int,
        default=50,
        help="Number of inference steps."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for generation."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Output image height."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Output image width."
    )
    parser.add_argument(
        "--max_input_image_pixels",
        type=int,
        default=1048576,
        help="Maximum number of pixels for each input image."
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default='bf16',
        choices=['fp32', 'fp16', 'bf16'],
        help="Data type for model weights."
    )
    parser.add_argument(
        "--text_guidance_scale",
        type=float,
        default=4.0,
        help="Text guidance scale."
    )
    parser.add_argument(
        "--image_guidance_scale",
        type=float,
        default=2.0,
        help="Image guidance scale."
    )
    parser.add_argument(
        "--cfg_range_start",
        type=float,
        default=0.0,
        help="Start of the CFG range."
    )
    parser.add_argument(
        "--cfg_range_end",
        type=float,
        default=1.0,
        help="End of the CFG range."
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="a photo of a purple computer keyboard and a blue scissors",
        help="Text prompt for generation."
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar",
        help="Negative prompt for generation."
    )
    parser.add_argument(
        "--input_image_path",
        type=str,
        nargs='+',
        default=None,
        help="Path(s) to input image(s)."
    )
    parser.add_argument(
        "--output_image_path",
        type=str,
        default="vis",
        help="Path to save output image."
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images to generate per prompt."
    )
    parser.add_argument(
        "--enable_model_cpu_offload",
        action="store_true",
        help="Enable model CPU offload."
    )
    parser.add_argument(
        "--enable_sequential_cpu_offload",
        action="store_true",
        help="Enable sequential CPU offload."
    )
    parser.add_argument(
        "--enable_group_offload",
        action="store_true",
        help="Enable group offload."
    )
    parser.add_argument(
        "--enable_teacache",
        action="store_true",
        help="Enable teacache to speed up inference."
    )
    parser.add_argument(
        "--teacache_rel_l1_thresh",
        type=float,
        default=0.05,
        help="Relative L1 threshold for teacache."
    )
    parser.add_argument(
        "--enable_taylorseer",
        action="store_true",
        help="Enable TaylorSeer Caching."
    )
    parser.add_argument(
        "--think",
        action="store_true",
    )
    return parser.parse_args()

def load_pipeline(args: argparse.Namespace, accelerator: Accelerator, weight_dtype: torch.dtype) -> ThinkGenCoTPipeline:
    pipeline = ThinkGenCoTPipeline.from_pretrained(args.model_path, torch_dtype=weight_dtype, trust_remote_code=True, low_cpu_mem_usage=False, ignore_mismatched_sizes=True
    )

    if args.transformer_path:
        print(f"Transformer weights loaded from {args.transformer_path}")
        pipeline.transformer = ThinkGenTransformer2DModel.from_pretrained(
            args.transformer_path,
            torch_dtype=weight_dtype, low_cpu_mem_usage=False
        )

    if args.mllm:
        print(f"mllm weights loaded from {args.mllm}")
        pipeline.mllm = Qwen3VLForConditionalGeneration.from_pretrained(args.mllm, torch_dtype=weight_dtype, )
        pipeline.processor = AutoProcessor.from_pretrained(args.mllm, torch_dtype=weight_dtype, )


    if args.transformer_lora_path:
        print(f"LoRA weights loaded from {args.transformer_lora_path}")
        pipeline.load_lora_weights(args.transformer_lora_path)

    if args.enable_teacache and args.enable_taylorseer:
        print("WARNING: enable_teacache and enable_taylorseer are mutually exclusive. enable_teacache will be ignored.")

    if args.enable_taylorseer:
        pipeline.enable_taylorseer = True
    elif args.enable_teacache:
        pipeline.transformer.enable_teacache = True
        pipeline.transformer.teacache_rel_l1_thresh = args.teacache_rel_l1_thresh

    if args.scheduler == "dpmsolver++":
        from ThinkGen.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
        scheduler = DPMSolverMultistepScheduler(
            algorithm_type="dpmsolver++",
            solver_type="midpoint",
            solver_order=2,
            prediction_type="flow_prediction",
        )
        pipeline.scheduler = scheduler

    if args.enable_sequential_cpu_offload:
        pipeline.enable_sequential_cpu_offload()
    elif args.enable_model_cpu_offload:
        pipeline.enable_model_cpu_offload()
    elif args.enable_group_offload:
        apply_group_offloading(pipeline.transformer, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
        apply_group_offloading(pipeline.mllm, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
        apply_group_offloading(pipeline.vae, onload_device=accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
    else:

        for name, param in pipeline.mllm.named_parameters():
            if param.device.type == "meta":
                print(f"[META PARAM] {name}: shape={param.shape}, dtype={param.dtype}")
        for name, buf in pipeline.mllm.named_buffers():
            if buf.device.type == "meta":
                print(f"[META BUFFER] {name}: shape={buf.shape}, dtype={buf.dtype}")

        pipeline = pipeline.to(accelerator.device)

    return pipeline

def preprocess(input_image_path: List[str] = []) -> Tuple[str, str, List[Image.Image]]:
    """Preprocess the input images."""
    # Process input images
    input_images = None

    if input_image_path:
        input_images = []
        if isinstance(input_image_path, str):
            input_image_path = [input_image_path]

        if len(input_image_path) == 1 and os.path.isdir(input_image_path[0]):
            input_images = [Image.open(os.path.join(input_image_path[0], f)).convert("RGB")
                          for f in os.listdir(input_image_path[0])]
        else:
            input_images = [Image.open(path).convert("RGB") for path in input_image_path]

        input_images = [ImageOps.exif_transpose(img) for img in input_images] # 自动处理图片朝向

    return input_images

def run(args: argparse.Namespace, 
        accelerator: Accelerator, 
        pipeline: ThinkGenCoTPipeline, 
        instruction: str, 
        negative_prompt: str, 
        input_images: List[Image.Image]) -> Image.Image:
    """Run the image generation pipeline with the given parameters."""
    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    results = pipeline(
        prompt=instruction,
        input_images=input_images,
        width=args.width,
        height=args.height,
        num_inference_steps=args.num_inference_step,
        max_sequence_length=2048,
        text_guidance_scale=args.text_guidance_scale,
        image_guidance_scale=args.image_guidance_scale,
        cfg_range=(args.cfg_range_start, args.cfg_range_end),
        negative_prompt=negative_prompt,
        num_images_per_prompt=args.num_images_per_prompt,
        generator=generator,
        output_type="pil",
        think=args.think, 
    )
    return results

def create_collage(images: List[torch.Tensor]) -> Image.Image:
    """Create a horizontal collage from a list of images."""
    max_height = max(img.shape[-2] for img in images)
    total_width = sum(img.shape[-1] for img in images)
    canvas = torch.zeros((3, max_height, total_width), device=images[0].device)
    
    current_x = 0
    for img in images:
        h, w = img.shape[-2:]
        canvas[:, :h, current_x:current_x+w] = img * 0.5 + 0.5
        current_x += w
    
    return to_pil_image(canvas)

def main(args: argparse.Namespace, root_dir: str) -> None:
    """Main function to run the image generation process."""
    # Initialize accelerator
    accelerator = Accelerator(mixed_precision=args.dtype if args.dtype != 'fp32' else 'no')

    # Set weight dtype
    weight_dtype = torch.float32
    if args.dtype == 'fp16':
        weight_dtype = torch.float16
    elif args.dtype == 'bf16':
        weight_dtype = torch.bfloat16

    # Load pipeline and process inputs
    pipeline = load_pipeline(args, accelerator, weight_dtype)

    while True:
        instruction = input("Please describe the image you'd like to generate (or type 'exit' to quit): ")
        if instruction.lower() == 'exit':
            print("Exiting the image generator. Goodbye!")
            break

        input_image_path = []
        while True:
            user_input = input("Please type input_image_path: ")
            if user_input.lower() in ['', 'exit']:
                break
            input_image_path.append(user_input)

        input_image_path = None if len(input_image_path)==0 else input_image_path
        input_images = preprocess(input_image_path)
        # input_images = [None, None]

        # Generate and save image
        results = run(args, accelerator, pipeline, instruction, args.negative_prompt, input_images)
        if input_image_path is None:
            task = "t2i"
        elif len(input_image_path) == 1:
            task = "edit"
        elif len(input_image_path) > 1:
            task = "in-context"

        output_image_path = os.path.join(args.output_image_path, task)
        os.makedirs(output_image_path, exist_ok=True)
        # os.makedirs(os.path.dirname(args.output_image_path), exist_ok=True)

        save_p = instruction[:30].replace(" ", "_").replace(",", "_").replace(".", "_")
        output_image_path = os.path.join(output_image_path, f"{save_p}--{time_str}.png")

        if len(results.images) > 1:
            for i, image in enumerate(results.images):
                image_name, ext = os.path.splitext(output_image_path)
                image.save(f"{image_name}_{i}{ext}")

        vis_images = [to_tensor(image) * 2 - 1 for image in results.images]
        output_image = create_collage(vis_images)

        output_image.save(output_image_path)
        print(f"Image saved to {output_image_path}")
        if args.think:
            print(f"cot & rewrite prompt: \n{results.prompt_cot}")

if __name__ == "__main__":
    root_dir = os.path.abspath(os.path.join(__file__, os.path.pardir))
    args = parse_args()


    main(args, root_dir)




