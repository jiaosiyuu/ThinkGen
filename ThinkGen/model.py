import os
import argparse
import torch
import warnings
from typing import List, Tuple, Union, Dict, Any
from PIL import Image, ImageOps
from accelerate import Accelerator
from diffusers.hooks import apply_group_offloading
from ThinkGen.pipelines.pipeline_thinkgen import ThinkGenCoTPipeline

def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return image
    raise ValueError(f'Invalid image: {image}')

# 辅助函数
def ensure_image_path(image_path: str) -> str:
    if os.path.exists(image_path):
        return image_path
    raise ValueError(f'Invalid image path: {image_path}')

class ThinkGen_Chat:
    def __init__(
        self,
        model_path: str,
        transformer_lora_path: str | None = None,
        dtype: str = 'bf16',
        device: str = "cuda",
        # gen params
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        text_guidance_scale: float = 4.0,
        image_guidance_scale: float = 2.0,
        seed: int = 0,
        # und params
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,

        **kwargs
    ):
        """
        初始化 ThinkGen_Chat
        """
        self.args = argparse.Namespace()
        self.args.model_path = model_path
        self.args.transformer_lora_path = transformer_lora_path
        self.args.dtype = dtype
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.total_pixels = total_pixels
        self.max_new_tokens = max_new_tokens
        self.fps = kwargs.pop('fps', 2)
        self.nframe = kwargs.pop('nframe', 64)


        self.default_gen_kwargs = {
            "height": height,
            "width": width,
            "num_inference_steps": num_inference_steps,
            "text_guidance_scale": text_guidance_scale,
            "image_guidance_scale": image_guidance_scale,
            "seed": seed,
            "cfg_range_start": kwargs.get("cfg_range_start", 0.0),
            "cfg_range_end": kwargs.get("cfg_range_end", 1.0),
            "negative_prompt": kwargs.get("negative_prompt", "(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), fused fingers, messy drawing, broken legs censor, censored, censor_bar"),
            "num_images_per_prompt": kwargs.get("num_images_per_prompt", 1),
            "think": False,
        }

        self.accelerator = Accelerator(mixed_precision=dtype if dtype != 'fp32' else 'no')
        
        self.weight_dtype = torch.float32
        if dtype == 'fp16':
            self.weight_dtype = torch.float16
        elif dtype == 'bf16':
            self.weight_dtype = torch.bfloat16

        self.pipeline = ThinkGenCoTPipeline.from_pretrained(model_path, torch_dtype=self.weight_dtype, trust_remote_code=True, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)

        if transformer_lora_path:
            print(f"Loading LoRA weights from {transformer_lora_path}...")
            self.pipeline.load_lora_weights(transformer_lora_path)

        scheduler_type = kwargs.get("scheduler", "euler")
        if scheduler_type == "dpmsolver++":
            from ThinkGen.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
            scheduler = DPMSolverMultistepScheduler(
                algorithm_type="dpmsolver++",
                solver_type="midpoint",
                solver_order=2,
                prediction_type="flow_prediction",
            )
            self.pipeline.scheduler = scheduler

        enable_model_cpu_offload = kwargs.get("enable_model_cpu_offload", False)
        enable_sequential_cpu_offload = kwargs.get("enable_sequential_cpu_offload", False)
        enable_group_offload = kwargs.get("enable_group_offload", False)

        if enable_sequential_cpu_offload:
            self.pipeline.enable_sequential_cpu_offload()
        elif enable_model_cpu_offload:
            self.pipeline.enable_model_cpu_offload()
        elif enable_group_offload:
            apply_group_offloading(self.pipeline.transformer, onload_device=self.accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
            apply_group_offloading(self.pipeline.mllm, onload_device=self.accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
            apply_group_offloading(self.pipeline.vae, onload_device=self.accelerator.device, offload_type="block_level", num_blocks_per_group=2, use_stream=True)
        else:
            self.pipeline = self.pipeline.to(self.accelerator.device)
        
        torch.cuda.empty_cache()

    def _preprocess_images(self, image_paths: List[str]) -> List[Image.Image]:
        if not image_paths:
            return None
        
        input_images = []
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            img = ImageOps.exif_transpose(img)
            input_images.append(img)
        return input_images

    def _parse_messages(self, messages: List[Dict[str, str]]) -> Tuple[str, List[str]]:
        prompt_parts = []
        image_paths = []
        
        for msg in messages:
            if msg['type'] == 'text':
                prompt_parts.append(msg['value'])
            elif msg['type'] == 'image':
                path = ensure_image_path(msg['value'])
                image_paths.append(path)
            else:
                warnings.warn(f"Unsupported message type: {msg['type']}, skipping.")
        
        instruction = " ".join(prompt_parts)
        return instruction, image_paths

    def generate_image(self, messages: List[Dict[str, str]], **kwargs) -> Any:
        """
        Execute generation.
        Args:
            messages: A list of input messages, e.g. [{"type": "image", "value": "..."}, {"type": "text", "value": "..."}]
            **kwargs: Override default generation parameters (height, width, seed, etc.)
        Returns:
            ThinkGen_ChatPipelineOutput (contains a .images list)
        """
        instruction, image_paths = self._parse_messages(messages)
        input_images = self._preprocess_images(image_paths)

        gen_kwargs = self.default_gen_kwargs.copy()
        gen_kwargs.update(kwargs)

        generator = torch.Generator(device=self.accelerator.device).manual_seed(gen_kwargs['seed'])

        results = self.pipeline(
            prompt=instruction,
            input_images=input_images,
            width=gen_kwargs['width'],
            height=gen_kwargs['height'],
            num_inference_steps=gen_kwargs['num_inference_steps'],
            max_sequence_length=1024,
            text_guidance_scale=gen_kwargs['text_guidance_scale'],
            image_guidance_scale=gen_kwargs['image_guidance_scale'],
            cfg_range=(gen_kwargs['cfg_range_start'], gen_kwargs['cfg_range_end']),
            negative_prompt=gen_kwargs['negative_prompt'],
            num_images_per_prompt=gen_kwargs['num_images_per_prompt'],
            generator=generator,
            output_type="pil",
            think=gen_kwargs['think'],
        )

        return results





    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """s
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        image_content_prepare_func = ensure_image_url
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': image_content_prepare_func(s['value'])}
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
            elif s['type'] == 'video':
                item = {
                    'type': 'video',
                    'video': s['value']
                }
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
                if self.total_pixels is not None:
                    item['total_pixels'] = self.total_pixels
                if self.fps is not None:
                    item['fps'] = self.fps
                elif self.nframe is not None:
                    import cv2
                    video = cv2.VideoCapture(s['value'])
                    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                    video.release()
                    if frame_count < self.nframe:
                        new_frame_count = frame_count // self.FRAME_FACTOR * self.FRAME_FACTOR
                        print(f"use {new_frame_count} for {s['value']}")
                        item['nframes'] = new_frame_count
                    else:
                        item['nframes'] = self.nframe
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            elif s['type'] == 'audio':
                item = {'type':'audio','audio':s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def generate_text(self, message):

        messages = []
        messages.append({'role': 'user', 'content': self._prepare_content(message)})

        inputs = self.pipeline.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.pipeline.device)

        # Inference: Generation of the output
        generated_ids = self.pipeline.mllm.generate(**inputs, **self.generate_kwargs,)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.pipeline.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )



        return output_text




