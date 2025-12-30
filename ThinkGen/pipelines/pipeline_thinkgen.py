import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import math

from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import copy

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from diffusers.models.autoencoders import AutoencoderKL
from ..models.transformers import ThinkGenTransformer2DModel
from ..models.transformers.repo import ThinkGenRotaryPosEmbed
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline

from dataclasses import dataclass, field

import PIL.Image

from diffusers.utils import BaseOutput

from ThinkGen.pipelines.image_processor import ThinkGenImageProcessor

from ThinkGen.utils.teacache_util import TeaCacheParams

from .lora_pipeline import ThinkGenLoraLoaderMixin


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

from ..cache_functions import cache_init 

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class FMPipelineOutput(BaseOutput):
    images: Union[List[PIL.Image.Image], np.ndarray]
    prompt_cot: Union[List[str], np.ndarray] = field(default_factory=lambda: [""])


def find_next_token_index(tensor, split_seq):
    batch_size, seq_len = tensor.shape
    window_size = split_seq.shape[0]
    windows = tensor.unfold(1, window_size, 1)  # [batch, num_windows, window_size]

    matches = (windows == split_seq)  # [batch, num_windows, window_size]
    matches = matches.all(-1)         # [batch, num_windows]

    first_match_indices = torch.where(matches, torch.arange(matches.size(1)).to(tensor.device), torch.full_like(matches.int(), seq_len))
    first_match_indices = first_match_indices.min(dim=1).values  # [batch_size]

    target_indices = first_match_indices + window_size
    target_indices = torch.where(target_indices < seq_len, target_indices, torch.full_like(target_indices, 0))

    return target_indices 

def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):

    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class ThinkGenCoTPipeline(DiffusionPipeline, ThinkGenLoraLoaderMixin):

    model_cpu_offload_seq = "mllm->transformer->vae"

    def __init__(
        self,
        transformer: ThinkGenTransformer2DModel,
        vae: AutoencoderKL,
        scheduler: FlowMatchEulerDiscreteScheduler,
        mllm: Qwen3VLForConditionalGeneration,
        processor,
    ) -> None:
        """
        Args:
            transformer: The transformer model for image generation.
            vae: The VAE model for image encoding/decoding.
            scheduler: The scheduler for noise scheduling.
            text_encoder: The text encoder model.
            tokenizer: The tokenizer for text processing.
        """
        super().__init__()

        self.register_modules(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            mllm=mllm,
            processor=processor,
            _split_seq=torch.tensor([151668]),
        )
        # self.mllm = mllm
        # self.processor = processor

        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = ThinkGenImageProcessor(vae_scale_factor=self.vae_scale_factor * 2, do_resize=True)
        self.default_sample_size = 128

        self.SYSPROMPT = '''You are a helpful, general-purpose AI assistant with the ability to generate images and understand images.

Your primary goal is to assist the user effectively. When generating/edit an image, provide a clear, one-sentence caption or edit instruction that accurately describes the requested image.
'''


    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator],
        latents: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Prepare the initial latents for the diffusion process.

        Args:
            batch_size: The number of images to generate.
            num_channels_latents: The number of channels in the latent space.
            height: The height of the generated image.
            width: The width of the generated image.
            dtype: The data type of the latents.
            device: The device to place the latents on.
            generator: The random number generator to use.
            latents: Optional pre-computed latents to use instead of random initialization.

        Returns:
            torch.FloatTensor: The prepared latents tensor.
        """
        height = int(height) // self.vae_scale_factor
        width = int(width) // self.vae_scale_factor

        shape = (batch_size, num_channels_latents, height, width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        return latents

    def encode_vae(self, img: torch.FloatTensor) -> torch.FloatTensor:
        """
        Encode an image into the VAE latent space.

        Args:
            img: The input image tensor to encode.

        Returns:
            torch.FloatTensor: The encoded latent representation.
        """
        z0 = self.vae.encode(img.to(dtype=self.vae.dtype)).latent_dist.sample()
        if self.vae.config.shift_factor is not None:
            z0 = z0 - self.vae.config.shift_factor
        if self.vae.config.scaling_factor is not None:
            z0 = z0 * self.vae.config.scaling_factor
        z0 = z0.to(dtype=self.vae.dtype)
        return z0

    def prepare_image(
        self,
        images: Union[List[PIL.Image.Image], PIL.Image.Image],
        batch_size: int,
        num_images_per_prompt: int,
        max_pixels: int,
        max_side_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> List[Optional[torch.FloatTensor]]:
        """
        Prepare input images for processing by encoding them into the VAE latent space.

        Args:
            images: Single image or list of images to process.
            batch_size: The number of images to generate per prompt.
            num_images_per_prompt: The number of images to generate for each prompt.
            device: The device to place the encoded latents on.
            dtype: The data type of the encoded latents.

        Returns:
            List[Optional[torch.FloatTensor]]: List of encoded latent representations for each image.
        """
        if batch_size == 1:
            images = [images]
        latents = []
        for i, img in enumerate(images):
            if img is not None and len(img) > 0:
                ref_latents = []
                for j, img_j in enumerate(img):
                    img_j = self.image_processor.preprocess(img_j, max_pixels=max_pixels, max_side_length=max_side_length)
                    ref_latents.append(self.encode_vae(img_j.to(device=device)).squeeze(0))
            else:
                ref_latents = None
            for _ in range(num_images_per_prompt):
                latents.append(ref_latents)

        return latents
    
    def _get_qwen3_prompt_embeds(
        self,
        prompt: Union[str, List[str]],
        input_images = None,
        device: Optional[torch.device] = None,
        cot = False,
        max_sequence_length: int = 256,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get prompt embeddings from the Qwen3VL text encoder.

        Args:
            prompt: The prompt or list of prompts to encode.
            device: The device to place the embeddings on. If None, uses the pipeline's device.
            max_sequence_length: Maximum sequence length for tokenization.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The prompt embeddings tensor
                - The attention mask tensor

        Raises:
            Warning: If the input text is truncated due to sequence length limitations.
        """
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt

        if cot:
            inputs = self.processor(
                text=prompt,
                images=input_images,
                videos=None,
                padding=True,
                return_tensors="pt",
                max_pixels=262144
            ).to(device)

            generated_dict = self.mllm.generate(**inputs, max_new_tokens=4096, return_dict_in_generate=True, output_hidden_states=True)
            hidden_states = [torch.cat(i[-2:], dim=-1) for i in generated_dict.hidden_states]
            prompt_embeds = torch.cat(hidden_states, dim=1) 

            generated_ids = generated_dict.sequences
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )

            if self._split_seq.device != generated_ids.device:
                self._split_seq = self._split_seq.to(generated_ids.device)

            target_indices = find_next_token_index(generated_ids, self._split_seq)
            assert prompt_embeds.shape[0] == 1

            prompt_embeds = prompt_embeds[:, target_indices[0]: ]
            prompt_attention_mask = torch.ones(1, prompt_embeds.shape[1]).to(torch.int64).to(prompt_embeds.device)
            text_input_ids_cut = generated_ids[:, target_indices[0]: ]


        else:
            text_inputs = self.processor.tokenizer(
                prompt,
                padding="longest",
                max_length=max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids.to(device)
            untruncated_ids = self.processor.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids.to(device)

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
                removed_text = self.processor.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because Gemma can only handle sequences up to"
                    f" {max_sequence_length} tokens: {removed_text}"
                )

            prompt_attention_mask = text_inputs.attention_mask.to(device)
            
            if self._split_seq.device != text_input_ids.device:
                self._split_seq = self._split_seq.to(text_input_ids.device)
            target_indices = find_next_token_index(text_input_ids, self._split_seq)
            max_prefix_tokens = target_indices.max()
            max_L = text_input_ids.shape[1]
            cut_length = max_L - max_prefix_tokens
            
            hidden_states = self.mllm(text_input_ids, attention_mask=prompt_attention_mask, output_hidden_states=True, ).hidden_states
            text_feats_all = torch.cat([hidden_states[-2], hidden_states[-1]], dim = -1)



            text_feats = []
            text_mask_cut = []
            text_input_ids_cut = []
            for b in range(text_feats_all.shape[0]):
                if max_prefix_tokens == target_indices[b]:
                    text_feats.append(text_feats_all[b, target_indices[b]: ])
                    text_mask_cut.append(prompt_attention_mask[b, target_indices[b]: ])
                    text_input_ids_cut.append(text_input_ids[b, target_indices[b]: ])
                else:
                    text_feats.append(text_feats_all[b, target_indices[b]: target_indices[b] + cut_length])
                    text_mask_cut.append(prompt_attention_mask[b, target_indices[b]: target_indices[b] + cut_length])
                    text_input_ids_cut.append(text_input_ids[b, target_indices[b]: target_indices[b] + cut_length])

            prompt_embeds = torch.stack(text_feats)
            prompt_attention_mask = torch.stack(text_mask_cut)
            text_input_ids_cut = torch.stack(text_input_ids_cut)


            output_text = ""

        


        if self.mllm is not None:
            dtype = self.mllm.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, prompt_attention_mask, text_input_ids_cut, output_text
    
    def _apply_chat_template(self, instruction: str, images: List = None, cot: bool = False):
        prompt = [
            {
                "role": "system",
                "content": self.SYSPROMPT,
            },]
        if images is not None:
            instruction = "".join(
                [
                    f"<|vision_start|><|image_pad|><|vision_end|>"
                    for i in range(1, len(images) + 1)
                ]
            ) + instruction
        prompt.append({"role": "user", "content": instruction})
        prompt = self.processor.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=False)
        if cot:
            prompt = prompt+f"<|im_start|>assistant\n<think>\n"
        else:
            prompt = prompt+f"<|im_start|>assistant\n<think>\n</think>{instruction}"
        return prompt

    
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        input_images: Optional[List[torch.Tensor]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 256,
        think = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Encodes the prompt into text encoder hidden states.
        """

        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        assert len(prompt) == 1
        prompt = [self._apply_chat_template(_prompt, input_images, think) for _prompt in prompt]

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask, text_input_ids_cut, output_text = self._get_qwen3_prompt_embeds(
                prompt=prompt,
                input_images=input_images,
                device=device,
                cot = think,
                max_sequence_length=max_sequence_length
            )

        cut_text = self.processor.tokenizer.batch_decode(text_input_ids_cut)
        # assert prompt_ori == cut_text
        # print("text_input_ids_cut: ", cut_text)

        batch_size, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)
        prompt_attention_mask = prompt_attention_mask.view(batch_size * num_images_per_prompt, -1)

        # Get negative embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt if negative_prompt is not None else ""

            # Normalize str to list
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            negative_prompt = [self._apply_chat_template(_negative_prompt, None, False) for _negative_prompt in negative_prompt]

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            negative_prompt_embeds, negative_prompt_attention_mask, negative_text_input_ids_cut, _ = self._get_qwen3_prompt_embeds(
                prompt=negative_prompt,
                device=device,
                max_sequence_length=max_sequence_length,
                cot=False
            )

            negative_cut_text = self.processor.tokenizer.batch_decode(negative_text_input_ids_cut)

            batch_size, seq_len, _ = negative_prompt_embeds.shape
            # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.view(
                batch_size * num_images_per_prompt, -1
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask, output_text
    
    @property
    def num_timesteps(self):
        return self._num_timesteps
    
    @property
    def text_guidance_scale(self):
        return self._text_guidance_scale
    
    @property
    def image_guidance_scale(self):
        return self._image_guidance_scale
    
    @property
    def cfg_range(self):
        return self._cfg_range
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Optional[Union[str, List[str]]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.LongTensor] = None,
        negative_prompt_attention_mask: Optional[torch.LongTensor] = None,
        max_sequence_length: Optional[int] = None,
        callback_on_step_end_tensor_inputs: Optional[List[str]] = None,
        input_images: Optional[List[PIL.Image.Image]] = None,
        num_images_per_prompt: int = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        max_pixels: int = 1024 * 1024,
        max_input_image_side_length: int = 1024,
        align_res: bool = True,
        num_inference_steps: int = 28,
        text_guidance_scale: float = 4.0,
        image_guidance_scale: float = 1.0,
        cfg_range: Tuple[float, float] = (0.0, 1.0),
        attention_kwargs: Optional[Dict[str, Any]] = None,
        timesteps: List[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        verbose: bool = False,
        step_func=None,
        think=False,
    ):

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        self._text_guidance_scale = text_guidance_scale
        self._image_guidance_scale = image_guidance_scale
        self._cfg_range = cfg_range
        self._attention_kwargs = attention_kwargs

        # 2. Define call parameters
        # prompt=[prompt, "A young woman wearing a stra aw hat, standing in a golden wheat field."]
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        (
            prompt_embeds,
            prompt_attention_mask,
            negative_prompt_embeds,
            negative_prompt_attention_mask,
            output_text,
        ) = self.encode_prompt(
            prompt,
            self.text_guidance_scale > 1.0,
            input_images=input_images,
            negative_prompt=negative_prompt,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            max_sequence_length=max_sequence_length,
            think=think,
        )

        dtype = self.vae.dtype
        # 3. Prepare control image
        ref_latents = self.prepare_image(
            images=input_images,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            max_pixels=max_pixels,
            max_side_length=max_input_image_side_length,
            device=device,
            dtype=dtype,
        )

        if input_images is None:
            input_images = []
        
        if len(input_images) == 1 and align_res:
            width, height = ref_latents[0][0].shape[-1] * self.vae_scale_factor, ref_latents[0][0].shape[-2] * self.vae_scale_factor
            ori_width, ori_height = width, height
        else:
            ori_width, ori_height = width, height

            cur_pixels = height * width
            ratio = (max_pixels / cur_pixels) ** 0.5
            ratio = min(ratio, 1.0)

            height, width = int(height * ratio) // 16 * 16, int(width * ratio) // 16 * 16
        
        if len(input_images) == 0:
            self._image_guidance_scale = 1

        # 4. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        freqs_cis = ThinkGenRotaryPosEmbed.get_freqs_cis(
            self.transformer.config.axes_dim_rope,
            self.transformer.config.axes_lens,
            theta=10000,
        )
        
        image = self.processing(
            latents=latents,
            ref_latents=ref_latents,
            prompt_embeds=prompt_embeds,
            freqs_cis=freqs_cis,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            device=device,
            dtype=dtype,
            verbose=verbose,
            step_func=step_func,
        )

        image = F.interpolate(image, size=(ori_height, ori_width), mode='bilinear')

        image = self.image_processor.postprocess(image, output_type=output_type)
        
        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return image, output_text
        else:
            return FMPipelineOutput(images=image, prompt_cot=output_text)

    def processing(
        self,
        latents,
        ref_latents,
        prompt_embeds,
        freqs_cis,
        negative_prompt_embeds,
        prompt_attention_mask,
        negative_prompt_attention_mask,
        num_inference_steps,
        timesteps,
        device,
        dtype,
        verbose,
        step_func=None
    ):
        batch_size = latents.shape[0]

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            num_tokens=latents.shape[-2] * latents.shape[-1]
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        enable_taylorseer = getattr(self, "enable_taylorseer", False)
        if enable_taylorseer:
            model_pred_cache_dic, model_pred_current = cache_init(self, num_inference_steps)
            model_pred_ref_cache_dic, model_pred_ref_current = cache_init(self, num_inference_steps)
            model_pred_uncond_cache_dic, model_pred_uncond_current = cache_init(self, num_inference_steps)
            self.transformer.enable_taylorseer = True
        elif self.transformer.enable_teacache:
            # Use different TeaCacheParams for different conditions
            teacache_params = TeaCacheParams()
            teacache_params_uncond = TeaCacheParams()
            teacache_params_ref = TeaCacheParams()

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if enable_taylorseer:
                    self.transformer.cache_dic = model_pred_cache_dic
                    self.transformer.current = model_pred_current
                elif self.transformer.enable_teacache:
                    teacache_params.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                    self.transformer.teacache_params = teacache_params

                model_pred = self.predict(
                    t=t,
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    freqs_cis=freqs_cis,
                    prompt_attention_mask=prompt_attention_mask,
                    ref_image_hidden_states=ref_latents,
                )
                text_guidance_scale = self.text_guidance_scale if self.cfg_range[0] <= i / len(timesteps) <= self.cfg_range[1] else 1.0
                image_guidance_scale = self.image_guidance_scale if self.cfg_range[0] <= i / len(timesteps) <= self.cfg_range[1] else 1.0
                
                if text_guidance_scale > 1.0 and image_guidance_scale > 1.0:
                    if enable_taylorseer:
                        self.transformer.cache_dic = model_pred_ref_cache_dic
                        self.transformer.current = model_pred_ref_current
                    elif self.transformer.enable_teacache:
                        teacache_params_ref.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                        self.transformer.teacache_params = teacache_params_ref

                    model_pred_ref = self.predict(
                        t=t,
                        latents=latents,
                        prompt_embeds=negative_prompt_embeds,
                        freqs_cis=freqs_cis,
                        prompt_attention_mask=negative_prompt_attention_mask,
                        ref_image_hidden_states=ref_latents,
                    )

                    if enable_taylorseer:
                        self.transformer.cache_dic = model_pred_uncond_cache_dic
                        self.transformer.current = model_pred_uncond_current
                    elif self.transformer.enable_teacache:
                        teacache_params_uncond.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                        self.transformer.teacache_params = teacache_params_uncond

                    model_pred_uncond = self.predict(
                        t=t,
                        latents=latents,
                        prompt_embeds=negative_prompt_embeds,
                        freqs_cis=freqs_cis,
                        prompt_attention_mask=negative_prompt_attention_mask,
                        ref_image_hidden_states=None,
                    )

                    model_pred = model_pred_uncond + image_guidance_scale * (model_pred_ref - model_pred_uncond) + \
                        text_guidance_scale * (model_pred - model_pred_ref)
                elif text_guidance_scale > 1.0:
                    if enable_taylorseer:
                        self.transformer.cache_dic = model_pred_uncond_cache_dic
                        self.transformer.current = model_pred_uncond_current
                    elif self.transformer.enable_teacache:
                        teacache_params_uncond.is_first_or_last_step = i == 0 or i == len(timesteps) - 1
                        self.transformer.teacache_params = teacache_params_uncond

                    model_pred_uncond = self.predict(
                        t=t,
                        latents=latents,
                        prompt_embeds=negative_prompt_embeds,
                        freqs_cis=freqs_cis,
                        prompt_attention_mask=negative_prompt_attention_mask,
                        ref_image_hidden_states=None,
                    )
                    model_pred = model_pred_uncond + text_guidance_scale * (model_pred - model_pred_uncond)

                # a_latents = (latents + (1-self.scheduler.timesteps[i]) * model_pred)#.cpu()
                # a_latents = a_latents.to(dtype=dtype)
                # if self.vae.config.scaling_factor is not None:
                #     a_latents = a_latents / self.vae.config.scaling_factor
                # if self.vae.config.shift_factor is not None:
                #     a_latents = a_latents + self.vae.config.shift_factor
                # a_latents = self.vae.decode(a_latents, return_dict=False)[0]

                # a_latents = self.image_processor.postprocess(a_latents, output_type="pil")
                # a_latents[0].save(f"vis/ex/time_{t.item()}.png")

                
                latents = self.scheduler.step(model_pred, t, latents, return_dict=False)[0]

                latents = latents.to(dtype=dtype)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                
                if step_func is not None:
                    step_func(i, self._num_timesteps)

        if enable_taylorseer:
            del model_pred_cache_dic, model_pred_ref_cache_dic, model_pred_uncond_cache_dic
            del model_pred_current, model_pred_ref_current, model_pred_uncond_current

        latents = latents.to(dtype=dtype)
        if self.vae.config.scaling_factor is not None:
            latents = latents / self.vae.config.scaling_factor
        if self.vae.config.shift_factor is not None:
            latents = latents + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        
        return image

    def predict(
        self,
        t,
        latents,
        prompt_embeds,
        freqs_cis,
        prompt_attention_mask,
        ref_image_hidden_states,
    ):
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        batch_size, num_channels_latents, height, width = latents.shape
        
        optional_kwargs = {}
        if 'ref_image_hidden_states' in set(inspect.signature(self.transformer.forward).parameters.keys()):
            optional_kwargs['ref_image_hidden_states'] = ref_image_hidden_states
        
        model_pred = self.transformer(
            latents,
            timestep,
            prompt_embeds,
            freqs_cis,
            prompt_attention_mask,
            **optional_kwargs
        )
        return model_pred