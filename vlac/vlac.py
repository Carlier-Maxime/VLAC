from typing import Dict

import torch
import PIL.Image
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM

from vlac.model.multimodal_encoder.rqvaesigliptransformer import RQVAESIGLIPTransformerConfig
from vlac.model.multimodal_encoder.rqvaesigliptransformer_encoder import RQVAESIGLIPTransformerVisionTower
from vlac.model.multimodal_projector.base_projector import MultimodalProjectorConfig, MultimodalProjector
from vlac.model.text_embedding import TextEmbedding
from vlac.model.vlac_vlm import VLACForCausalLM, VLACVLMConfig
from vlac.utils.gpu_memory_utils import track_gpu_memory_usage, print_gpus_memory_usage


class VLACConfig(PretrainedConfig):
    model_type = "vlac"

    def __init__(
            self,
            llm_type: str = 'meta-llama/Llama-2-7b',
            vision_tower_type: str | dict = './vila-u-7b-256/vision_tower',
            mm_projector_type: str = './vila-u-7b-256/mm_projector',
            text_embeds_type: str = './vila-u-7b-256/text_embeddings',
            text_tokenizer_type: str = './vila-u-7b-256/llm',
            verbose: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.llm_type = llm_type
        self.vision_tower_type = vision_tower_type
        self.mm_projector_type = mm_projector_type
        self.text_embeds_type = text_embeds_type
        self.text_tokenizer_type = text_tokenizer_type
        self.verbose = verbose


class VLAC(PreTrainedModel):
    config_class = VLACConfig

    def __init__(self, config: VLACConfig):
        super().__init__(config)
        self.config = config
        self.device_map: Dict = config.device_map
        if config.verbose:
            print("\n=== Loading Model =============")
            print_gpus_memory_usage("Initial")
            track_gpu_memory_usage(self.__load_vision_tower, self.device_map["vision_tower"])
            track_gpu_memory_usage(self.__load_projector, self.device_map["mm_projector"])
            track_gpu_memory_usage(self.__load_llm, self.device_map["llm"])
            self.__wrap_vlm()
            print_gpus_memory_usage("Final")
            print("===============================\n")
        else:
            self.__load_vision_tower()
            self.__load_projector()
            self.__load_llm()
            self.__wrap_vlm()

    def __load_vision_tower(self):
        is_pretrain = isinstance(self.config.vision_tower_type, str)
        if is_pretrain:
            config = RQVAESIGLIPTransformerConfig.from_pretrained(self.config.vision_tower_type)
        else:
            config = RQVAESIGLIPTransformerConfig.from_dict(self.config.vision_tower_type)
        config.device_map = self.device_map
        self.vision_tower = RQVAESIGLIPTransformerVisionTower(self.config.vision_tower_type if is_pretrain else None, config).to(self.device_map["vision_tower"])

    def __load_projector(self):
        config = MultimodalProjectorConfig.from_pretrained(self.config.mm_projector_type)
        config.device_map = self.device_map
        self.mm_projector = MultimodalProjector(config, self.config).to(self.device_map["mm_projector"]).to(torch.bfloat16)

    def __load_llm(self):
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.config.text_tokenizer_type, trust_remote_code=True)
        device = self.device_map["llm"]
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.config.llm_type,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=device
        )
        print(f"{TextEmbedding} : {self.config.text_embeds_type}")
        self.text_embeds = TextEmbedding.from_pretrained(self.config.text_embeds_type, trust_remote_code=True).to(device)
        self.llm.set_input_embeddings(self.text_embeds.embeds)

    def __wrap_vlm(self):
        config = VLACVLMConfig(self)
        self.vlm = VLACForCausalLM(config)
        self.llm.to(self.device_map["llm"])

    def forward(self, prompt, vision, max_len: int = 128, generation_nums: int = 1, cfg: float = 3, **_):
        inputs_embeds, attention_mask = self.make_multimodal_embeds(prompt, vision)
        outputs = self.vlm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=self.vision_tower.image_tokens, cfg=cfg)
        response = self.text_tokenizer.decode(outputs.flatten(), skip_special_tokens=True).strip()
        print(response)

        out_vision = self.encode_decode_images(vision)[0]
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_length=max_len,
                num_return_sequences=1,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=self.text_tokenizer.eos_token_id,
                eos_token_id=self.text_tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                num_beams=2
            )
        response = self.text_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response, out_vision

    def make_multimodal_embeds(self, prompt, vision):
        inputs = self.text_tokenizer(prompt, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        images = self.prepare_images(vision)
        _, _, attention_mask, _, inputs_embeds, _ = self.vlm.prepare_embeds_for_multimodal(input_ids, None, attention_mask, None, None, images)
        return inputs_embeds, attention_mask

    def prepare_images(self, images):
        if isinstance(images, PIL.Image.Image) or isinstance(images, list):
            images = self.vision_tower.image_processor(images, return_tensors="pt")["pixel_values"]
        return images.to(self.vision_tower.device).to(torch.bfloat16)

    def project_img_features(self, features):
        device = self.device_map["mm_projector"]
        self.mm_projector.to(device)
        embeds = self.mm_projector(features.to(device)).to(self.llm.device)
        return embeds.reshape(embeds.shape[0], -1, embeds.shape[-1])

    def unproject_img_features(self, embeds):
        image_hidden_states, code = self.vision_tower.rqtransformer.generate(embeds.to(self.vision_tower.device), self.vision_tower.rqvaesiglip)
        N, T, C = image_hidden_states.shape
        sT = int(T**0.5)
        image_hidden_states = image_hidden_states.reshape(N, sT, sT, C)
        code = code.reshape(N, sT, sT, code.shape[-1])
        return image_hidden_states, code

    def encode_images(self, images):
        images = self.prepare_images(images)
        image_features, tokens = self.vision_tower(images, self.llm.vocab_size)
        return image_features, tokens

    def encode_decode_images(self, images, with_proj: bool = False):
        img_features, img_tokens = self.encode_images(images)
        img_embeds = self.project_img_features(img_features) if with_proj else None
        img_hidden_states, code = self.unproject_img_features(img_embeds) if with_proj else None, None
        out_vision = self.vision_tower.decode_features(img_hidden_states if with_proj else img_features)
        return out_vision, img_hidden_states, img_embeds, img_features, img_tokens

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs) -> "PreTrainedModel":
        config = kwargs.pop("config", None)
        if config is None:
            config = VLACConfig.from_pretrained(pretrained_model_name_or_path)
        return super().from_pretrained(pretrained_model_name_or_path, config=config, *args, **kwargs)
