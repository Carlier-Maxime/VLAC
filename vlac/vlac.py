import warnings
from typing import Dict

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM

from vlac.model.multimodal_encoder.rqvaesigliptransformer import RQVAESIGLIPTransformerConfig
from vlac.model.multimodal_encoder.rqvaesigliptransformer_encoder import RQVAESIGLIPTransformerVisionTower
from vlac.model.multimodal_projector.base_projector import MultimodalProjectorConfig, MultimodalProjector
from vlac.model.vlac_vlm import VLACForCausalLM, VLACVLMConfig
from vlac.utils.gpu_memory_utils import track_gpu_memory_usage, print_gpus_memory_usage
from vlac.utils.tokenizer import tokenize_conversation


class VLACConfig(PretrainedConfig):
    model_type = "vlac"

    def __init__(
            self,
            llm_type: str,
            vision_tower_type: str,
            mm_projector_type: str,
            verbose: bool = False,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.llm_type = llm_type
        self.vision_tower_type = vision_tower_type
        self.mm_projector_type = mm_projector_type
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
        config = RQVAESIGLIPTransformerConfig.from_pretrained(self.config.vision_tower_type)
        config.device_map = self.device_map
        self.vision_tower = RQVAESIGLIPTransformerVisionTower(self.config.vision_tower_type, config).to(self.device_map["vision_tower"])

    def __load_projector(self):
        config = MultimodalProjectorConfig.from_pretrained(self.config.mm_projector_type)
        config.device_map = self.device_map
        self.mm_projector = MultimodalProjector(config, self.config).to(self.device_map["mm_projector"]).to(torch.bfloat16)

    def __load_llm(self):
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.config.llm_type, trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.config.llm_type,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=self.device_map["llm"]
        )

    def __wrap_vlm(self):
        config = VLACVLMConfig(self)
        self.vlm = VLACForCausalLM(config)

    def forward(self, prompt, vision, max_len: int = 128, generation_nums: int = 1, cfg: float = 3, **_):
        inputs = self.text_tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}

        input_ids = [
            tokenize_conversation(conversation, self.text_tokenizer, add_generation_prompt=True, image_generation=True).to(self.llm.device).repeat(generation_nums, 1)
            for conversation in [[{"from": "human", "value": prompt}], [{"from": "human", "value": " "}]]
        ]
        max_length = max([input_ids_part.shape[1] for input_ids_part in input_ids])
        input_ids = torch.cat([
            F.pad(input_ids_part, (max_length - input_ids_part.shape[1], 0))
            for input_ids_part in input_ids
        ], dim=0)
        attention_mask = input_ids.ne(0)
        vision_preprocessed = self.vision_tower.image_processor(vision, return_tensors="pt")["pixel_values"].to(self.vision_tower.device).to(torch.bfloat16)
        _, _, attention_mask, _, inputs_embeds, _ = self.vlm.prepare_embeds_for_multimodal(input_ids, None, attention_mask, None, None, vision_preprocessed)
        outputs = self.vlm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_length=max_len, cfg=cfg)
        response = self.text_tokenizer.decode(outputs.flatten(), skip_special_tokens=True).strip()
        print(response)

        img_tokens, img_features = self.vision_tower.rqvaesiglip.encode_image(vision_preprocessed)
        out_vision = self.vision_tower.rqvaesiglip.decode(img_features).to(torch.float32).add_(1).mul_(127.5).clamp_(0, 255).chunk(2)[0]
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

    def encode_images(self, images):
        if isinstance(self.vision_tower, RQVAESIGLIPTransformerVisionTower):
            self.vision_tower.rqvaesiglip.eval()
        else:
            raise NotImplementedError()

        image_features, tokens = self.vision_tower(images, self.llm.vocab_size)
        image_features = self.mm_projector(image_features)

        return image_features, tokens

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs) -> "PreTrainedModel":
        config = kwargs.pop("config", None)
        if config is None:
            config = VLACConfig.from_pretrained(pretrained_model_name_or_path)
        return super().from_pretrained(pretrained_model_name_or_path, config=config, *args, **kwargs)
