import PIL.Image
import torch
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM, AutoConfig

from vlac.model.multimodal_encoder.rqvaesigliptransformer import RQVAESIGLIPTransformerConfig
from vlac.model.multimodal_encoder.rqvaesigliptransformer_encoder import RQVAESIGLIPTransformerVisionTower
from vlac.model.multimodal_projector.base_projector import MultimodalProjectorConfig, MultimodalProjector
from vlac.model.text_embedding import TextEmbedding
from vlac.model.vlac_vlm import VLACForCausalLM, VLACVLMConfig


class VLACConfig(PretrainedConfig):
    model_type = "vlac"

    def __init__(
            self,
            llm_type: str = 'meta-llama/Llama-2-7b',
            vision_tower_type: str | dict = './vila-u-7b-256/vision_tower',
            mm_projector_type: str = './vila-u-7b-256/mm_projector',
            text_embeds_type: str = './vila-u-7b-256/text_embeddings',
            text_tokenizer_type: str = './vila-u-7b-256/llm',
            project_multimodal_type: str = 'linear',
            mm_hidden_size: int = 1024,
            hidden_size: int = 4096,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.llm_type = llm_type
        self.vision_tower_type = vision_tower_type
        self.mm_projector_type = mm_projector_type
        self.text_embeds_type = text_embeds_type
        self.text_tokenizer_type = text_tokenizer_type
        self.project_multimodal_type = project_multimodal_type
        self.mm_hidden_size = mm_hidden_size
        self.hidden_size = hidden_size


class VLAC(PreTrainedModel):
    config_class = VLACConfig

    def __init__(self, config: VLACConfig):
        super().__init__(config)
        self.config = config
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
        self.vision_tower = RQVAESIGLIPTransformerVisionTower(self.config.vision_tower_type if is_pretrain else None, config)

    def __load_projector(self):
        config = MultimodalProjectorConfig.from_pretrained(self.config.mm_projector_type)
        self.mm_projector = MultimodalProjector(config, self.config).to(torch.bfloat16)

    def __load_llm(self):
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.config.text_tokenizer_type, trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.config.llm_type,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self.llm.set_input_embeddings(TextEmbedding.from_pretrained(self.config.text_embeds_type).embeds)

    def __wrap_vlm(self):
        config = VLACVLMConfig(self)
        self.vlm = VLACForCausalLM(config)

    @property
    def text_embeds(self):
        return self.llm.get_input_embeddings()

    def forward(self, prompt, vision, max_len: int = 128, generation_nums: int = 1, cfg: float = 3, **_):
        _, inputs_embeds, attention_mask = self.make_multimodal_embeds(prompt, vision)
        image_ids = []
        outputs = self.vlm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=self.vision_tower.image_tokens,
            pad_token_id=self.text_tokenizer.eos_token_id,
            eos_token_id=self.text_tokenizer.eos_token_id,
            image_ids=image_ids,
            cfg=cfg
        )
        out_prompt = self.text_tokenizer.decode(outputs.flatten(), skip_special_tokens=True)
        out_vision = None
        return out_prompt, out_vision

    def make_multimodal_embeds(self, prompt, vision):
        inputs = self.text_tokenizer(prompt, padding=True, return_tensors="pt")
        inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        images = self.prepare_images(vision)
        _, _, attention_mask, _, multimodal_embeds, multimodal_tokens = self.vlm.prepare_for_multimodal(input_ids, None, attention_mask, input_ids, images)
        return multimodal_tokens, multimodal_embeds, attention_mask

    def prepare_images(self, images):
        if isinstance(images, PIL.Image.Image) or isinstance(images, list):
            images = self.vision_tower.image_processor(images, return_tensors="pt")["pixel_values"]
        return images.to(self.vision_tower.device).to(torch.bfloat16)

    def project_img_features(self, features):
        embeds = self.mm_projector(features).to(self.llm.device)
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


AutoConfig.register("vlac", VLACConfig)
AutoModelForCausalLM.register(VLACConfig, VLAC)
