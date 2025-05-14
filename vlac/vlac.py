from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
import torch

from vlac.gpu_memory_utils import print_gpu_memory_usage, track_gpu_memory_usage
from vlac.model.multimodal_encoder.rqvaesigliptransformer import RQVAESIGLIPTransformerConfig
from vlac.model.multimodal_encoder.rqvaesigliptransformer_encoder import RQVAESIGLIPTransformerVisionTower


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
        self.text_tokenizer = None
        self.llm = None
        self.vision_tower = None
        self.mm_projector = None
        self.device_map = config.device_map
        if config.verbose:
            print("\n=== Loading Model =============")
            print_gpu_memory_usage(self.device_map, "Initial")
            track_gpu_memory_usage(self.load_vision_tower)
            track_gpu_memory_usage(self.load_llm)
            print_gpu_memory_usage(self.device_map, "Final")
            print("===============================\n")
        else:
            self.load_vision_tower()
            self.load_llm()

    def load_vision_tower(self):
        config = RQVAESIGLIPTransformerConfig.from_pretrained(self.config.vision_tower_type)
        config.device_map = self.device_map
        self.vision_tower = RQVAESIGLIPTransformerVisionTower(self.config.vision_tower_type, config).to(self.device_map["vision_tower"])

    def load_llm(self):
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.config.llm_type, trust_remote_code=True)
        self.llm = AutoModelForCausalLM.from_pretrained(
            self.config.llm_type,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=self.device_map["llm"]
        )

    def forward(self, prompt, vision, max_length=512, **_):
        inputs = self.text_tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.llm.device) for k, v in inputs.items()}
        vision_processed = self.vision_tower.image_processor(vision, return_tensors="pt")["pixel_values"].to(self.vision_tower.device).to(torch.bfloat16)
        img_tokens, img_features = self.vision_tower.rqvaesiglip.encode_image(vision_processed)
        out_vision = self.vision_tower.rqvaesiglip.decode(img_features).to(torch.float32).add_(1).mul_(127.5).clamp_(0, 255).chunk(2)[0]
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_length=max_length,
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

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs) -> "PreTrainedModel":
        config = kwargs.pop("config", None)
        if config is None:
            config = VLACConfig.from_pretrained(pretrained_model_name_or_path)
        return super().from_pretrained(pretrained_model_name_or_path, config=config, *args, **kwargs)
