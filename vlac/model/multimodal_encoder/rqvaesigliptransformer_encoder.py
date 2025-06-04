import torch
import torch.nn as nn

from transformers import CLIPImageProcessor
from .rqvaesigliptransformer import RQVAESIGLIPTransformerConfig, RQVAESIGLIPTransformer


class RQVAESIGLIPTransformerVisionTower(nn.Module):
    def __init__(self, model_name_or_path=None, config=None):
        super().__init__()
        self.config = RQVAESIGLIPTransformerConfig.from_pretrained(model_name_or_path) if config is None else config
        self.base_model = RQVAESIGLIPTransformer.from_pretrained(model_name_or_path, config=self.config) if model_name_or_path is not None else RQVAESIGLIPTransformer(config=self.config)
        self.rqvaesiglip = self.base_model.rqvaesiglip.to(torch.bfloat16)
        self.rqtransformer = self.base_model.rqtransformer
        self.is_loaded = True

        def select_value_by_hidden_size(value_for_1152, value_for_1024):
            return value_for_1152 if self.config.hidden_size == 1152 else value_for_1024 if self.config.hidden_size == 1024 else NotImplementedError()

        img_size = select_value_by_hidden_size(384, 256)
        self.image_processor = CLIPImageProcessor(
            size={"height": img_size, "width": img_size},
            crop_size={"height": img_size, "width": img_size},
            image_mean=[0.5, 0.5, 0.5],
            image_std=[0.5, 0.5, 0.5]
        )
        self.image_tokens = select_value_by_hidden_size(729, 256)

    def forward(self, images, text_vocab_size):
        tokens, image_features,  = self.rqvaesiglip.encode_image(images)
        tokens.add_(text_vocab_size)
        return image_features, tokens

    def decode_tokens(self, img_tokens, text_vocab_size):
        img_tokens.sub_(text_vocab_size)
        img_embeds_vt = self.rqtransformer.embed_with_model_aux(img_tokens, self.rqvaesiglip)
        img_embeds_vt = torch.cumsum(img_embeds_vt, dim=-2)[:, :, :, -1, :]
        return self.decode_features(img_embeds_vt)

    def decode_features(self, img_features):
        return self.rqvaesiglip.decode(img_features)

    @property
    def device(self):
        return next(self.base_model.parameters()).device
