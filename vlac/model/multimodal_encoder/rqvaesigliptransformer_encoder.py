import torch
import torch.nn as nn

from transformers import CLIPImageProcessor
from .rqvaesigliptransformer import RQVAESIGLIPTransformerConfig, RQVAESIGLIPTransformer


class RQVAESIGLIPTransformerVisionTower(nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()
        self.config = RQVAESIGLIPTransformerConfig.from_pretrained(model_name_or_path)
        self.vision_tower = RQVAESIGLIPTransformer.from_pretrained(model_name_or_path, config=self.config)
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
        output = self.vision_tower.rqvaesiglip.encode_image(images)
        image_features, tokens = output[-1], output[-2]

        bs, patch_size, _, dim = image_features.shape
        image_features = torch.reshape(image_features, [bs, patch_size ** 2, dim])
        tokens = torch.add(torch.reshape(tokens, [bs, patch_size ** 2, -1]), text_vocab_size)

        return image_features, tokens
