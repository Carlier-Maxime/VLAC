import torch
from torch import nn
from transformers import AutoConfig, PretrainedConfig

from vlac import MultimodalProjector, MultimodalProjectorConfig
from vlac.model.utils import load_weights_of_keys_start_with


class VLACEncodeDecode(nn.Module):
    def __init__(self, cfg, **kwargs):
        super().__init__(**kwargs)
        llm_cfg = AutoConfig.from_pretrained(cfg.llm_type)
        enc = "vlm.encoder"
        dec = "vlm.decoder"
        states = load_weights_of_keys_start_with("./vlac_base", [enc, dec])
        mm_conf = MultimodalProjectorConfig(cfg.project_multimodal_type)
        self.encoder = MultimodalProjector(mm_conf, PretrainedConfig(mm_hidden_size=cfg.hidden_size, hidden_size=llm_cfg.hidden_size)).to(torch.bfloat16)
        self.decoder = MultimodalProjector(mm_conf, PretrainedConfig(mm_hidden_size=cfg.hidden_size, hidden_size=llm_cfg.hidden_size)).to(torch.bfloat16)
        self.encoder.load_state_dict(states[enc])
        self.decoder.load_state_dict(states[dec])

    def forward(self, embeds):
        return self.decoder(self.encoder(embeds))
