import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer


class VLACTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        text_tokens = inputs["text_tokens"]
        vision = inputs["vision"].to(torch.bfloat16)
        model = model.module if hasattr(model, "module") else model
        out_vision, img_hidden_states, img_embeds, img_features, img_tokens = model.encode_decode_images(vision)
        mse = nn.MSELoss()
        reconstruction_loss = mse(vision, out_vision) + mse(img_hidden_states, img_features)
        text_embeds = model.llm.get_input_embeddings()(text_tokens.to(model.llm.device))
        loss_i, loss_t = self.info_nce_loss(text_tokens, img_tokens, text_embeds, img_embeds)
        contrastive_loss = (loss_i + loss_t) / 2
        outputs = {
            "text_tokens": text_tokens,
            "out_vision": out_vision,
        }
        loss = (reconstruction_loss + contrastive_loss)
        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def info_nce_loss(text_tokens, image_tokens, text_embeds, img_embeds):
        N, H, W, C = image_tokens.shape
        I_e = F.normalize(torch.einsum('bijk,bijkl->bl', image_tokens.to(torch.bfloat16), img_embeds.view(N, H, W, C, -1)).repeat(1, C), dim=1)
        T_e = F.normalize(torch.einsum('bi,bij->bj', text_tokens.to(torch.bfloat16), text_embeds), dim=1)
        one = torch.tensor(1, dtype=torch.bfloat16, device=I_e.device)
        logits = torch.matmul(I_e, T_e.T) * torch.exp(one)
        loss_i = F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
        loss_t = F.cross_entropy(logits.T, torch.arange(len(logits), device=logits.device))
        return loss_i, loss_t


class EncodeDecodeTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mse = nn.MSELoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        embeds = inputs["embeds"]
        return self.mse(embeds, model(embeds))


def getTrainerCls(name: str) -> type[Trainer]:
    name = name.lower()
    if "vlac" in name:
        return VLACTrainer
    elif "encodedecode" in name:
        return EncodeDecodeTrainer
    else:
        raise ValueError(f"Unknown trainer name: {name}")
