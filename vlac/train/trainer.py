import os
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import Trainer, PretrainedConfig
from transformers.modeling_utils import unwrap_model


class VLACTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        text_tokens = inputs["text_tokens"]
        vision = inputs["vision"]
        model = model.module if hasattr(model, "module") else model
        out_vision, img_features, img_tokens = model.encode_decode_images(vision)
        reconstruction_loss = torch.nn.MSELoss()(vision, out_vision)
        img_features = model.mm_projector(img_features)
        text_embeds = model.llm.get_input_embeddings()(text_tokens.to(model.llm.device))
        loss_i, loss_t = self.info_nce_loss(text_tokens, img_tokens, text_embeds, img_features)
        contrastive_loss = (loss_i + loss_t) / 2
        outputs = {
            "text_tokens": text_tokens,
            "out_vision": out_vision,
        }
        loss = (reconstruction_loss + contrastive_loss)
        return (loss, outputs) if return_outputs else loss

    @staticmethod
    def info_nce_loss(text_tokens, image_tokens, text_embeds, img_features):
        N, H, W, C = image_tokens.shape
        I_e = F.normalize(torch.einsum('bijk,bijkl->bl', image_tokens.to(torch.bfloat16), img_features.view(N, H, W, C, -1)).repeat(1, C), dim=1)
        T_e = F.normalize(torch.einsum('bi,bij->bj', text_tokens.to(torch.bfloat16), text_embeds), dim=1)
        one = torch.tensor(1, dtype=torch.bfloat16, device=I_e.device)
        logits = torch.matmul(I_e, T_e.T) * torch.exp(one)
        loss_i = F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
        loss_t = F.cross_entropy(logits.T, torch.arange(len(logits), device=logits.device))
        return loss_i, loss_t

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        model_to_save = unwrap_model(self.model)

        state_dict_path = os.path.join(output_dir, "pytorch_model.bin")
        torch.save(model_to_save.state_dict(), state_dict_path)

        if hasattr(model_to_save, "config") and isinstance(model_to_save.config, PretrainedConfig):
            model_to_save.config.save_pretrained(output_dir)

        if hasattr(model_to_save, "text_tokenizer") and model_to_save.text_tokenizer is not None:
            model_to_save.text_tokenizer.save_pretrained(output_dir)

        self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

        print(f"[VLACTrainer] Save terminated in {output_dir}")

