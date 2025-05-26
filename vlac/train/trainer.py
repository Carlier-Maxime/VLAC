import torch
import torch.nn.functional as F
from transformers import Trainer


class VLACTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        prompt = inputs["prompt"]
        vision = inputs["vision"]
        text_tokens = model.text_tokenizer(prompt,  return_tensors="pt")
        input_ids = text_tokens["input_ids"].to("cuda")
        in_vision = model.vision_tower.image_processor(vision, return_tensors="pt")["pixel_values"].to(model.vision_tower.device).to(torch.bfloat16)
        img_tokens, img_features = model.vision_tower.rqvaesiglip.encode_image(in_vision)
        out_vision = model.vision_tower.rqvaesiglip.decode(img_features).to(torch.float32).add_(1).mul_(127.5).clamp_(0, 255).chunk(2)[0]
        reconstruction_loss = torch.nn.MSELoss()(in_vision, out_vision)
        img_features = model.mm_projector(img_features)
        text_embeds = model.llm.get_input_embeddings()(input_ids)
        N, H, W, C = img_tokens.shape
        multiplier = 16384
        I_e = F.normalize(torch.einsum('bijk,bijkl->bl', img_tokens.to(torch.bfloat16).mul(multiplier), img_features.view(N, H, W, C, -1).mul(multiplier)).repeat(1, C), dim=1)
        T_e = F.normalize(torch.einsum('bi,bij->bj', input_ids.to(torch.bfloat16).mul(multiplier), text_embeds.mul(multiplier)), dim=1)
        one = torch.tensor(1, dtype=torch.bfloat16, device=I_e.device)
        logits = torch.matmul(I_e, T_e.T) * torch.exp(one)
        loss_i = F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))
        loss_t = F.cross_entropy(logits.T, torch.arange(len(logits), device=logits.device))
        contrastive_loss = (loss_i + loss_t) / 2
        outputs = {
            "text_tokens": text_tokens,
            "out_vision": out_vision,
        }
        loss = (reconstruction_loss + contrastive_loss).to(torch.float)
        return (loss, outputs) if return_outputs else loss
