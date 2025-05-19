import warnings

from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

from vlac.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from vlac.utils.gpu_memory_utils import print_gpu_memory_usage, track_gpu_memory_usage
from vlac.model.multimodal_encoder.rqvaesigliptransformer import RQVAESIGLIPTransformerConfig
from vlac.model.multimodal_encoder.rqvaesigliptransformer_encoder import RQVAESIGLIPTransformerVisionTower
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

    def forward(self, prompt, vision, max_length: int = 512, generation_nums: int = 1, **_):
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
        #  _, _, attention_mask, _, input_embeds, _ = self.prepare_embeds_for_multimodal(input_ids, None, attention_mask, None, None, vision_preprocessed)

        img_tokens, img_features = self.vision_tower.rqvaesiglip.encode_image(vision_preprocessed)
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

    def prepare_embeds_for_multimodal(  # TODO Optimize this func for more parallelism with torch
            self,
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
    ):
        if self.vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                    past_key_values is not None
                    and self.vision_tower is not None
                    and images is not None
                    and input_ids.shape[1] == 1
            ):
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.ones(
                            (
                                attention_mask.shape[0],
                                target_shape - attention_mask.shape[1],
                            ),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        ),
                    ),
                    dim=1,
                )
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )

        if type(images) is list:
            images = torch.cat(images, dim=0)
        elif images.ndim == 5:
            images = images.flatten(0, 1)

        input_image_ids = input_ids[input_ids == IMAGE_TOKEN_INDEX]
        image_features, tokens = self.encode_images(images, input_image_ids)

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids_copy = input_ids.clone()
        input_ids_copy[input_ids_copy == IMAGE_TOKEN_INDEX] = 0
        input_embeds = self.llm.model.embed_tokens(input_ids_copy)

        input_ids = [
            cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        input_embeds_1 = [
            cur_input_embeds[cur_attention_mask]
            for cur_input_embeds, cur_attention_mask in zip(input_embeds, attention_mask)
        ]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_input_ids = input_ids[batch_idx]
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[0]
                cur_input_embeds_1 = input_embeds_1[batch_idx]
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx].unsqueeze(1).expand(-1, tokens.shape[-1]))
                continue

            cur_input_embeds = input_embeds_1[batch_idx]
            image_token_indices = (
                    [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            )
            cur_labels = labels[batch_idx]

            cur_input_ids_noim = []
            cur_labels_noim = []
            cur_input_embeds_no_im = []

            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1: image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1: image_token_indices[i + 1]])
                cur_input_embeds_no_im.append(cur_input_embeds[image_token_indices[i] + 1: image_token_indices[i + 1]])

            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i].unsqueeze(1).expand(-1, tokens.shape[-1]))
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_tokens = tokens[cur_image_idx]
                    cur_new_input_embeds.append(cur_image_features)
                    if self.config.mm_use_vi_start_end:
                        if (cur_input_ids[-3] == -200 and self.llm.vocab_size - 4 in cur_new_labels[-1]) \
                                or all(x == -200 for x in cur_input_ids[-10:-3]):
                            cur_new_labels.append(cur_tokens)
                        else:
                            cur_new_labels.append(
                                torch.full(
                                    (cur_image_features.shape[0], tokens.shape[-1]),
                                    IGNORE_INDEX,
                                    device=cur_labels.device,
                                    dtype=cur_labels.dtype,
                                )
                            )
                    else:
                        if (cur_input_ids[-3] == -200 and self.llm.vocab_size - 2 in cur_new_labels[-1]):
                            cur_new_labels.append(cur_tokens)
                        else:
                            cur_new_labels.append(
                                torch.full(
                                    (cur_image_features.shape[0], tokens.shape[-1]),
                                    IGNORE_INDEX,
                                    device=cur_labels.device,
                                    dtype=cur_labels.dtype,
                                )
                            )
                    cur_image_idx += 1

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        tokenizer_model_max_length = getattr(self.llm.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None:
            if any(len(x) > tokenizer_model_max_length for x in new_input_embeds):
                warnings.warn("Inputs truncated!")
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len, tokens.shape[-1]),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.llm.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args, **kwargs) -> "PreTrainedModel":
        config = kwargs.pop("config", None)
        if config is None:
            config = VLACConfig.from_pretrained(pretrained_model_name_or_path)
        return super().from_pretrained(pretrained_model_name_or_path, config=config, *args, **kwargs)
