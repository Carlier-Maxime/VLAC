import warnings
from typing import Optional, List

import torch
from transformers import PreTrainedModel, PretrainedConfig, AutoModelForCausalLM, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from vlac.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX


class VLACVLMConfig(PretrainedConfig):
    def __init__(self, vlac = None, **kwargs):
        super().__init__(**kwargs)
        self.vlac = vlac
        self.is_encoder_decoder = False
        self.is_decoder = True
        self.add_cross_attention = False
        self.tie_word_embeddings = True
        if self.vlac is not None:
            self.vocab_size = vlac.llm.config.vocab_size
            self.hidden_size = vlac.llm.config.hidden_size


class VLACForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = VLACVLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        object.__setattr__(self, 'vlac', self.config.vlac)
        self.lm_head = self.vlac.llm.lm_head
        self.llm = self.vlac.llm
        self.config.is_decoder = True
        self.config.is_encoder_decoder = False

    def prepare_embeds_for_multimodal(  # TODO Optimize this func for more parallelism with torch
            self,
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
    ):
        if self.vlac.vision_tower is None or images is None or input_ids.shape[1] == 1:
            if (
                    past_key_values is not None
                    and self.vlac.vision_tower is not None
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

        image_features, tokens = self.vlac.encode_images(images)
        image_features = image_features.to(self.llm.device).flatten(1, -2)
        tokens = tokens.to(self.llm.device)

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
                    if self.vlac.config.mm_use_vi_start_end:
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

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            seqlens_in_batch: Optional[torch.LongTensor] = None,
            vision_tower=None,
            mm_projector=None,
            image_ids=None,
            cfg=None,
    ):
        if image_ids is None:
            image_ids = []
        outputs = self.llm.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            seqlens_in_batch=seqlens_in_batch,
            vision_tower=vision_tower,
            mm_projector=mm_projector,
            image_ids=image_ids,
        )
        hidden_states = outputs.last_hidden_state
        self.vlac.vision_tower.rqtransformer.eval()
        hidden_states = hidden_states.to(torch.float)
        image_hidden_state, code = self.vlac.vision_tower.rqtransformer.generate(hidden_states.to(self.vlac.vision_tower.device), self.vlac.vision_tower.rqvaesiglip, cfg)
        image_hidden_state = self.vlac.mm_projector(image_hidden_state)
        hidden_states = image_hidden_state
        # image_ids.append(code)
        loss = None
        logits = self.lm_head(hidden_states).float()
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings


AutoModelForCausalLM.register(VLACVLMConfig, VLACForCausalLM)
