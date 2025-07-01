import warnings
from io import UnsupportedOperation
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, AutoModelForCausalLM, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from vlac.model.multimodal_projector.base_projector import MultimodalProjectorConfig, MultimodalProjector
from vlac.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class VLACVLMConfig(PretrainedConfig):
    def __init__(self, vlac=None, **kwargs):
        super().__init__(**kwargs)
        self.vlac = vlac
        self.is_encoder_decoder = False
        self.is_decoder = True
        self.add_cross_attention = False
        self.tie_word_embeddings = True
        if self.vlac is None:
            return
        self.vocab_size = self.vlac.text_embeds.num_embeddings
        self.hidden_size = vlac.llm.config.hidden_size


class VLACForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = VLACVLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        if self.config.vocab_size != self.lm_head.out_features:
            print(f"WARNING : lm_head has been recreate because out_features ({self.lm_head.out_features}) mismatch with vocab_size ({self.config.vocab_size}) !")
            self.llm.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False).to(self.lm_head.weight.device).to(self.lm_head.weight.dtype)
        self.config.is_decoder = True
        self.config.is_encoder_decoder = False
        self.IM_START_TOKEN_INDEX = self.vlac.text_tokenizer.convert_tokens_to_ids(DEFAULT_IM_START_TOKEN)
        self.IM_END_TOKEN_INDEX = self.vlac.text_tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        mm_conf = MultimodalProjectorConfig(self.vlac.config.project_multimodal_type)
        self.encoder = MultimodalProjector(mm_conf, PretrainedConfig(mm_hidden_size=self.vlac.config.hidden_size, hidden_size=self.config.hidden_size)).to(torch.bfloat16)
        self.decoder = MultimodalProjector(mm_conf, PretrainedConfig(mm_hidden_size=self.config.hidden_size, hidden_size=self.vlac.config.hidden_size)).to(torch.bfloat16)

    @property
    def vlac(self):
        return self.config.vlac

    @property
    def llm(self):
        return self.vlac.llm

    @property
    def lm_head(self):
        return self.vlac.llm.lm_head

    def prepare_for_multimodal(
            self,
            input_ids,
            position_ids,
            attention_mask,
            labels,
            images,
            encode: bool = True
    ):
        if images is None or input_ids.shape[1] == 1:
            input_embeds = self.vlac.text_embeds(input_ids).to(self.encoder.dtype)
            return (
                None,
                position_ids,
                attention_mask,
                self.encoder(input_embeds) if encode else input_embeds,
                labels,
            )

        if type(images) is list:
            images = torch.cat(images, dim=0)
        elif images.ndim == 5:
            images = images.flatten(0, 1)

        image_features, tokens = self.vlac.encode_images(images)
        image_features = self.vlac.project_img_features(image_features).to(self.llm.device)
        if encode: image_features = self.encoder(image_features)
        tokens = tokens.to(self.llm.device).flatten(1, -2)

        input_ids_copy = input_ids.clone()
        img_token_mask = input_ids_copy == IMAGE_TOKEN_INDEX
        input_ids_copy[img_token_mask] = 0
        input_embeds = self.vlac.text_embeds(input_ids_copy).to(image_features.dtype)
        if encode: input_embeds = self.encoder(input_embeds)

        B, N, D = input_embeds.shape
        im_seq_len = image_features.shape[1]

        wIMs = torch.where(input_ids_copy.eq(self.IM_START_TOKEN_INDEX))
        wIMe = torch.where(input_ids_copy.eq(self.IM_END_TOKEN_INDEX))
        assert len(wIMs[0]) == len(image_features) and wIMs[0].shape == wIMe[0].shape and wIMs[0].eq(wIMe[0]).all().item() and wIMs[1].add(1).eq(wIMe[1]).all().item(), f'Bad synthax of IM_TOKEN, info : {len(image_features)} & {wIMs} & {wIMe}'
        im_per_batch = wIMs[0].unique(return_counts=True)[1]
        del wIMs, wIMe
        M = im_per_batch.max().item() * im_seq_len
        W = N + M

        if getattr(self.llm.config, "tokenizer_padding_side", "right") == "left":
            raise UnsupportedOperation("Left padding is actually not supported")

        im_ids = torch.nonzero(input_ids_copy.eq(self.IM_END_TOKEN_INDEX))
        counts = im_per_batch.mul(im_seq_len).unsqueeze(-1)
        decals = torch.arange(0, counts.max(), im_seq_len, device=counts.device)[None].expand(B, -1)
        decals = decals[decals < counts]
        im_ids0 = im_ids[:, 0].unsqueeze(1).expand(-1, im_seq_len).flatten()
        im_ids1 = im_ids[:, 1].add_(decals).unsqueeze(1).add(torch.arange(im_seq_len, device=im_ids.device)[None]).flatten()
        im_mask = torch.zeros((B, W), dtype=torch.bool, device=decals.device)
        im_mask[[im_ids0, im_ids1]] = True

        counts = im_mask.sum(dim=1).add_(N).unsqueeze(-1)
        ids = torch.arange(W, device=counts.device)[None].expand(B, -1)
        new_attention_mask = ids.lt(counts)
        if position_ids is not None:
            position_ids = torch.zeros((B, W), dtype=position_ids.dtype, device=position_ids.device)
            position_ids[new_attention_mask] = ids[new_attention_mask]
        text_mask = new_attention_mask & ~im_mask
        if attention_mask is not None:
            new_attention_mask = new_attention_mask.to(dtype=attention_mask.dtype)
            new_attention_mask[text_mask] = attention_mask.flatten()
        else:
            new_attention_mask = None

        new_input_embeds = input_embeds.new_empty((B, W, D))
        new_input_embeds[im_mask] = image_features.flatten(0, 1)
        new_input_embeds[text_mask] = input_embeds.flatten(0, 1)
        if labels is not None:
            new_labels = labels.new_empty((B, W, tokens.shape[-1]))
            new_labels[im_mask] = tokens.flatten(0, 1)
            new_labels[text_mask] = labels.flatten().unsqueeze(-1).expand(-1, tokens.shape[-1])
        else:
            new_labels = None

        tokenizer_model_max_length = getattr(self.llm.config, "tokenizer_model_max_length", None)
        if tokenizer_model_max_length is not None and W > tokenizer_model_max_length:
            warnings.warn("Inputs truncated!")
            new_input_embeds = new_input_embeds[:, :tokenizer_model_max_length]
            if labels is not None: new_labels = new_labels[:, :tokenizer_model_max_length]
            if attention_mask is not None: new_attention_mask = new_attention_mask[:, :tokenizer_model_max_length]
            if position_ids is not None: position_ids = position_ids[:, :tokenizer_model_max_length]

        return (
            None,
            position_ids,
            new_attention_mask,
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
            image_ids=None
    ):
        if image_ids is None:
            image_ids = []
        gen_image = False
        if input_ids is not None:
            if inputs_embeds is not None:
                raise UnsupportedOperation("You cannot specify both input_ids and inputs_embeds at the same time")
            gen_image = input_ids[:, -1].eq(self.IM_START_TOKEN_INDEX).any()
            if input_ids[:, -1].eq(self.IM_END_TOKEN_INDEX).any():
                raise RuntimeError("TODO : make image")
            input_ids, position_ids, attention_mask, inputs_embeds, labels = self.prepare_for_multimodal(input_ids, position_ids, attention_mask, labels, None)
            inputs_embeds = inputs_embeds

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
            seqlens_in_batch=seqlens_in_batch
        )
        hidden_states = outputs.last_hidden_state

        if gen_image:
            self.vlac.vision_tower.rqtransformer.eval()
            image_hidden_state, code = self.vlac.vision_tower.rqtransformer.generate(self.decoder(hidden_states[:, -1]).to(torch.float).to(self.vlac.vision_tower.device), self.vlac.vision_tower.rqvaesiglip)
            image_hidden_state = self.vlac.mm_projector(image_hidden_state)
            hidden_states[:, -1] = image_hidden_state
            image_ids.append(code)

        logits = self.lm_head(hidden_states).float()
        loss = None if labels is None else F.cross_entropy(logits, labels)

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
        self.vlac.llm.lm_head = new_embeddings


AutoModelForCausalLM.register(VLACVLMConfig, VLACForCausalLM)
