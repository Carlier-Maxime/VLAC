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
            self.llm.vocab_size = self.config.vocab_size
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
        im_ids1 = im_ids[:, 1].add(decals).unsqueeze(1).add(torch.arange(im_seq_len, device=im_ids.device)[None]).flatten()
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
            im_ignore = labels[im_ids[:, 0], im_ids[:, 1]].eq(IGNORE_INDEX)
            im_ignore_mask = im_mask.clone()
            im_ignore_mask[im_mask] = im_ignore[:, None].expand(-1, 256).flatten()
            new_labels[im_ignore_mask] = IGNORE_INDEX
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
            image_ids=None,
    ):
        if image_ids is None:
            image_ids = []
        while len(image_ids) < attention_mask.shape[0]:
            image_ids.append([])
        if input_ids is not None:
            if inputs_embeds is not None:
                raise UnsupportedOperation("You cannot specify both input_ids and inputs_embeds at the same time")
            mask = input_ids[:, -1].eq(self.IM_END_TOKEN_INDEX)
            for i in range(len(mask)):
                if not mask[i]: continue
                if len(image_ids[i]) == 0:
                    input_ids[i, -1] = 0
                    continue
                embeds = torch.stack(image_ids[i])
                _, code = self.vlac.vision_tower.rqtransformer.generate(embeds, self.vlac.vision_tower.rqvaesiglip)
                raise RuntimeError("TODO: Make Image")
            _, position_ids, attention_mask, inputs_embeds, labels = self.prepare_for_multimodal(input_ids, position_ids, attention_mask, labels, None)
            inputs_embeds = inputs_embeds

        outputs = self.llm.model(
            input_ids=None,
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
        B, seq_len, _ = hidden_states.shape

        vocab_txt = self.config.vocab_size
        logits = self.lm_head(hidden_states).float()
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                torch.where(labels[:, :, 0].lt(vocab_txt), labels[:, :, 0], IGNORE_INDEX).view(-1),
                ignore_index=IGNORE_INDEX
            )

        if labels is not None:
            mask = labels.ge(vocab_txt).all(dim=2)
            len_of_im = mask.sum(dim=1).unique()
            assert len_of_im.shape[0], "a different number of images in each sample in batch is not supported yet"
            embeds = self.decoder(hidden_states[mask]).to(torch.float).to(self.vlac.vision_tower.device)
            embeds = embeds.view(B, len_of_im[0], embeds.shape[-1])
            _, _, vision_logits = self.vlac.vision_tower.rqtransformer.generate(embeds, self.vlac.vision_tower.rqvaesiglip, return_logits=True)
            loss += F.cross_entropy(
                vision_logits.reshape(-1, vision_logits.shape[-1]),
                labels[mask].sub(vocab_txt).view(-1),
                ignore_index=IGNORE_INDEX
            )
        elif input_ids is not None:
            mask = input_ids[:, -1].eq(self.IM_START_TOKEN_INDEX) | torch.tensor([len(ids) > 0 for ids in image_ids], dtype=torch.bool, device=input_ids.device)
            embeds = self.decoder(hidden_states[:, -1:]).to(torch.float).to(self.vlac.vision_tower.device)
            for i in range(len(mask)):
                if mask[i]: image_ids[i].append(embeds[i])

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
