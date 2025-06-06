import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel


class TextEmbeddingConfig(PretrainedConfig):
    model_type = "text_embedding"

    def __init__(self, num_embeddings: int = 32004, embedding_dim: int = 4096, padding_idx: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx


class TextEmbedding(PreTrainedModel):
    config_class = TextEmbeddingConfig

    def __init__(self, config: TextEmbeddingConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        self.embeds = nn.Embedding(config.num_embeddings, config.embedding_dim, config.padding_idx)

    def forward(self, input_ids, *args, **kwargs):
        return self.embeds(input_ids)
