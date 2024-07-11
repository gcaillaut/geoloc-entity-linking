from transformers import CamembertConfig


class CamembertDualEncoderConfig(CamembertConfig):
    model_type = 'camembert-dual-encoder'

    def __init__(self, encoder_pretrained_model='camembert-base', mention_entity_similarity='cos', output_size=None, **kwargs):
        super().__init__(**kwargs)
        self.encoder_pretrained_model = encoder_pretrained_model
        self.output_size = output_size or self.hidden_size
        self.mention_entity_similarity = mention_entity_similarity
