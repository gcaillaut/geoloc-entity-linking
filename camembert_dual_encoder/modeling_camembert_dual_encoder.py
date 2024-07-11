from dataclasses import dataclass
from typing import Dict, Optional
from transformers import PreTrainedModel, CamembertModel
from transformers.utils import ModelOutput
import torch
from torch import nn
import torch.nn.functional as F

from camembert_dual_encoder.configuration_camembert_dual_encoder import CamembertDualEncoderConfig

class TokenClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels, bias=True)

    def forward(self, inputs):
        return self.classifier(self.dropout(inputs))

class EntityHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.output_size, bias=True)
        
    def forward(self, inputs):
        return self.dense(inputs)


class CamembertMentionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inner_model = CamembertModel(config)
        self.token_classifier = TokenClassificationHead(config)
        self.entity_head = EntityHead(config)
        self.config = config

    def forward(self, input_ids, attention_mask):
        outputs = self.inner_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        last_hidden_state = outputs.last_hidden_state
        ner_logits = self.token_classifier(last_hidden_state)
        mention_logits = self.entity_head(last_hidden_state)
        return ner_logits, mention_logits
    
    def load_pretrained_weights_for_language_models(self, model_name):
        self.inner_model = CamembertModel.from_pretrained(model_name, config=self.config)

class CamembertEntityEncoder(nn.Module):
    '''Compute embedding from an entity description'''

    def __init__(self, config):
        super().__init__()
        self.inner_model = CamembertModel(config)
        self.entity_head = EntityHead(config)
        self.config = config

    def forward(self, input_ids, attention_mask):
        if len(input_ids) == 0 and len(attention_mask) == 0:
            return torch.zeros((0, self.config.output_size), device=input_ids.device)
        outputs = self.inner_model(input_ids, attention_mask, return_dict=True)
        return self.entity_head(outputs.pooler_output)
    
    def load_pretrained_weights_for_language_models(self, model_name):
        self.inner_model = CamembertModel.from_pretrained(model_name, config=self.config)

@dataclass
class CamembertDualEncoderOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mention_logits: torch.FloatTensor = None
    ner_logits: torch.FloatTensor = None
    entity_logits: torch.FloatTensor = None
    losses: Dict[str, torch.FloatTensor] = None


class CamembertDualEncoderModel(PreTrainedModel):
    config_class = CamembertDualEncoderConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.mention_encoder = CamembertMentionEncoder(config)
        self.entity_encoder = CamembertEntityEncoder(config)

        self.ner_loss_func = nn.CrossEntropyLoss()
        self.el_loss_func = nn.CrossEntropyLoss()
        self.hn_loss_func = nn.BCEWithLogitsLoss()

    def load_pretrained_weights_for_language_models(self, model_name):
        self.mention_encoder.load_pretrained_weights_for_language_models(model_name)
        self.entity_encoder.load_pretrained_weights_for_language_models(model_name)

    def forward(self, mention_input_ids, mention_attention_mask, entity_input_ids=None, entity_attention_mask=None, ner_labels=None, entity_mappings=None, hard_negatives=None, return_dict=True):
        losses = {}
        loss = None

        ner_logits, mention_logits = self.mention_encoder(
            mention_input_ids,
            mention_attention_mask
        )

        entity_logits = None
        if entity_input_ids is not None and entity_attention_mask is not None and entity_mappings is not None:
            entity_logits = self.entity_encoder(entity_input_ids, entity_attention_mask)
            
            entity_rows = [x[0] for x in entity_mappings]
            entity_cols = [x[1] for x in entity_mappings]
            
            entity_mention_logits = mention_logits[entity_rows, entity_cols, :]

            n_entities = entity_mention_logits.size(0)
            if n_entities == 0:
                el_loss = torch.tensor([0], device=entity_mention_logits.device)
            else:
                # in-bach cross-entropy (https://arxiv.org/pdf/1811.08008.pdf)
                if self.config.mention_entity_similarity == 'dot':
                    matrix_sims = torch.mm(
                        entity_mention_logits,
                        torch.t(entity_logits)
                    )
                elif self.config.mention_entity_similarity == 'cos':
                    matrix_sims = torch.mm(
                        F.normalize(entity_mention_logits),
                        torch.t(F.normalize(entity_logits))
                    )
                el_target = torch.arange(n_entities, device=matrix_sims.device)
                el_loss = self.el_loss_func(matrix_sims, el_target)
            losses["el"] = el_loss

            if n_entities > 0 and hard_negatives is not None and len(hard_negatives) > 0:
                hn_sims = []
                hn_labels = []
                device = entity_mention_logits.device
                
                for x, hn in zip(entity_mention_logits, hard_negatives):
                    hn_input_ids = hn["input_ids"].to(device)
                    hn_attention_mask = hn["attention_mask"].to(device)
                    hn_logits = self.entity_encoder(hn_input_ids, hn_attention_mask)
                    del hn_input_ids, hn_attention_mask
                    
                    hn_sims.append(torch.matmul(hn_logits, x))
                    hn_labels.append(hn["labels"])
                    
                hn_sims = torch.hstack(hn_sims)
                hn_labels = torch.hstack(hn_labels).to(device)
                hn_loss = self.hn_loss_func(hn_sims, hn_labels)
                losses["hn"] = hn_loss
                del hn_sims, hn_labels
            
        if ner_labels is not None:
            ner_loss = self.ner_loss_func(
                ner_logits.view(-1, ner_logits.size(-1)), 
                ner_labels.view(-1)
            )
            losses['ner'] = ner_loss

        if len(losses) > 0:
            loss = sum(losses.values())
            losses['loss'] = loss
        
        output = CamembertDualEncoderOutput(loss, mention_logits, ner_logits, entity_logits, losses)
        if return_dict:
            return output
        else:
            return output.to_tuple()
