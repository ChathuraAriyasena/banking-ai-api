"""
MultiTaskModel definition.
This MUST match the architecture used in ML_Banking_Implementation.ipynb
and LLM_Banking_Implementation.ipynb exactly, otherwise the saved
state_dict will fail to load.
"""

import torch.nn as nn
from transformers import AutoModel


class MultiTaskModel(nn.Module):
    def __init__(self, model_name, num_labels_dict, dropout=0.2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        h = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.intent_head        = nn.Linear(h, num_labels_dict["intent"])
        self.issue_type_head    = nn.Linear(h, num_labels_dict["issue_type"])
        self.product_head       = nn.Linear(h, num_labels_dict["product"])
        self.urgency_head       = nn.Linear(h, num_labels_dict["urgency"])
        self.sentiment_head     = nn.Linear(h, num_labels_dict["sentiment"])
        self.routing_queue_head = nn.Linear(h, num_labels_dict["routing_queue"])

    def forward(self, input_ids, attention_mask):
        x = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        x = self.dropout(x)
        return {
            "intent":        self.intent_head(x),
            "issue_type":    self.issue_type_head(x),
            "product":       self.product_head(x),
            "urgency":       self.urgency_head(x),
            "sentiment":     self.sentiment_head(x),
            "routing_queue": self.routing_queue_head(x),
        }
