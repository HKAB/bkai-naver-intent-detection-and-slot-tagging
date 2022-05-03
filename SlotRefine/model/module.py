import torch
import torch.nn as nn
import numpy as np


class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(
                self, 
                input_dim,
                num_slot_labels,
                dropout_rate=0.0,
    ):
        super(SlotClassifier, self).__init__()
        # origin slotrefine
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim*2, num_slot_labels)

    def forward(self, x):

        # concat cls to all other token
        # x: (batch x max_seq_length x hidden_size)
        cls_addition = torch.repeat_interleave(x[:, 0, :].unsqueeze(dim=1), x.shape[1], dim=1)
        x = torch.cat([x, cls_addition], dim=2)
        x = self.dropout(x)
        return self.linear(x)