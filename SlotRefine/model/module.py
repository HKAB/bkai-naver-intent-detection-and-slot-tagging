import torch
import torch.nn as nn


class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim*2, num_slot_labels)

    def forward(self, x):

        # x: (batch x max_seq_length x hidden_size)
        new_x = torch.zeros_like(x)
        new_x[:, 1:, :] +=  x[:, :-1, :]
        new_x = torch.cat([x, new_x], dim=2)

        new_x = self.dropout(new_x)
        return self.linear(new_x)
