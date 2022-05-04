import torch
import torch.nn as nn
from torchcrf import CRF
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)

class LSTMDecoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout = 0.2, use_crf = True, max_len = None, use_linear = True) -> None:
        super().__init__()

        self.lstm = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, batch_first = True,\
             bidirectional = True, dropout = dropout, num_layers = 1)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim * 2, output_dim) if use_linear else None# bidirectional
        self.use_crf = use_crf
        self.max_len = max_len
        if use_crf:
            self.crf = CRF(output_dim, batch_first = True)
        else:
            self.nllloss = nn.NLLLoss(reduction = 'sum')

    def forward(self, inputs, len_list):

        packed_inputs = pack_padded_sequence(inputs, len_list, batch_first = True)
        seq_out, (h_state, c_state) = self.lstm(packed_inputs)
        seq_out, _ = pad_packed_sequence(seq_out, batch_first = True, total_length = self.max_len)

        if self.linear is not None:
            seq_out = self.linear(seq_out)
        return seq_out

    def get_loss(self, logits, labels, mask):
        if self.use_crf:
            loss = - self.crf(logits, labels, mask.byte(), reduction = 'mean')
        else:
            loss = self.nllloss(logits, labels)
            loss = torch.masked_select(loss, mask).mean()
        return loss