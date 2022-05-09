import numpy as np
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


class Attention(nn.Module):

    def __init__(self, query_dim, value_dim, hidden_dim, out_dim, dropout = 0.) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query_mlp = nn.Sequential(nn.Linear(query_dim, hidden_dim),
                                        nn.Dropout(dropout),
                                        nn.ReLU())
        self.value_mlp = nn.Sequential(nn.Linear(value_dim, hidden_dim),
                                        nn.Dropout(dropout),
                                        nn.ReLU())
        self.out_mlp = nn.Sequential(nn.Linear(hidden_dim, out_dim),
                                        nn.Dropout(dropout),
                                        nn.ReLU())

    def forward(self, queries, values, value_mask):
        queries = self.query_mlp(queries) # bs x 1 x hidden_dim
        values = self.value_mlp(values) # bs x seq_len
        att_scores = torch.bmm(queries, values.transpose(1, 2).contiguous()) / np.sqrt(self.hidden_dim)
        att_scores.transpose(1, 2)[value_mask == 0] = float('-inf') # bs x 1 x seq_len
        att_weights = torch.softmax(att_scores, dim = 2)
        att_output = torch.bmm(att_weights, values)
        output = self.out_mlp(att_output)
        return output

class IDSFAttention(nn.Module):
    
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.value_mlp_in = nn.Linear(in_dim, hidden_dim, bias = False)
        self.proj = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim),
                                 nn.ReLU())
                                   
    
    def forward(self, queries, values, query_mask):
        """
            queries: bs x seq_len x slot_hidden_dim
            values: bs  x 1 x num_intent
            Returns:
                output: bs x seq_len x slot_hidden_dim
            
        """
        values = self.value_mlp_in(values)
        att_scores = torch.bmm(queries, values.transpose(1, 2).contiguous()) / np.sqrt(self.hidden_dim)
        att_scores[query_mask == 0] = float('-inf') # bs x seq_len x 1
        att_weights = torch.softmax(att_scores, dim = 1)
        att_output = torch.bmm(att_weights, values)
        mix = torch.cat((queries, att_output), dim = 2)
        output = self.proj(mix)
        return output
    
class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, num_intent = None, use_attention = False, hidden_dim = 200, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)
        self.use_attention = use_attention
        if use_attention and num_intent is not None:
            self.slot_linear_in = nn.Linear(input_dim, hidden_dim)
            self.attention = IDSFAttention(num_intent, hidden_dim)
            self.slot_linear_out = nn.Linear(hidden_dim, num_slot_labels)

    def forward(self, x, intent_context = None, mask = None):
        if not self.use_attention :
            x = self.dropout(x)
            return self.linear(x)
        else:
            assert intent_context is not None and mask is not None
            x = self.slot_linear_in(x)
            intent_prob = torch.softmax(intent_context, dim = 1)
            intent_prob = intent_prob.unsqueeze(1)
            output = self.attention(x, intent_prob, mask)
            output = self.slot_linear_out(output)
            output = self.dropout(output)
            return output
        
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
            self.loss = nn.CrossEntropyLoss(reduction = 'mean')

    def forward(self, inputs, len_list):

        packed_inputs = pack_padded_sequence(inputs, len_list, batch_first = True)
        seq_out, (h_state, c_state) = self.lstm(packed_inputs)
        seq_out, _ = pad_packed_sequence(seq_out, batch_first = True, total_length = self.max_len)

        if self.linear is not None:
            seq_out = self.linear(seq_out)
        return seq_out

    def get_loss(self, logits, labels, mask):
        if self.use_crf:
            loss = - self.crf(logits, labels, mask.bool(), reduction = 'mean')
        else:
            mask = mask.bool()
            loss = self.loss(logits[mask, :], labels[mask])
        return loss
    
    def predict(self, logits):
        if self.use_crf:
            return self.crf.decode(logits)
        else:
            return logits.argmax(dim = -1).cpu()
        
