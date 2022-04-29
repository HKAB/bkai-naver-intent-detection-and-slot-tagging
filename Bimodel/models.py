import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF

class WordEmbedding(nn.Module):

    def __init__(self, num_word, emb_dim) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_word, emb_dim)

    def forward(self, input):
        return self.embedding(input)
        
class Encoder(nn.Module):

    def __init__(self, emb_dim, hidden_dim, dropout, max_len) -> None:
        super().__init__()
        assert hidden_dim % 2 == 0
        self.lstm = nn.LSTM(emb_dim, hidden_dim // 2, num_layers = 2, bidirectional = True, batch_first = True)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(self, input, len_list):
        input = pack_padded_sequence(input, len_list, batch_first = True)
        seq_out, (h_state, c_state) = self.lstm(input)
        seq_out, _ = pad_packed_sequence(seq_out, batch_first = True, total_length = self.max_len)
        seq_out = self.dropout(seq_out)
        return seq_out#, h_state, c_state
    

class IntentDecoder(nn.Module):
    
    def __init__(self, hidden_dim, num_label, dropout) -> None:
        super().__init__()
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim // 2, bidirectional = True, batch_first = True)
        self.linear = nn.Linear(hidden_dim, num_label)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, len_list):
        input = pack_padded_sequence(input, len_list, batch_first = True)
        _, (h_state, c_state) = self.lstm(input)
        h_state = torch.cat([h_state[-1, :, :], h_state[-2, :, :]], dim = 1)
        h_state = self.dropout(h_state)
        out = self.linear(h_state)
        return out

class IntentModel(nn.Module):

    def __init__(self, embedding_model, emb_dim, hidden_dim, num_intent, dropout, max_len) -> None:
        super().__init__()
        self.embedding_model = embedding_model
        self.enc = Encoder(emb_dim, hidden_dim, dropout, max_len)
        self.dec = IntentDecoder(hidden_dim, num_intent, dropout)
        self.loss = nn.CrossEntropyLoss()

    def encode(self, text, len_list):
        embedding = self.embedding_model(text)
        h_feat = self.enc(embedding, len_list)
        return h_feat

    def decode(self, main_feat, aux_feat, len_list):
        h_feat = torch.cat([main_feat, aux_feat], dim = -1)
        out = self.dec(h_feat, len_list)
        return out

    def get_loss(self, output, labels):
        loss = self.loss(output, labels)
        return loss

class SlotDecoder(nn.Module):

    def __init__(self, hidden_dim, num_slot, dropout, max_len) -> None:
        super().__init__()
        self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim // 2, bidirectional = True, batch_first = True)
        self.linear = nn.Linear(hidden_dim, num_slot)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

    def forward(self, input, len_list):
        input = pack_padded_sequence(input, len_list, batch_first = True)
        seq_out, _ = self.lstm(input)
        seq_out, _ = pad_packed_sequence(seq_out, batch_first = True, total_length = self.max_len)
        seq_out = self.dropout(seq_out)
        output = self.linear(seq_out)
        return output

class SlotModel(nn.Module):
    
    def __init__(self, embedding_model, emb_dim, hidden_dim, num_slot, dropout, max_len) -> None:
        super().__init__()
        self.embedding_model = embedding_model
        self.enc = Encoder(emb_dim, hidden_dim, dropout, max_len)
        self.dec = SlotDecoder(hidden_dim, num_slot, dropout, max_len)
        self.crf = CRF(num_slot, batch_first = True)

    def encode(self, text, len_list):
        embedding = self.embedding_model(text)
        h_feat = self.enc(embedding, len_list)
        return h_feat

    def decode(self, main_feat, aux_feat, len_list):
        h_feat = torch.cat([main_feat, aux_feat], dim = -1)
        out = self.dec(h_feat, len_list)
        return out

    def get_loss(self, output, labels, mask):
        loss = - self.crf(output, labels, mask.byte(), reduction = 'mean')
        return loss