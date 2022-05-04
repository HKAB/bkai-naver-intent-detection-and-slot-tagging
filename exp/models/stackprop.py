from scipy import stats
import torch
import torch.nn as nn
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel, XLMRobertaConfig
from .modules import *

class StackPropagation(nn.Module):

    def __init__(self, pretrained_model, hidden_dim, num_intent, num_slot, dropout = 0.2, max_len = None) -> None:
        super().__init__()

        self.max_len = max_len
        config = XLMRobertaConfig.from_pretrained(pretrained_model)
        self.xlmr = XLMRobertaModel.from_pretrained(pretrained_model)
        self.intent_dec = LSTMDecoder(config.hidden_size, hidden_dim, num_intent, dropout = dropout, max_len = max_len)
        slot_input_dim = config.hidden_size + num_intent
        self.slot_dec = LSTMDecoder(slot_input_dim, hidden_dim, num_slot, dropout = dropout, max_len = max_len)

    def forward(self, input_ids, att_mask, len_list):
        trf_output = self.xlmr(input_ids, attention_mask = att_mask)
        embedding, pooler = trf_output.last_hidden_state, trf_output.pooler_output

        intent_pred = self.intent_dec(embedding, len_list)
        slot_inputs = torch.cat([embedding, intent_pred], dim = -1)

        slot_pred = self.slot_dec(slot_inputs, len_list)
        
        return intent_pred, slot_pred

    def get_loss(self, intent_logits, slot_logits, intent_labels, slot_labels, mask, intent_coeff = 0.5):
        intent_labels = intent_labels.unsqueeze(1).expand(-1, self.max_len)
        intent_loss = self.intent_dec.get_loss(intent_logits, intent_labels, mask)
        slot_loss = self.slot_dec.get_loss(slot_logits, slot_labels, mask)
        return intent_coeff * intent_loss + (1 - intent_coeff) * slot_loss, intent_loss, slot_loss

    def get_intent(self, seq_pred, mask):
        intent_pred = []
        # print(mask, seq_pred)
        for sent_pred, sent_mask in zip(seq_pred, mask):
            true_sent = [label for label, m in zip(sent_pred, sent_mask) if m]
            intent_pred.append(stats.mode(true_sent)[0][0])
        return torch.tensor(intent_pred)

    def predict(self, text, att_mask, slots, len_list, device, perm_idx = None, intents = None):
        intent_out, slot_out = self.forward(text, att_mask, len_list)
        intent_out = self.intent_dec.crf.decode(intent_out)
        seq_mask = slots >= 0
        intent_out = self.get_intent(intent_out, seq_mask).to(device)
        slot_out = self.slot_dec.crf.decode(slot_out)
        return intent_out, slot_out

class StackPropagationAtt(nn.Module):

    def __init__(self, pretrained_model, hidden_dim, num_intent, num_slot, dropout = 0.2, max_len = None) -> None:
        super().__init__()

        self.max_len = max_len
        config = XLMRobertaConfig.from_pretrained(pretrained_model)
        self.xlmr = XLMRobertaModel.from_pretrained(pretrained_model)

        self.intent_lstm = LSTMDecoder(config.hidden_size, hidden_dim // 2, hidden_dim, dropout = dropout, max_len = max_len, use_linear=False)
        self.intent_att = nn.Linear(hidden_dim, 1)
        self.intent_linear = nn.Sequential(nn.Dropout(dropout),
                                                nn.Linear(hidden_dim, num_intent))
        self.intent_loss = nn.CrossEntropyLoss()

        slot_input_dim = config.hidden_size + hidden_dim
        self.slot_dec = LSTMDecoder(slot_input_dim, hidden_dim, num_slot, dropout = dropout, max_len = max_len)

    def forward(self, input_ids, att_mask, len_list):
        trf_output = self.xlmr(input_ids, attention_mask = att_mask)
        embedding, pooler = trf_output.last_hidden_state, trf_output.pooler_output

        intent_embedding = self.intent_lstm(embedding, len_list)
        intent_pred = self.intent_classifier(intent_embedding, att_mask)

        slot_inputs = torch.cat([embedding, intent_embedding], dim = -1)

        slot_pred = self.slot_dec(slot_inputs, len_list)
        
        return intent_pred, slot_pred

    def intent_classifier(self, intent_seq_emb, att_mask):
        """
            intent_emb: batchsize x max_len x embedding_dim
            att_mask: batchsize x max_len
        """
        intent_score = self.intent_att(intent_seq_emb).squeeze(2)
        intent_score[~att_mask.bool()] = float('-inf')
        # print(att_mask, intent_score)
        intent_score = torch.softmax(intent_score, dim = 1)
        intent_emb = (intent_seq_emb.transpose(1, 2) @ intent_score.unsqueeze(-1)).squeeze(-1)

        intent_logits = self.intent_linear(intent_emb)
        return intent_logits


    def get_loss(self, intent_logits, slot_logits, intent_labels, slot_labels, mask, intent_coeff = 0.5):
        intent_loss = self.intent_loss(intent_logits, intent_labels)
        slot_loss = self.slot_dec.get_loss(slot_logits, slot_labels, mask)
        return intent_coeff * intent_loss + (1 - intent_coeff) * slot_loss, intent_loss, slot_loss

    def predict(self, text, att_mask, slots, len_list, device, perm_idx = None, intents = None):
        intent_out, slot_out = self.forward(text, att_mask, len_list)
        slot_out = self.slot_dec.crf.decode(slot_out)
        intent_out = intent_out.argmax(dim = -1)
        return intent_out, slot_out
