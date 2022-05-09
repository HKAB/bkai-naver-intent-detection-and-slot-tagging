import torch
import torch.nn as nn
from torchcrf import CRF
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaModel, XLMRobertaConfig
from .modules import IntentClassifier, SlotClassifier, Attention

class CrossAttIntent(nn.Module):

    def __init__(self, pretrained, num_intent, num_slot, dropout, hidden_dim = 200) -> None:
        super().__init__()
        config = XLMRobertaConfig.from_pretrained(pretrained)
        self.xlmr = XLMRobertaModel.from_pretrained(pretrained)
        intent_att_dim = config.hidden_size + num_slot
        self.intent_att = Attention(config.hidden_size, intent_att_dim, hidden_dim, num_intent, dropout)
        # self.intent_dec = IntentClassifier(config.hidden_size, num_intent, dropout_rate = dropout)
        self.slot_dec = SlotClassifier(config.hidden_size, num_slot, num_intent = num_intent, dropout_rate = dropout)
        self.crf = CRF(num_slot, batch_first = True)
        self.intent_loss = nn.CrossEntropyLoss()

    def forward(self, input_ids, att_mask, len_list = None):
        trf_output = self.xlmr(input_ids, attention_mask = att_mask)
        embedding, pooler = trf_output.last_hidden_state, trf_output.pooler_output

        pooler = pooler.unsqueeze(1)
        
        slot_logits = self.slot_dec(embedding, att_mask)
        intent_att_inputs = torch.cat([embedding, slot_logits], dim = -1)
        intent_logits = self.intent_att(pooler, intent_att_inputs, att_mask).squeeze(1)
        return intent_logits, slot_logits

    def get_loss(self, intent_logits, slot_logits, intent_labels, slot_labels, mask, intent_coeff = 0.5):
        intent_loss = self.intent_loss(intent_logits, intent_labels)
        slot_loss = - self.crf(slot_logits, slot_labels, mask.bool(), reduction = 'mean')
        # intent_coeff = (intent_labels.size(0) / mask.sum()).detach()
        total_loss = intent_loss * intent_coeff + (1 - intent_coeff) * slot_loss
        return total_loss, intent_loss, slot_loss

    def predict(self, text, att_mask, slots, len_list, device, perm_idx = None, intents = None, return_logits = False):
        intent_logits, slot_out = self.forward(text, att_mask)
        slot_out = self.crf.decode(slot_out)
        intent_out = intent_logits.argmax(dim = -1)
        if return_logits:
            return intent_out, slot_out, intent_logits
        else:
            return intent_out, slot_out